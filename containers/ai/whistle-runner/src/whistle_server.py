# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Whistle speech-to-text over HTTP, through the C API of the Cactus runtime (cactus.h).

The runtime keeps one process-wide model state, so the server answers one request at a time.

    GET  /health      -> {"status": "ok"} once the model is loaded
    POST /transcribe  {"pcm": base64 float32 little-endian samples, 16 kHz mono,
                       "language": ISO 639-1 code or null to detect it,
                       "keywords": words or phrases to favour, or null}
                      -> the runtime result: "text", "language" and timing fields
"""

import argparse
import base64
import binascii
import ctypes
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
MAX_SAMPLES = 30 * SAMPLE_RATE  # one runtime call transcribes at most 30 s
MAX_BODY = 4 * MAX_SAMPLES * 4 // 3 + 64 * 1024  # base64 of 30 s of float32, plus the other fields
OUT_CAPACITY = 1 << 16
DEFAULT_LIB = Path("/usr/local/lib/cactus/libcactus.so")
DEFAULT_MODEL = Path("/models/whistle.cact")


class WhistleEngine:
    """The Whistle model loaded in the Cactus runtime."""

    def __init__(self, lib_path: Path, model_path: Path) -> None:
        """
        Load the runtime library and the model.

        Raises:
            FileNotFoundError: If one of the files is missing.
            OSError: If the runtime library cannot be loaded.
            RuntimeError: If the runtime rejects the model.
        """
        for path in (lib_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
        lib = ctypes.CDLL(str(lib_path))
        self._load = lib.cactus_load
        self._load.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong]
        self._load.restype = ctypes.c_int
        self._last_error = lib.cactus_last_error
        self._last_error.argtypes = []
        self._last_error.restype = ctypes.c_char_p
        self._transcribe = lib.cactus_transcribe
        self._transcribe.argtypes = [
            ctypes.c_char_p,  # pcm: const float*
            ctypes.c_int,  # samples
            ctypes.c_char_p,  # language, NULL to detect it
            ctypes.c_char_p,  # keywords, one per line, or NULL
            ctypes.c_int,  # word timestamps
            ctypes.c_char_p,  # out: JSON result
            ctypes.c_int,  # out capacity
        ]
        self._transcribe.restype = ctypes.c_int
        self._out = ctypes.create_string_buffer(OUT_CAPACITY)
        # Kept alive with the engine: the runtime may keep pointers into the buffer it loaded
        self._model = model_path.read_bytes()
        if self._load(self._model, len(self._model)) < 0:
            raise RuntimeError(f"The runtime rejected {model_path}: {self._error()}")

    def _error(self) -> str:
        message = self._last_error()
        return message.decode(errors="replace") if message else "unknown error"

    def transcribe(self, pcm: bytes, language: str | None = None, keywords: list[str] | None = None) -> dict[str, Any]:
        """
        Transcribe one utterance of float32 little-endian samples, 16 kHz mono. Audio past 30 s is ignored.

        Raises:
            RuntimeError: If the runtime fails.
        """
        pcm = pcm[: 4 * MAX_SAMPLES]
        lang = language.encode() if language else None
        words = "\n".join(keywords).encode() if keywords else None
        if self._transcribe(pcm, len(pcm) // 4, lang, words, 0, self._out, OUT_CAPACITY) < 0:
            raise RuntimeError(f"Whistle transcription failed: {self._error()}")
        return json.loads(self._out.value)


class Handler(BaseHTTPRequestHandler):
    engine: WhistleEngine

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"error": f"no {self.path}"})

    def do_POST(self) -> None:
        # Read in full before any answer: a reply over an unread body reaches the client as a reset connection
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/transcribe":
            self._reply(404, {"error": f"no {self.path}"})
            return
        if len(body) > MAX_BODY:
            self._reply(413, {"error": f"more than {MAX_SAMPLES // SAMPLE_RATE} s of audio"})
            return
        try:
            request = json.loads(body)
            pcm = base64.b64decode(request["pcm"], validate=True)
            language, keywords = request.get("language"), request.get("keywords")
            if not isinstance(language, str | None):
                raise ValueError("language must be a string")
            if keywords is not None and not (isinstance(keywords, list) and all(isinstance(k, str) for k in keywords)):
                raise ValueError("keywords must be a list of strings")
        except (ValueError, KeyError, TypeError, binascii.Error) as e:
            self._reply(400, {"error": f"bad request: {e}"})
            return
        try:
            self._reply(200, self.engine.transcribe(pcm, language, keywords))
        except RuntimeError as e:
            self._reply(500, {"error": str(e)})

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args(argv)

    # PID 1 of the container: without a handler SIGTERM is ignored and docker stop waits its grace period out
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Handler.engine = WhistleEngine(args.lib, args.model)
    with HTTPServer(("", args.port), Handler) as server:
        print(f"Whistle ready on port {args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
