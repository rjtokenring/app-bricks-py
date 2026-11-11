# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""HTTP API of the Kokoro text-to-speech runner.

GET  /health      200 {"status": "ok"}: the server listens only once the model is loaded.
GET  /voices      200 {"voices": [{"name": "af_heart", "language": "en-us"}, ...],
                       "default_voice": "af_heart", "sample_rate": 24000}
POST /synthesize  body {"text": str, "voice"?: str, "speed"?: number, "sample_rate"?: int}
                  200 audio/pcm: mono 16-bit little-endian PCM at sample_rate, streamed one
                  sentence at a time. Closing the connection stops the synthesis of the
                  sentences still to come.

Errors are JSON: {"error": {"message": str}}, with status 400, 404 or 500.
"""

import json
import logging
import signal
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from synthesizer import DEFAULT_VOICE, MODEL_SAMPLE_RATE, Synthesizer, UnknownVoiceError

logger = logging.getLogger("kokoro-tts")

MAX_BODY_BYTES = 1024 * 1024
SPEED_RANGE = (0.5, 2.0)
SAMPLE_RATE_RANGE = (8000, 48000)


class BadRequest(ValueError):
    """The request cannot be served as it is: the message says what to change."""


def parse_request(body: bytes) -> dict[str, Any]:
    """Validate a /synthesize body and return the arguments of Synthesizer.synthesize()."""
    try:
        data = json.loads(body)
    except ValueError:
        raise BadRequest("The request body is not valid JSON.")
    if not isinstance(data, dict):
        raise BadRequest("The request body must be a JSON object.")

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise BadRequest("'text' must be a non-empty string.")
    voice = data.get("voice") or DEFAULT_VOICE
    if not isinstance(voice, str):
        raise BadRequest("'voice' must be a string.")
    speed = data.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
        raise BadRequest(f"'speed' must be a number between {SPEED_RANGE[0]} and {SPEED_RANGE[1]}.")
    sample_rate = data.get("sample_rate", MODEL_SAMPLE_RATE)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not SAMPLE_RATE_RANGE[0] <= sample_rate <= SAMPLE_RATE_RANGE[1]:
        raise BadRequest(f"'sample_rate' must be an integer between {SAMPLE_RATE_RANGE[0]} and {SAMPLE_RATE_RANGE[1]}.")
    return {"text": text, "voice": voice, "speed": float(speed), "sample_rate": sample_rate}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    synthesizer: Synthesizer

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/voices":
            voices = [{"name": name, "language": language} for name, language in self.synthesizer.voices.items()]
            self._send_json(HTTPStatus.OK, {"voices": voices, "default_voice": DEFAULT_VOICE, "sample_rate": MODEL_SAMPLE_RATE})
        else:
            self._send_error(HTTPStatus.NOT_FOUND, f"No such endpoint: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/synthesize":
            self._send_error(HTTPStatus.NOT_FOUND, f"No such endpoint: {self.path}")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"The request body is larger than {MAX_BODY_BYTES} bytes.")
            return
        try:
            arguments = parse_request(self.rfile.read(length))
            speech = self.synthesizer.synthesize(**arguments)
            # The first sentence is generated before the status line, so that a failure can still be reported
            first = next(speech, b"")
        except (BadRequest, UnknownVoiceError) as e:
            self._send_error(HTTPStatus.BAD_REQUEST, str(e))
            return
        except Exception as e:
            logger.exception("Synthesis failed")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Synthesis failed: {e}")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"audio/pcm;rate={arguments['sample_rate']};channels=1")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._write_chunk(first)
            for pcm in speech:
                self._write_chunk(pcm)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            logger.info("Client closed the connection, synthesis stopped")
            self.close_connection = True
        except Exception:
            # The status is gone: an unterminated body tells the client the audio is incomplete
            logger.exception("Synthesis failed")
            self.close_connection = True

    def _write_chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": {"message": message}})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format, *args)


def make_server(synthesizer: Synthesizer, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """An HTTP server speaking with synthesizer, bound and not yet serving."""
    handler = type("KokoroHandler", (Handler,), {"synthesizer": synthesizer})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # As PID 1 of the container the process has no default action for SIGTERM: without a handler a stop waits for the kill
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    models = Path("/models")
    synthesizer = Synthesizer.load(models / "kokoro.onnx", models)
    # One run before listening: the first request does not pay for the lazy initializations of the session
    for _ in synthesizer.synthesize("Ready."):
        pass
    server = make_server(synthesizer, 9100)
    logger.info("Kokoro TTS ready on port 9100, %d voices", len(synthesizer.voices))
    server.serve_forever()


if __name__ == "__main__":
    main()
