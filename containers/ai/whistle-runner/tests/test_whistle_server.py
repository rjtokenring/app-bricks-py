# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The HTTP API of the Whistle server, with the Cactus runtime replaced by a fake engine."""

import base64
import json
import struct
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import HTTPServer
from pathlib import Path

import pytest

import whistle_server
from whistle_server import MAX_SAMPLES, Handler, WhistleEngine


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None, list[str] | None]] = []
        self.error: str | None = None

    def transcribe(self, pcm: bytes, language: str | None = None, keywords: list[str] | None = None) -> dict:
        self.calls.append((pcm, language, keywords))
        if self.error:
            raise RuntimeError(self.error)
        return {"text": "Accendi la luce.", "language": language or "it"}


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def url(engine: FakeEngine, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(Handler, "engine", engine, raising=False)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _request(url: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url + path, data=body, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _transcribe(url: str, **fields: object) -> tuple[int, dict]:
    return _request(url, "/transcribe", json.dumps(fields).encode())


def _pcm(*samples: float) -> str:
    return base64.b64encode(struct.pack(f"<{len(samples)}f", *samples)).decode()


def test_health(url: str):
    assert _request(url, "/health") == (200, {"status": "ok"})


def test_transcribe_returns_the_runtime_result(url: str):
    assert _transcribe(url, pcm=_pcm(0.0, 0.5)) == (200, {"text": "Accendi la luce.", "language": "it"})


def test_samples_language_and_keywords_reach_the_engine(url: str, engine: FakeEngine):
    _transcribe(url, pcm=_pcm(0.25, -0.5), language="it", keywords=["UNO Q", "Arduino"])
    assert engine.calls == [(struct.pack("<2f", 0.25, -0.5), "it", ["UNO Q", "Arduino"])]


def test_language_and_keywords_are_optional(url: str, engine: FakeEngine):
    _transcribe(url, pcm=_pcm(0.0))
    assert engine.calls[0][1:] == (None, None)


@pytest.mark.parametrize(
    "fields",
    [{}, {"pcm": "not base64!"}, {"pcm": _pcm(0.0), "language": 3}, {"pcm": _pcm(0.0), "keywords": "UNO Q"}, {"pcm": _pcm(0.0), "keywords": [1]}],
)
def test_a_malformed_request_is_400(url: str, engine: FakeEngine, fields: dict):
    status, body = _transcribe(url, **fields)
    assert status == 400 and "error" in body
    assert engine.calls == []


def test_more_than_30_s_of_audio_is_413(url: str, engine: FakeEngine):
    status, _ = _transcribe(url, pcm=_pcm(*[0.0] * (MAX_SAMPLES + 16000)))
    assert status == 413
    assert engine.calls == []


def test_a_runtime_failure_is_500_with_its_message(url: str, engine: FakeEngine):
    engine.error = "Whistle transcription failed: boom"
    assert _transcribe(url, pcm=_pcm(0.0)) == (500, {"error": "Whistle transcription failed: boom"})


def test_unknown_paths_are_404(url: str):
    assert _request(url, "/nope")[0] == 404
    assert _request(url, "/nope", b"{}")[0] == 404


def test_engine_reports_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        WhistleEngine(tmp_path / "libcactus.so", tmp_path / "whistle.cact")


def test_server_exits_when_the_model_cannot_be_loaded(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        whistle_server.main(["--lib", str(tmp_path / "libcactus.so"), "--model", str(tmp_path / "whistle.cact"), "--port", "0"])
