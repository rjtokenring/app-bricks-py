# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from arduino.app_bricks.ocr import OCR, OcrError, OcrResult, TextDetection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner_metadata() -> dict:
    """Metadata shaped exactly like the ocr-runner's build_metadata output."""
    return {
        "text": "HELLO\nWORLD",
        "detections": [
            {
                "text": "HELLO",
                "confidence": 0.91,
                "bounding_box_xyxy": [10, 20, 110, 60],
                "polygon": [[10, 20], [110, 20], [110, 60], [10, 60]],
                "type": "horizontal",
            },
            {
                "text": "WORLD",
                "confidence": 0.85,
                "bounding_box_xyxy": [12, 80, 118, 122],
                "polygon": [[14, 84], [118, 80], [116, 118], [12, 122]],
                "type": "free",
            },
        ],
    }


class FakeConnection:
    """Context-manager stand-in for websockets.sync.client.connect()."""

    def __init__(self, uri: str, messages: list[str] | None = None, recv_error: Exception | None = None):
        self.uri = uri
        self.sent: list[str] = []
        self._messages = list(messages or [])
        self._recv_error = recv_error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:
        if self._recv_error is not None:
            raise self._recv_error
        return self._messages.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ocr(monkeypatch: pytest.MonkeyPatch, **kwargs) -> OCR:
    """Return an OCR instance with infrastructure mocked out."""
    fake_compose = {"services": {"ocr": {}}}
    monkeypatch.setattr("arduino.app_bricks.ocr.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_bricks.ocr.resolve_address", lambda host: "127.0.0.1")
    return OCR(**kwargs)


@pytest.fixture()
def ocr(monkeypatch: pytest.MonkeyPatch) -> OCR:
    return _make_ocr(monkeypatch)


def _patch_connect(monkeypatch: pytest.MonkeyPatch, connections: dict[str, FakeConnection]):
    """Route module-level connect(uri) calls to the given fake connections by port."""

    def fake_connect(uri: str, **kwargs):
        port = uri.rsplit(":", 1)[1]
        return connections[port]

    monkeypatch.setattr("arduino.app_bricks.ocr.connect", fake_connect)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_resolves_runner_endpoints(ocr: OCR):
    assert ocr._ws_send_url == "ws://127.0.0.1:5000"
    assert ocr._ws_recv_url == "ws://127.0.0.1:5001"


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def test_encode_image_accepts_numpy_arrays():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded = OCR._encode_image(frame)
    data = base64.b64decode(encoded)
    assert data[:2] == b"\xff\xd8"  # JPEG magic number


def test_encode_image_accepts_encoded_bytes():
    payload = b"\xff\xd8fake-jpeg-bytes"
    assert base64.b64decode(OCR._encode_image(payload)) == payload


def test_encode_image_accepts_file_paths(tmp_path: Path):
    image_file = tmp_path / "image.jpg"
    image_file.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    assert base64.b64decode(OCR._encode_image(image_file)) == b"\xff\xd8fake-jpeg-bytes"
    assert base64.b64decode(OCR._encode_image(str(image_file))) == b"\xff\xd8fake-jpeg-bytes"


def test_encode_image_rejects_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        OCR._encode_image(tmp_path / "missing.jpg")


def test_encode_image_rejects_unsupported_types():
    with pytest.raises(TypeError):
        OCR._encode_image(12345)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def test_parse_metadata_round_trip():
    result = OCR._parse_metadata(_runner_metadata())

    assert isinstance(result, OcrResult)
    assert result.text == "HELLO\nWORLD"
    assert str(result) == "HELLO\nWORLD"
    assert [d.text for d in result.detections] == ["HELLO", "WORLD"]

    first = result.detections[0]
    assert isinstance(first, TextDetection)
    assert first.confidence == pytest.approx(0.91)
    assert first.bounding_box_xyxy == (10, 20, 110, 60)
    assert first.polygon == [(10, 20), (110, 20), (110, 60), (10, 60)]


def test_parse_metadata_handles_empty_payloads():
    result = OCR._parse_metadata({})
    assert result.text == ""
    assert result.detections == []


def test_parse_metadata_skips_malformed_detections_and_rebuilds_text():
    metadata = {
        "detections": [
            {"text": "OK", "confidence": 0.9, "bounding_box_xyxy": [0, 0, 5, 5], "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]]},
            {"confidence": 0.9},  # missing every other field
        ]
    }
    result = OCR._parse_metadata(metadata)
    assert [d.text for d in result.detections] == ["OK"]
    assert result.text == "OK"  # no 'text' key: rebuilt from the detections


# ---------------------------------------------------------------------------
# extract_text request/response
# ---------------------------------------------------------------------------


def test_extract_text_sends_config_then_frame_and_parses_answer(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    answer = json.dumps({"frame": None, "metadata": _runner_metadata()})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    result = ocr.extract_text(b"\xff\xd8fake-jpeg-bytes")

    assert result.text == "HELLO\nWORLD"
    assert len(send_ws.sent) == 2
    # The config travels before the frame, so the runner applies it to this image
    config = json.loads(send_ws.sent[0])
    assert config == {"config": {"allowlist": ""}}
    sent = json.loads(send_ws.sent[1])
    assert base64.b64decode(sent["frame"]) == b"\xff\xd8fake-jpeg-bytes"


def test_extract_text_filters_by_constructor_confidence(monkeypatch: pytest.MonkeyPatch):
    metadata = _runner_metadata()  # HELLO at 0.91, WORLD at 0.85
    answer = json.dumps({"frame": None, "metadata": metadata})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, confidence=0.9)
    result = ocr.extract_text(b"\xff\xd8fake-jpeg-bytes")

    assert [d.text for d in result.detections] == ["HELLO"]
    assert result.text == "HELLO"  # rebuilt from the kept detections


def test_extract_text_confidence_call_override_wins(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    answer = json.dumps({"frame": None, "metadata": _runner_metadata()})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    result = ocr.extract_text(b"\xff\xd8fake-jpeg-bytes", confidence=0.9)

    assert [d.text for d in result.detections] == ["HELLO"]


def test_confidence_is_validated():
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(1.5)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(-0.1)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence(True)
    with pytest.raises(ValueError):
        OCR._validate_min_confidence("0.5")


def test_extract_text_sends_constructor_allowlist(monkeypatch: pytest.MonkeyPatch):
    answer = json.dumps({"frame": None, "metadata": {}})
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=[answer])
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, allowlist="0123456789")
    ocr.extract_text(b"\xff\xd8fake-jpeg-bytes")

    assert json.loads(send_ws.sent[0]) == {"config": {"allowlist": "0123456789"}}


def test_extract_text_allowlist_call_override_wins(monkeypatch: pytest.MonkeyPatch):
    answers = [json.dumps({"frame": None, "metadata": {}})] * 2
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", messages=answers)
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    ocr = _make_ocr(monkeypatch, allowlist="0123456789")
    ocr.extract_text(b"\xff\xd8fake-jpeg-bytes", allowlist="ABC")
    ocr.extract_text(b"\xff\xd8fake-jpeg-bytes", allowlist="")  # lifts the restriction for this call

    configs = [json.loads(send_ws.sent[i]) for i in (0, 2)]
    assert configs[0] == {"config": {"allowlist": "ABC"}}
    assert configs[1] == {"config": {"allowlist": ""}}


def test_extract_text_raises_when_runner_is_unreachable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("arduino.app_bricks.ocr._RETRY_INTERVAL_SEC", 0.01)

    def refuse(uri: str, **kwargs):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr("arduino.app_bricks.ocr.connect", refuse)

    ocr = _make_ocr(monkeypatch, timeout=0.05)
    with pytest.raises(OcrError):
        ocr.extract_text(b"\xff\xd8fake-jpeg-bytes")


def test_extract_text_raises_when_runner_does_not_answer(ocr: OCR, monkeypatch: pytest.MonkeyPatch):
    send_ws = FakeConnection("5000")
    recv_ws = FakeConnection("5001", recv_error=TimeoutError())
    _patch_connect(monkeypatch, {"5000": send_ws, "5001": recv_ws})

    with pytest.raises(OcrError):
        ocr.extract_text(b"\xff\xd8fake-jpeg-bytes")
