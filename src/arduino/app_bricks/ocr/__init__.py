# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils import AppError, Logger, brick
from arduino.app_utils.image.adjustments import compress_to_jpeg

logger = Logger("OCR")

_WS_SEND_PORT = 5000  # model runner websocket input (frames in)
_WS_RECV_PORT = 5001  # model runner websocket output (results out)
_RETRY_INTERVAL_SEC = 1.0


class OcrError(AppError):
    """Raised when the OCR model runner cannot be reached or does not answer in time."""


@dataclass
class TextDetection:
    """One piece of text found in the image.

    Attributes:
        text (str): The recognized text.
        confidence (float): Recognition confidence in [0.0, 1.0].
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) axis-aligned
            box enclosing the text, in image pixel coordinates.
        polygon (list[tuple[int, int]]): The 4 (x, y) vertices of the detected text
            region, in image pixel coordinates, ordered top-left, top-right,
            bottom-right, bottom-left. They differ from the bounding box when the
            text is slanted.
    """

    text: str
    confidence: float
    bounding_box_xyxy: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]


@dataclass
class OcrResult:
    """The text extracted from one image.

    Converting the result to a string (`str(result)` or `print(result)`) yields
    the full extracted text.

    Attributes:
        text (str): Every recognized string joined by newlines, in reading order
            (top to bottom, left to right). Empty when no text was found.
        detections (list[TextDetection]): One entry per piece of text found, in
            reading order, each with its position and confidence.
    """

    text: str
    detections: list[TextDetection] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


@brick
class OCR:
    """Extracts text from images (OCR) using the EasyOCR model runner.

    The brick sends each image to the model runner over WebSocket and returns
    the recognized text with per-detection positions and confidences:

        ocr = OCR()
        result = ocr.extract_text(image)
        print(result.text)
    """

    def __init__(self, confidence: float = 0.3, allowlist: str | None = None, timeout: float = 30.0) -> None:
        """Initialize the OCR brick.

        Args:
            confidence (float): Minimum recognition confidence for a piece of
                text to be reported, in [0.0, 1.0]. Detections below it are dropped
                from the result. Default is 0.3; pass 0.0 to report everything the
                model finds. Can be overridden per call in `extract_text`.
            allowlist (str): Restrict recognition to these characters, e.g.
                "0123456789" to read only digits. Applied by the model runner while
                decoding, so it improves accuracy on constrained text rather than
                just filtering the output. Default is None (no restriction). Can be
                overridden per call in `extract_text`.
            timeout (float): Maximum seconds `extract_text` waits for the model
                runner, connection retries included. Default is 30.

        Raises:
            ValueError: If `confidence` is not a number in [0.0, 1.0].
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._confidence = self._validate_min_confidence(confidence)
        self._allowlist = allowlist
        self._timeout = timeout
        # extract_text calls are serialized so each sent image matches its own answer
        self._lock = threading.Lock()

        infra = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        for k, _ in infra["services"].items():
            self._host = k
            break  # Only one service is expected

        self._host = resolve_address(self._host)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._ws_send_url = f"ws://{self._host}:{_WS_SEND_PORT}"
        self._ws_recv_url = f"ws://{self._host}:{_WS_RECV_PORT}"
        logger.info(f"[{self.__class__.__name__}] Host: {self._host}")

    def extract_text(
        self,
        image: np.ndarray | bytes | str | Path,
        confidence: float | None = None,
        allowlist: str | None = None,
    ) -> OcrResult:
        """Extract the text visible in an image.

        Blocks until the model runner answers. Concurrent calls are serialized,
        so each image is matched with its own result.

        Args:
            image (np.ndarray | bytes | str | Path): The image to read: a numpy
                array in BGR channel order (as returned by `Camera.capture()`),
                the raw bytes of an encoded image file (e.g. JPEG or PNG), or a
                path to an image file.
            confidence (float): Override the constructor's `confidence`
                for this call only. None (default) uses the constructor value.
            allowlist (str): Override the constructor's `allowlist` for this call
                only, e.g. "0123456789" to read only digits from this image. None
                (default) uses the constructor value; pass "" to lift the
                restriction for this call.

        Returns:
            OcrResult: The extracted text, with one `TextDetection` per piece of
                text found. `result.text` is empty when no text was recognized.

        Raises:
            TypeError: If `image` is not one of the supported types.
            ValueError: If `image` could not be encoded, or `confidence` is
                not a number in [0.0, 1.0].
            FileNotFoundError: If `image` is a path that does not exist.
            OcrError: If the model runner cannot be reached or does not answer
                within the configured timeout.
        """
        confidence = self._confidence if confidence is None else self._validate_min_confidence(confidence)
        allowlist = self._allowlist if allowlist is None else allowlist

        payload = json.dumps({"frame": self._encode_image(image)})
        # The runner keeps the allowlist across calls (and clients), so it is
        # restated on every request to make each call self-contained.
        config = json.dumps({"config": {"allowlist": allowlist or ""}})
        with self._lock:
            metadata = self._request(payload, config)
        return self._parse_metadata(metadata, confidence)

    @staticmethod
    def _validate_min_confidence(value: float) -> float:
        """Normalize a minimum-confidence value, rejecting anything outside [0.0, 1.0]."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"confidence must be a number in [0.0, 1.0], got {value!r}")
        return float(value)

    @staticmethod
    def _encode_image(image: np.ndarray | bytes | str | Path) -> str:
        """Turn any supported image input into a base64-encoded JPEG/PNG string."""
        if isinstance(image, np.ndarray):
            jpeg = compress_to_jpeg(image)
            if jpeg is None:
                raise ValueError("The image array could not be encoded to JPEG.")
            data = jpeg.tobytes()
        elif isinstance(image, (bytes, bytearray, memoryview)):
            data = bytes(image)
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            data = path.read_bytes()
        else:
            raise TypeError(f"Unsupported image type {type(image).__name__}: pass a numpy array, encoded image bytes, or a file path.")
        return base64.b64encode(data).decode("utf-8")

    def _request(self, payload: str, config: str) -> dict:
        """Send one frame to the model runner and return the metadata it answers with.

        The configuration is sent before the frame on the same socket, so the runner
        applies it to this image. The result socket is connected before the frame is
        sent, so the runner's broadcast cannot be missed. Connection errors are
        retried until the deadline: the runner container may still be starting up.
        """
        deadline = time.monotonic() + self._timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with connect(self._ws_recv_url) as recv_ws, connect(self._ws_send_url) as send_ws:
                    send_ws.send(config)
                    send_ws.send(payload)
                    try:
                        message = recv_ws.recv(timeout=max(deadline - time.monotonic(), 0.1))
                    except TimeoutError as e:
                        raise OcrError(
                            f"The OCR model runner accepted the image but did not answer within {self._timeout:.0f}s.",
                            hint="Check the OCR container logs; if the board is under heavy load, retry with a larger `timeout`.",
                        ) from e
                    data = json.loads(message)
                    metadata = data.get("metadata")
                    return metadata if isinstance(metadata, dict) else {}
            except (OSError, ConnectionClosed) as e:
                last_error = e
                logger.debug(f"OCR model runner not reachable yet ({e}); retrying...")
                time.sleep(_RETRY_INTERVAL_SEC)
        raise OcrError(
            f"Could not reach the OCR model runner at '{self._host}' within {self._timeout:.0f}s.",
            hint="Check that the OCR container is up and healthy, then try again.",
        ) from last_error

    @staticmethod
    def _parse_metadata(metadata: dict, min_confidence: float = 0.0) -> OcrResult:
        """Build an OcrResult out of the model runner's metadata payload.

        Detections below `min_confidence` are dropped, and the result text is
        rebuilt from the kept detections (they arrive in reading order).
        """
        detections: list[TextDetection] = []
        for det in metadata.get("detections", []):
            try:
                detection = TextDetection(
                    text=str(det["text"]),
                    confidence=float(det["confidence"]),
                    bounding_box_xyxy=tuple(int(v) for v in det["bounding_box_xyxy"]),
                    polygon=[(int(x), int(y)) for x, y in det["polygon"]],
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed detection {det!r}: {e}")
                continue
            if detection.confidence >= min_confidence:
                detections.append(detection)

        return OcrResult(text="\n".join(d.text for d in detections), detections=detections)


__all__ = [
    "OCR",
    "OcrError",
    "OcrResult",
    "TextDetection",
]
