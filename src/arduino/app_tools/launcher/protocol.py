# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Wire format of arduino-app-launcher: one JSON object per line.

The same framing serves the control socket (clients to the supervisor) and the channel between the
supervisor and each worker. worker.py repeats the few lines it needs instead of importing this module:
it runs on the interpreter of the app venv, where this package may be another version or missing.
"""

import json
from typing import Any

PROTOCOL_VERSION = 1
"""Bumped on any incompatible change; the worker reports its own in the hello message."""

MAX_LINE_BYTES = 1 << 20
"""Longest message accepted, newline included."""

Message = dict[str, Any]

# Error codes of the replies
ERR_BAD_REQUEST = "bad_request"
ERR_NOT_FOUND = "not_found"
ERR_PREPARE_FAILED = "prepare_failed"
ERR_SPAWN_FAILED = "spawn_failed"
ERR_TIMEOUT = "timeout"
ERR_BUSY = "busy"
ERR_INTERNAL = "internal"


class ProtocolError(Exception):
    """A line that is not a JSON object, or is too long."""


def encode(message: Message) -> bytes:
    """Serialize a message as a single line, newline included."""
    return json.dumps(message, separators=(",", ":"), default=str).encode() + b"\n"


def decode(line: bytes) -> Message:
    """Parse a line into a message.

    Raises:
        ProtocolError: if the line is not a JSON object.
    """
    try:
        obj = json.loads(line)
    except ValueError as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("not a JSON object")
    return obj  # pyright: ignore[reportUnknownVariableType]


def error(code: str, message: str, **details: Any) -> Message:
    """Build a failed reply."""
    return {"ok": False, "error": {"code": code, "message": message, **details}}


class LineSplitter:
    """Splits a byte stream read in arbitrary chunks into complete lines."""

    def __init__(self, max_line: int = MAX_LINE_BYTES) -> None:
        self._buffer = bytearray()
        self._max_line = max_line

    def feed(self, data: bytes) -> list[bytes]:
        """Add a chunk and return the lines it completed, without their newline.

        Raises:
            ProtocolError: if a line grows past the limit.
        """
        self._buffer += data
        lines: list[bytes] = []
        while True:
            end = self._buffer.find(b"\n")
            if end < 0:
                break
            lines.append(bytes(self._buffer[:end]))
            del self._buffer[: end + 1]
        if len(self._buffer) > self._max_line:
            self._buffer.clear()
            raise ProtocolError(f"line longer than {self._max_line} bytes")
        return lines

    def rest(self) -> bytes:
        """Return and drop the incomplete tail, e.g. the last line of a stream that ended without a newline."""
        tail = bytes(self._buffer)
        self._buffer.clear()
        return tail
