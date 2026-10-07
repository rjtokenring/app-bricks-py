# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Synchronous client of the arduino-app-launcher control socket."""

import itertools
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import protocol
from .protocol import Message


class LauncherUnavailable(Exception):
    """Nothing listens on the control socket."""


class Client:
    def __init__(self, socket_path: Path, timeout: float | None = 60.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self._ids = itertools.count(1)

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(str(self.socket_path))
        except (FileNotFoundError, ConnectionRefusedError) as e:
            sock.close()
            raise LauncherUnavailable(f"arduino-app-launcher is not listening on {self.socket_path}: {e}") from e
        return sock

    def call(self, cmd: str, **args: Any) -> Message:
        """Send one request and return its reply."""
        with self._connect() as sock:
            request: Message = {"v": protocol.PROTOCOL_VERSION, "id": next(self._ids), "cmd": cmd, **args}
            sock.sendall(protocol.encode(request))
            for message in _messages(sock):
                return message
        raise LauncherUnavailable("the launcher closed the connection without replying")

    def stream(self, cmd: str, **args: Any) -> Iterator[Message]:
        """Send a streaming request (`events`, `logs` with follow) and yield what follows its first reply."""
        sock = self._connect()
        sock.settimeout(None)
        try:
            sock.sendall(protocol.encode({"v": protocol.PROTOCOL_VERSION, "id": next(self._ids), "cmd": cmd, **args}))
            messages = _messages(sock)
            first = next(messages, None)
            if first is None or not first.get("ok"):
                raise LauncherUnavailable(f"{cmd} refused: {first}")
            yield from messages
        finally:
            sock.close()


def _messages(sock: socket.socket) -> Iterator[Message]:
    splitter = protocol.LineSplitter()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        for line in splitter.feed(chunk):
            if line.strip():
                yield protocol.decode(line)
