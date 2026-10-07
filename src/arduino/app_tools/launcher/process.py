# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""A worker process seen from the supervisor: spawn, control channel, output, stop."""

import sys

if sys.platform == "win32":
    raise ImportError("arduino-app-launcher needs a POSIX system")

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import enum  # noqa: E402
import itertools  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

from . import protocol  # noqa: E402
from .appinfo import AppInfo  # noqa: E402
from .protocol import Message  # noqa: E402

WORKER_PY = Path(__file__).with_name("worker.py")

_worker_ids = itertools.count(1)


class WorkerState(enum.StrEnum):
    SPAWNED = "spawned"
    WARMING = "warming"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    EXITED = "exited"


class WorkerError(Exception):
    """The worker died, or did not answer in time, before reaching the state asked for."""


@dataclass
class WorkerSpec:
    """Everything a worker is started with; the fingerprint says when a spec is stale."""

    app: AppInfo
    interpreter: str
    env: dict[str, str]
    fingerprint: str
    modules: list[str] = field(default_factory=list[str])
    link: str = "/app"


@dataclass
class ExitInfo:
    returncode: int | None
    signal: str | None
    killed: bool = False
    stop_ms: float | None = None

    def as_dict(self) -> Message:
        return {"exit_code": self.returncode, "signal": self.signal, "killed": self.killed, "stop_ms": self.stop_ms}


OutputCallback = Callable[["Worker", bytes], None]
EventCallback = Callable[["Worker", Message], None]
ExitCallback = Callable[["Worker"], None]


class Worker:
    """One worker process, bound to one app venv, used for at most one run of the app."""

    def __init__(self, spec: WorkerSpec, on_output: OutputCallback, on_event: EventCallback, on_exit: ExitCallback) -> None:
        self.id = next(_worker_ids)
        self.spec = spec
        self.state = WorkerState.SPAWNED
        self.mode = "warm"
        self.created = time.time()
        self.pid: int | None = None
        self.hello: Message = {}
        self.ready_info: Message = {}
        self.started_info: Message = {}
        self.app_run_t: float | None = None
        self.suspended = False
        self.exit: ExitInfo | None = None
        self._on_output = on_output
        self._on_event = on_event
        self._on_exit = on_exit
        self._proc: asyncio.subprocess.Process | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._ready = asyncio.Event()
        self._started = asyncio.Event()
        self._leader_exited = asyncio.Event()
        self._exited = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def app_name(self) -> str:
        return self.spec.app.name

    @property
    def alive(self) -> bool:
        return not self._leader_exited.is_set()

    def describe(self) -> Message:
        return {
            "id": self.id,
            "pid": self.pid,
            "state": str(self.state),
            "mode": self.mode,
            "age_s": round(time.time() - self.created, 1),
            "warm_ms": self.ready_info.get("warm_ms"),
            "uss_kb": read_uss_kb(self.pid) if self.pid and self.alive else None,
            "failed_imports": len(self.ready_info.get("failed", [])),
            "suspended": self.suspended,
        }

    async def spawn(self, mode: str) -> None:
        """Start the process; in "warm" mode it begins importing, in "immediate" mode it waits for run().

        Raises:
            WorkerError: if the process cannot be started.
        """
        self.mode = mode
        parent, child = socket.socketpair()
        argv = [
            self.spec.interpreter,
            str(WORKER_PY),
            "--control-fd",
            str(child.fileno()),
            "--app",
            str(self.spec.app.path),
            "--link",
            self.spec.link,
        ]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.spec.app.path),
                env=self.spec.env,
                pass_fds=(child.fileno(),),
                start_new_session=True,
            )
        except OSError as e:
            parent.close()
            raise WorkerError(f"cannot start {self.spec.interpreter}: {e}") from e
        finally:
            child.close()
        self.pid = self._proc.pid
        reader, self._writer = await asyncio.open_unix_connection(sock=parent)
        self._tasks.append(asyncio.create_task(self._read_events(reader)))
        self._tasks.append(asyncio.create_task(self._read_output()))
        self._tasks.append(asyncio.create_task(self._wait_exit()))
        if mode == "warm":
            self.state = WorkerState.WARMING
            self._send({"cmd": "warm", "modules": self.spec.modules})

    def _send(self, message: Message) -> None:
        if self._writer is not None and not self._writer.is_closing():
            self._writer.write(protocol.encode(message))

    async def _read_events(self, reader: asyncio.StreamReader) -> None:
        splitter = protocol.LineSplitter()
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            try:
                lines = splitter.feed(chunk)
            except protocol.ProtocolError:
                continue
            for line in lines:
                try:
                    message = protocol.decode(line)
                except protocol.ProtocolError:
                    continue
                self._handle_event(message)

    def _handle_event(self, message: Message) -> None:
        event = message.get("event")
        if event == "hello":
            self.hello = message
        elif event == "ready":
            self.ready_info = message
            if self.state == WorkerState.WARMING:
                self.state = WorkerState.READY
            self._ready.set()
        elif event == "started":
            self.started_info = message
            self.state = WorkerState.RUNNING
            self._started.set()
        elif event == "app_run":
            self.app_run_t = float(message.get("t", time.time()))
        self._on_event(self, message)

    async def _read_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            chunk = await self._proc.stdout.read(65536)
            if not chunk:
                return
            self._on_output(self, chunk)

    async def _wait_exit(self) -> None:
        assert self._proc is not None
        returncode = await self._proc.wait()
        if self.exit is None:
            self.exit = exit_info(returncode)
        self._leader_exited.set()
        # Let the output and events already written reach their readers first; a child the app left
        # behind may hold the output pipe open, hence the bound
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.gather(*self._tasks[:2], return_exceptions=True), timeout=1.0)
        self.state = WorkerState.EXITED
        self._exited.set()
        if self._writer is not None:
            self._writer.close()
        self._on_exit(self)

    async def wait_ready(self, timeout: float) -> None:
        """Wait for the warm-up to complete, or to be cut short by run().

        Raises:
            WorkerError: if the worker exits first or the timeout expires.
        """
        await self._wait_for((self._ready, self._started), timeout, "ready")

    async def _wait_for(self, events: tuple[asyncio.Event, ...], timeout: float, what: str) -> None:
        waiters = [asyncio.create_task(event.wait()) for event in (*events, self._leader_exited)]
        try:
            done, _ = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        if any(event.is_set() for event in events):
            return
        if not done:
            raise WorkerError(f"worker {self.pid} of {self.app_name} not {what} after {timeout:.0f}s")
        raise WorkerError(f"worker {self.pid} of {self.app_name} exited before being {what} ({self.exit.as_dict() if self.exit else {}})")

    async def run(self, timeout: float, streamlit: bool) -> Message:
        """Tell the worker to run the app; returns its `started` event.

        Raises:
            WorkerError: if the worker exits before starting the app or does not answer in time.
        """
        self.resume()
        self._send({"cmd": "run", "streamlit": streamlit})
        await self._wait_for((self._started,), timeout, "started")
        return self.started_info

    def suspend(self) -> bool:
        """Stop the worker with SIGSTOP, to leave the CPU to an app starting; returns whether it did."""
        if not self.alive or self.suspended:
            return False
        self._killpg(signal.SIGSTOP)
        self.suspended = True
        return True

    def resume(self) -> None:
        if self.suspended:
            self.suspended = False
            self._killpg(signal.SIGCONT)

    async def quit(self) -> None:
        """End a worker that never ran its app."""
        self.resume()
        if not self.alive:
            return
        self._send({"cmd": "quit"})
        if not await self._wait_event(self._leader_exited, 1.0):
            self._killpg(signal.SIGKILL)
            await self._wait_event(self._leader_exited, 2.0)
        await self._wait_event(self._exited, 2.0)

    async def stop(self, timeout: float) -> ExitInfo:
        """Stop the app the way docker stop does: SIGTERM, then SIGKILL to the whole process group after the timeout.

        Returns once the leader has exited and nothing of its process group is left, so that whatever it held,
        devices, ports, the router connection, is free again.
        """
        started = time.monotonic()
        killed = False
        self.resume()
        if self.alive:
            self.state = WorkerState.STOPPING
            self._signal_leader(signal.SIGTERM)
            if not await self._wait_event(self._leader_exited, timeout):
                killed = True
                self._killpg(signal.SIGKILL)
                await self._wait_event(self._leader_exited, 5.0)
        # Children the app left behind go with it
        self._killpg(signal.SIGKILL)
        await self._wait_group_gone(1.0)
        await self._wait_event(self._exited, 2.0)
        info = self.exit or ExitInfo(None, None)
        info.killed = info.killed or killed
        info.stop_ms = round((time.monotonic() - started) * 1000, 1)
        self.exit = info
        return info

    @staticmethod
    async def _wait_event(event: asyncio.Event, timeout: float) -> bool:
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except TimeoutError:
            return False

    def _signal_leader(self, sig: signal.Signals) -> None:
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(self.pid, sig)

    def _killpg(self, sig: signal.Signals) -> None:
        if self.pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pid, sig)

    async def _wait_group_gone(self, timeout: float) -> None:
        if self.pid is None:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(self.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            await asyncio.sleep(0.02)


def exit_info(returncode: int) -> ExitInfo:
    """Exit code and signal of a process, from asyncio's returncode (negative for a signal)."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = str(-returncode)
        return ExitInfo(returncode=128 - returncode, signal=name)
    return ExitInfo(returncode=returncode, signal=None)


def read_uss_kb(pid: int | None) -> int | None:
    """Private memory of a process, from /proc/<pid>/smaps_rollup."""
    if pid is None:
        return None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            return sum(int(line.split()[1]) for line in f if line.startswith(("Private_Clean:", "Private_Dirty:")))
    except (OSError, ValueError, IndexError):
        return None


def read_meminfo_kb() -> dict[str, int]:
    """MemTotal and MemAvailable, in kB."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return values
