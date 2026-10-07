# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The supervisor of arduino-app-launcher.

It runs on the system interpreter, imports no app code and no heavy library, and keeps for every app one worker:
a process started with the interpreter of that app's venv, warm and waiting (see worker.py). A single app runs at
a time. Starting an app stops the running one, waits until nothing of it is left, points /app at the new app and
tells its worker to run main.py; the stopped app gets a new worker, ready for its next start.
"""

import sys

if sys.platform == "win32":
    raise ImportError("arduino-app-launcher needs a POSIX system")

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Coroutine, Mapping  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from arduino.version import __version__  # noqa: E402

from . import pool, protocol  # noqa: E402
from .appinfo import (  # noqa: E402
    AppInfo,
    AppNotFound,
    app_fingerprint,
    app_interpreter,
    brick_modules,
    compose_env,
    discover_apps,
    load_app,
    needs_prepare,
    record_prepared,
    resolve_app_path,
)
from .imports import DEFAULT_PRELOAD, local_module_names, scan_imports, warm_candidates  # noqa: E402
from .prepare import DEFAULT_RUN_SH, run_prepare  # noqa: E402
from .process import ExitInfo, Worker, WorkerError, WorkerSpec, WorkerState, read_meminfo_kb  # noqa: E402
from .protocol import Message  # noqa: E402

LOG_TAIL_LINES = 2000
WARM_OUTPUT_LIMIT = 256 * 1024
STREAMLIT_MODULES = ("streamlit", "streamlit.web.cli")


class PrepareFailed(Exception):
    """run.sh prepare failed on the app: its dependencies are not installed."""


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Config:
    """Settings of the supervisor, from APP_LAUNCHER_* environment variables."""

    apps_dir: Path = Path("/home/arduino/ArduinoApps")
    socket_path: Path = Path("/run/arduino-app-launcher/launcher.sock")
    app_link: str = "/app"
    """The path apps see as their root; in the image a symlink to current_link."""
    current_link: Path = Path("/home/app/.launcher/current")
    """The symlink the supervisor points at the running app, or at idle_dir."""
    idle_dir: Path = Path("/home/app/.launcher/none")
    state_file: Path = Path("/home/app/.launcher/state.json")
    run_sh: Path = DEFAULT_RUN_SH
    stop_timeout_s: float = 2.5
    """SIGTERM, then SIGKILL to the whole process group after this long."""
    settle_s: float = 0.1
    """Pause between the end of the old app and the start of the new one, for the router to drop its methods."""
    warm_concurrency: int = 2
    warm_timeout_s: float = 300.0
    run_timeout_s: float = 15.0
    replacement_delay_s: float = 5.0
    """How long after an app starts its next worker begins warming, not to compete with the app's own start."""
    start_priority_s: float = 15.0
    """While an app starts, the other warm-ups are suspended, at most this long; 0 disables it."""
    start_priority_tail_s: float = 1.0
    """The warm-ups resume this long after the app reaches App.run(), while its bricks start."""
    mem_reserve_mb: int = 400
    max_failures: int = 3
    preload: tuple[str, ...] = DEFAULT_PRELOAD
    warm_apps: str = pool.ALL_APPS
    interpreter_fallback: str = sys.executable

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Config":
        defaults = cls()
        preload = environ.get("APP_LAUNCHER_PRELOAD")
        return cls(
            apps_dir=Path(environ.get("APP_LAUNCHER_APPS_DIR", str(defaults.apps_dir))),
            socket_path=Path(environ.get("APP_LAUNCHER_SOCKET", str(defaults.socket_path))),
            app_link=environ.get("APP_LAUNCHER_APP_LINK", defaults.app_link),
            current_link=Path(environ.get("APP_LAUNCHER_CURRENT_LINK", str(defaults.current_link))),
            idle_dir=Path(environ.get("APP_LAUNCHER_IDLE_DIR", str(defaults.idle_dir))),
            state_file=Path(environ.get("APP_LAUNCHER_STATE_FILE", str(defaults.state_file))),
            run_sh=Path(environ.get("APP_LAUNCHER_RUN_SH", str(defaults.run_sh))),
            stop_timeout_s=_env_float(environ, "APP_LAUNCHER_STOP_TIMEOUT_S", defaults.stop_timeout_s),
            settle_s=_env_float(environ, "APP_LAUNCHER_SETTLE_S", defaults.settle_s),
            warm_concurrency=max(1, _env_int(environ, "APP_LAUNCHER_WARM_CONCURRENCY", defaults.warm_concurrency)),
            warm_timeout_s=_env_float(environ, "APP_LAUNCHER_WARM_TIMEOUT_S", defaults.warm_timeout_s),
            run_timeout_s=_env_float(environ, "APP_LAUNCHER_RUN_TIMEOUT_S", defaults.run_timeout_s),
            replacement_delay_s=_env_float(environ, "APP_LAUNCHER_REPLACEMENT_DELAY_S", defaults.replacement_delay_s),
            start_priority_s=_env_float(environ, "APP_LAUNCHER_START_PRIORITY_S", defaults.start_priority_s),
            start_priority_tail_s=_env_float(environ, "APP_LAUNCHER_START_PRIORITY_TAIL_S", defaults.start_priority_tail_s),
            mem_reserve_mb=_env_int(environ, "APP_LAUNCHER_MEM_RESERVE_MB", defaults.mem_reserve_mb),
            max_failures=_env_int(environ, "APP_LAUNCHER_MAX_FAILURES", defaults.max_failures),
            preload=tuple(name.strip() for name in preload.split(",") if name.strip()) if preload is not None else defaults.preload,
            warm_apps=environ.get("APP_LAUNCHER_WARM_APPS", defaults.warm_apps),
            interpreter_fallback=environ.get("APP_LAUNCHER_FALLBACK_PYTHON", defaults.interpreter_fallback),
        )


@dataclass
class Slot:
    """An app the supervisor knows, with its standby worker."""

    info: AppInfo
    worker: Worker | None = None
    env: dict[str, str] | None = None
    """Environment from the last start or prepare request, else the one of the CLI compose file."""
    failures: int = 0
    pending_warm: bool = False
    warming: bool = False
    """A warm-up is preparing or spawning its worker: one at a time per app."""
    warm_error: str | None = None
    last_exit: Message | None = None
    prepare_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ActiveRun:
    """The app running now."""

    slot: Slot
    worker: Worker
    run_id: int
    t_recv: float
    stopping: bool = False
    app_run_t: float | None = None
    log: deque[str] = field(default_factory=lambda: deque[str](maxlen=LOG_TAIL_LINES))
    log_splitter: protocol.LineSplitter = field(default_factory=protocol.LineSplitter)


def _log(message: str) -> None:
    print(f"[launcher] {message}", file=sys.stderr, flush=True)


class Supervisor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.slots: dict[str, Slot] = {}
        self.active: ActiveRun | None = None
        self._lifecycle = asyncio.Lock()
        self._warm_queue: asyncio.Queue[str] = asyncio.Queue()
        self._background: set[asyncio.Task[Any]] = set()
        self._subscribers: set[asyncio.Queue[Message | None]] = set()
        self._log_followers: set[asyncio.Queue[str | None]] = set()
        self._warm_output: dict[int, bytearray] = {}
        self._discarded: set[int] = set()
        self._run_ids = itertools.count(1)
        self._recent: list[str] = self._load_recent()
        self._closing = False
        self._shutdown = asyncio.Event()
        self._server: asyncio.Server | None = None
        self.started_at = time.time()
        # Start priority: warm-ups wait on the gate, and the workers warming are suspended, while an app starts
        self._warm_gate = asyncio.Event()
        self._warm_gate.set()
        self._suspended: list[Worker] = []
        self._resume_timer: asyncio.TimerHandle | None = None

    # Lifecycle of the supervisor

    async def serve(self) -> None:
        """Run until SIGTERM, SIGINT or a shutdown request."""
        await self.open()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown.set)
        await self._shutdown.wait()
        await self.close()

    async def open(self) -> None:
        """Prepare the folders, warm the apps and listen on the control socket."""
        self.config.idle_dir.mkdir(parents=True, exist_ok=True)
        self.config.current_link.parent.mkdir(parents=True, exist_ok=True)
        self._point_link(self.config.idle_dir)
        for _ in range(self.config.warm_concurrency):
            self._spawn_task(self._warm_consumer())
        names = self.rescan()
        socket_path = self.config.socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            socket_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(socket_path), limit=protocol.MAX_LINE_BYTES)
        os.chmod(socket_path, 0o660)
        _log(f"listening on {socket_path}, {len(names)} apps in {self.config.apps_dir}: {', '.join(names) or 'none'}")

    async def close(self) -> None:
        """Stop the running app, end every worker and stop listening."""
        self._closing = True
        self._resume_warm_ups()
        if self._server is not None:
            self._server.close()
        async with self._lifecycle:
            await self._stop_active()
        await asyncio.gather(*(self._discard(slot.worker) for slot in self.slots.values() if slot.worker), return_exceptions=True)
        for queue in [*self._subscribers, *self._log_followers]:
            queue.put_nowait(None)
        for task in list(self._background):
            task.cancel()
        with contextlib.suppress(FileNotFoundError):
            self.config.socket_path.unlink()
        _log("stopped")

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def _spawn_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task: asyncio.Task[Any] = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # Apps and their workers

    def rescan(self) -> list[str]:
        """Pick up app folders added, removed or changed, and make sure each selected app has a worker."""
        names = discover_apps(self.config.apps_dir)
        for name in names:
            path = Path(os.path.realpath(self.config.apps_dir / name))
            try:
                info = load_app(path)
            except OSError:
                continue
            if name in self.slots:
                self.slots[name].info = info
            else:
                self.slots[name] = Slot(info=info)
        for name in [name for name in self.slots if name not in names]:
            slot = self.slots[name]
            if self.active is not None and self.active.slot is slot:
                continue
            if slot.worker is not None:
                self._spawn_task(self._discard(slot.worker))
            del self.slots[name]
        for name in pool.warm_order(self._selected(), self._recent):
            self._request_warm(name)
        return names

    def _selected(self) -> list[str]:
        return pool.select_apps_to_warm(self.slots, self.config.warm_apps)

    def _slot_for(self, ref: str) -> Slot:
        path = resolve_app_path(self.config.apps_dir, ref)
        slot = self.slots.get(path.name)
        if slot is None or slot.info.path != path:
            self.rescan()
            slot = self.slots.get(path.name)
            if slot is None:
                raise AppNotFound(f"{ref!r} is not an app of {self.config.apps_dir}")
        slot.info = load_app(path)
        return slot

    def _app_env(self, slot: Slot) -> dict[str, str]:
        env = dict(slot.env) if slot.env is not None else compose_env(slot.info.path)
        env.setdefault("APP_HOME", str(slot.info.path))
        return env

    def _worker_env(self, slot: Slot, interpreter: str) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if not key.startswith("APP_LAUNCHER_") and key not in ("VIRTUAL_ENV", "PYTHONHOME")}
        env["PYTHONUNBUFFERED"] = "1"
        env["APP_SHUTDOWN_GRACE_PERIOD_S"] = str(self.config.stop_timeout_s)
        env.update(self._app_env(slot))
        venv = slot.info.venv_python.parent.parent
        if interpreter == str(slot.info.venv_python):
            env["VIRTUAL_ENV"] = str(venv)
            env["PATH"] = os.pathsep.join(filter(None, [str(venv / "bin"), env.get("PATH")]))
        return env

    def _fingerprint(self, slot: Slot) -> tuple[str, str]:
        interpreter = app_interpreter(slot.info, self.config.interpreter_fallback)
        extra = f"{__version__}|{protocol.PROTOCOL_VERSION}|{','.join(self.config.preload)}|{self.config.stop_timeout_s}"
        return app_fingerprint(slot.info, self._app_env(slot), interpreter, extra), interpreter

    def _warm_modules(self, info: AppInfo) -> list[str]:
        bricks = brick_modules(info.brick_ids)
        if info.streamlit:
            bricks += STREAMLIT_MODULES
        local = local_module_names(info.python_dir, info.bricks_dir)
        return warm_candidates(self.config.preload, bricks, scan_imports([info.python_dir, info.bricks_dir]), local)

    async def _spawn(self, slot: Slot, mode: str) -> Worker:
        fingerprint, interpreter = self._fingerprint(slot)
        spec = WorkerSpec(
            app=slot.info,
            interpreter=interpreter,
            env=self._worker_env(slot, interpreter),
            fingerprint=fingerprint,
            modules=self._warm_modules(slot.info) if mode == "warm" else [],
            link=self.config.app_link,
        )
        worker = Worker(spec, self._on_output, self._on_event, self._on_exit)
        self._warm_output[worker.id] = bytearray()
        await worker.spawn(mode)
        return worker

    async def _discard(self, worker: Worker | None) -> None:
        """End a worker that will never run its app."""
        if worker is None:
            return
        self._discarded.add(worker.id)
        self._warm_output.pop(worker.id, None)
        await worker.quit()

    async def _ensure_prepared(self, slot: Slot, force: bool = False) -> bool:
        """Run run.sh prepare when the app has no venv yet or its dependencies changed; returns whether it ran.

        Raises:
            PrepareFailed: if run.sh prepare fails.
        """
        async with slot.prepare_lock:
            slot.info = load_app(slot.info.path)
            if not force and not needs_prepare(slot.info):
                return False
            name = slot.info.name
            _log(f"preparing {name}")
            splitter = protocol.LineSplitter()

            def relay(chunk: bytes) -> None:
                for line in splitter.feed(chunk):
                    _log(f"[prepare {name}] {line.decode(errors='replace')}")

            started = time.monotonic()
            code = await run_prepare(slot.info, self.config.run_sh, relay)
            if code != 0:
                raise PrepareFailed(f"run.sh prepare of {name} exited with {code}")
            record_prepared(slot.info)
            _log(f"prepared {name} in {time.monotonic() - started:.1f}s")
            return True

    def _request_warm(self, name: str, delay: float = 0.0) -> None:
        """Queue an app for a new worker, unless it has one or is not selected."""
        slot = self.slots.get(name)
        if self._closing or slot is None or slot.pending_warm or slot.warming or name not in self._selected():
            return
        if slot.worker is not None and slot.worker.alive:
            return
        if slot.failures >= self.config.max_failures:
            return
        slot.pending_warm = True
        if delay > 0:
            asyncio.get_running_loop().call_later(delay, self._warm_queue.put_nowait, name)
        else:
            self._warm_queue.put_nowait(name)

    async def _warm_consumer(self) -> None:
        while True:
            name = await self._warm_queue.get()
            slot = self.slots.get(name)
            if slot is None:
                continue
            slot.pending_warm = False
            if self._closing or slot.warming or (slot.worker is not None and slot.worker.alive):
                continue
            await self._warm_gate.wait()
            if self._closing or (slot.worker is not None and slot.worker.alive):
                continue
            slot.warming = True
            try:
                await self._warm(slot)
            finally:
                slot.warming = False

    async def _warm(self, slot: Slot) -> None:
        name = slot.info.name
        running = self.active is not None and self.active.slot is slot
        if running and needs_prepare(slot.info):
            return  # Its venv is in use: prepared, and warmed, when it next starts
        available = read_meminfo_kb().get("MemAvailable")
        if not pool.memory_allows_spawn(available, self.config.mem_reserve_mb):
            slot.warm_error = f"not enough memory ({available} kB available)"
            _log(f"not warming {name}: {slot.warm_error}")
            return
        worker: Worker | None = None
        try:
            if not running:
                await self._ensure_prepared(slot)
            worker = await self._spawn(slot, "warm")
            if slot.worker is not None and slot.worker.alive:
                # A start spawned one meanwhile and left it as the standby: keep that one
                self._spawn_task(self._discard(worker))
                return
            slot.worker = worker
            slot.warming = False
            await worker.wait_ready(self.config.warm_timeout_s)
        except WorkerError as e:
            if worker is not None and (not worker.alive or slot.worker is not worker):
                return  # It died, and _on_exit tells why, or it was taken or replaced meanwhile
            self._warm_failed(slot, str(e), worker)
            return
        except (PrepareFailed, OSError) as e:
            self._warm_failed(slot, str(e), worker)
            return
        if worker.state != WorkerState.READY:
            return  # A start took it before it was done
        slot.failures = 0
        slot.warm_error = None
        info = worker.ready_info
        _log(f"{name} ready in {info.get('warm_ms')} ms, {len(info.get('imported', []))} modules, {info.get('uss_kb')} kB")
        if info.get("failed"):
            _log(f"{name}: imports that failed: {'; '.join(info['failed'])}")
        self._publish({"event": "worker_ready", "app": name, "pid": worker.pid, "warm_ms": info.get("warm_ms"), "uss_kb": info.get("uss_kb")})

    def _warm_failed(self, slot: Slot, reason: str, worker: Worker | None) -> None:
        name = slot.info.name
        slot.failures += 1
        slot.warm_error = reason
        _log(f"warming {name} failed ({slot.failures}/{self.config.max_failures}): {reason}")
        self._publish({"event": "worker_failed", "app": name, "error": reason})
        if worker is not None and slot.worker is worker:
            slot.worker = None
            self._spawn_task(self._discard(worker))
        if slot.failures < self.config.max_failures:
            self._request_warm(name, delay=2.0 * slot.failures)

    # Starting and stopping apps

    async def start(self, ref: str, env: dict[str, str] | None = None, prepare: str = "auto", mode: str = "auto") -> Message:
        """Start an app, stopping the running one first.

        Raises:
            AppNotFound: if the reference is not an app.
            PrepareFailed: if its dependencies cannot be installed.
            WorkerError: if its worker cannot run it.
        """
        async with self._lifecycle:
            t_recv = time.time()
            slot = self._slot_for(ref)
            if env is not None:
                slot.env = env
            name = slot.info.name
            stopped: Message | None = None

            # Dependencies first: a venv must not change under the app using it
            if prepare != "never" and needs_prepare(slot.info):
                if self.active is not None and self.active.slot is slot:
                    stopped = await self._stop_active(rewarm=False)
                await self._ensure_prepared(slot)

            fingerprint, _ = self._fingerprint(slot)
            worker, slot.worker = slot.worker, None
            if worker is not None and (not worker.alive or worker.spec.fingerprint != fingerprint or mode == "immediate"):
                reason = "immediate mode" if mode == "immediate" else ("it exited" if not worker.alive else "the app changed")
                _log(f"not using the warm worker of {name}: {reason}")
                self._spawn_task(self._discard(worker))
                worker = None
            was_ready = worker is not None and worker.state == WorkerState.READY

            # The stopped app gets its next worker once this one has started; a restart of the same app
            # needs none, it is asked for below
            restarting = self.active is not None and self.active.slot is slot
            stopping = (
                asyncio.create_task(self._stop_active(rewarm=not restarting, rewarm_delay=self.config.replacement_delay_s))
                if self.active is not None
                else None
            )
            try:
                if worker is None:
                    # With an app to stop first, warming overlaps its shutdown
                    worker = await self._spawn(slot, "warm" if stopping is not None and mode != "immediate" else "immediate")
            finally:
                if stopping is not None:
                    stopped = await stopping
            if stopped is not None and self.config.settle_s > 0:
                await asyncio.sleep(self.config.settle_s)

            self._point_link(slot.info.path)
            run_id = next(self._run_ids)
            active = ActiveRun(slot=slot, worker=worker, run_id=run_id, t_recv=t_recv)
            self.active = active
            self._flush_warm_output(worker)
            self._prioritize(worker)
            t_run = time.time()
            try:
                started = await worker.run(self.config.run_timeout_s, slot.info.streamlit)
            except WorkerError:
                if self.active is active:
                    await self._stop_active(rewarm_delay=2.0)
                raise

            self._recent = pool.touch_recent(self._recent, name)
            self._save_recent()
            self._request_warm(name, delay=self.config.replacement_delay_s)
            reply: Message = {
                "ok": True,
                "app": name,
                "run_id": run_id,
                "pid": worker.pid,
                "path": started.get("path"),
                "worker": {"mode": worker.mode, "was_ready": was_ready, "warm_ms": worker.ready_info.get("warm_ms")},
                "stopped": stopped,
                "t": {"recv": t_recv, "run": t_run, "started": started.get("t")},
            }
            if started.get("shadowed"):
                reply["shadowed"] = started["shadowed"]
            _log(f"started {name} (run {run_id}, pid {worker.pid}, {started.get('path')}) in {(time.time() - t_recv) * 1000:.0f} ms")
            self._publish({"event": "app_started", **{k: reply[k] for k in ("app", "run_id", "pid", "path", "t")}})
            return reply

    async def stop(self) -> Message:
        async with self._lifecycle:
            app = self.active.slot.info.name if self.active else None
            stopped = await self._stop_active()
            return {"ok": True, "app": app, "stopped": stopped}

    async def restart(self, ref: str | None) -> Message:
        """Start the app again, the running one when none is named.

        Raises:
            AppNotFound: if no app is named and none is running.
        """
        if ref is None:
            if self.active is None:
                raise AppNotFound("no app is running")
            ref = self.active.slot.info.name
        return await self.start(ref)

    async def _stop_active(self, rewarm: bool = True, rewarm_delay: float = 0.0) -> Message | None:
        """Stop the running app and wait until nothing of it is left; then its next worker starts warming."""
        active = self.active
        if active is None:
            return None
        active.stopping = True
        info = await active.worker.stop(self.config.stop_timeout_s)
        self._finish_run(active, info, rewarm, rewarm_delay)
        return info.as_dict()

    def _finish_run(self, active: ActiveRun, info: ExitInfo, rewarm: bool = True, rewarm_delay: float = 0.0) -> None:
        if self.active is not active:
            return
        self.active = None
        self._resume_warm_ups()
        self._point_link(self.config.idle_dir)
        tail = active.log_splitter.rest()
        if tail:
            self._add_log_line(active, tail.decode(errors="replace"))
        slot = active.slot
        slot.last_exit = {"run_id": active.run_id, "t": time.time(), **info.as_dict()}
        name = slot.info.name
        how = "killed" if info.killed else (f"signal {info.signal}" if info.signal else f"exit code {info.returncode}")
        _log(f"{name} (run {active.run_id}) ended: {how}")
        self._publish({"event": "app_exited", "app": name, "run_id": active.run_id, **info.as_dict()})
        for queue in list(self._log_followers):
            queue.put_nowait(None)
        if rewarm:
            # The clean context for the next start of the app
            self._request_warm(name, delay=rewarm_delay)

    async def prepare(self, ref: str, env: dict[str, str] | None = None) -> Message:
        """Run run.sh prepare on an app now, then give it a fresh worker if what it depends on changed.

        Raises:
            AppNotFound: if the reference is not an app.
            PrepareFailed: if run.sh prepare fails.
        """
        async with self._lifecycle:
            slot = self._slot_for(ref)
            if env is not None:
                slot.env = env
            if self.active is not None and self.active.slot is slot:
                return protocol.error(protocol.ERR_BUSY, f"{slot.info.name} is running, stop it first")
            await self._ensure_prepared(slot, force=True)
            fingerprint, _ = self._fingerprint(slot)
            if slot.worker is not None and slot.worker.spec.fingerprint != fingerprint:
                worker, slot.worker = slot.worker, None
                await self._discard(worker)
            slot.failures = 0
            self._request_warm(slot.info.name)
            return {"ok": True, "app": slot.info.name, "venv": str(slot.info.venv_python.parent.parent)}

    def warm(self, ref: str) -> Message:
        slot = self._slot_for(ref)
        slot.failures = 0
        self._request_warm(slot.info.name)
        return {"ok": True, "app": slot.info.name, "queued": slot.pending_warm}

    def status(self) -> Message:
        active = self.active
        selected = set(self._selected())
        apps: list[Message] = []
        for name in sorted(self.slots):
            slot = self.slots[name]
            apps.append({
                "name": name,
                "selected": name in selected,
                "worker": slot.worker.describe() if slot.worker is not None else None,
                "pending_warm": slot.pending_warm,
                "failures": slot.failures,
                "warm_error": slot.warm_error,
                "last_exit": slot.last_exit,
            })
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": round(time.time() - self.started_at, 1),
            "active": None
            if active is None
            else {
                "app": active.slot.info.name,
                "run_id": active.run_id,
                "pid": active.worker.pid,
                "path": active.worker.started_info.get("path"),
                "uptime_s": round(time.time() - active.t_recv, 1),
                "app_run_t": active.app_run_t,
                "uss_kb": active.worker.describe().get("uss_kb"),
            },
            "apps": apps,
            "mem": read_meminfo_kb(),
        }

    # Worker callbacks

    def _on_output(self, worker: Worker, data: bytes) -> None:
        active = self.active
        if active is not None and active.worker is worker:
            self._write_app_output(active, data)
            return
        buffer = self._warm_output.get(worker.id)
        if buffer is not None and len(buffer) < WARM_OUTPUT_LIMIT:
            buffer += data[: WARM_OUTPUT_LIMIT - len(buffer)]

    def _flush_warm_output(self, worker: Worker) -> None:
        """Output the worker wrote while warming, e.g. import warnings, becomes the start of the app log."""
        buffered = self._warm_output.pop(worker.id, None)
        if buffered and self.active is not None and self.active.worker is worker:
            self._write_app_output(self.active, bytes(buffered))

    def _write_app_output(self, active: ActiveRun, data: bytes) -> None:
        out = sys.stdout.buffer
        out.write(data)
        out.flush()
        try:
            lines = active.log_splitter.feed(data)
        except protocol.ProtocolError:
            return
        for line in lines:
            self._add_log_line(active, line.decode(errors="replace"))

    def _add_log_line(self, active: ActiveRun, line: str) -> None:
        active.log.append(line)
        for queue in list(self._log_followers):
            queue.put_nowait(line)

    def _on_event(self, worker: Worker, message: Message) -> None:
        active = self.active
        if message.get("event") == "app_run" and active is not None and active.worker is worker:
            active.app_run_t = worker.app_run_t
            self._publish({"event": "app_run", "app": worker.app_name, "run_id": active.run_id, "t": worker.app_run_t})
            if self._resume_timer is not None:
                # Its bricks are starting now: give them a moment more, then let the warm-ups go on
                self._resume_timer.cancel()
                self._resume_timer = asyncio.get_running_loop().call_later(self.config.start_priority_tail_s, self._resume_warm_ups)
        elif message.get("event") == "error":
            _log(f"worker {worker.pid} of {worker.app_name}: {message}")

    def _on_exit(self, worker: Worker) -> None:
        active = self.active
        if active is not None and active.worker is worker:
            if not active.stopping:
                # The app ended on its own: it returned, failed, or was killed from outside
                self._finish_run(active, worker.exit or ExitInfo(None, None))
            return
        slot = self.slots.get(worker.app_name)
        discarded = worker.id in self._discarded
        self._discarded.discard(worker.id)
        buffered = self._warm_output.pop(worker.id, None)
        if slot is None or slot.worker is not worker:
            return
        slot.worker = None
        if discarded or self._closing:
            return
        # A standby worker died: say why, with what it printed
        slot.failures += 1
        slot.warm_error = f"worker exited while waiting ({worker.exit.as_dict() if worker.exit else {}})"
        _log(f"{worker.app_name}: {slot.warm_error}")
        if buffered:
            sys.stderr.buffer.write(bytes(buffered))
            sys.stderr.buffer.flush()
        self._request_warm(worker.app_name, delay=2.0 * slot.failures)

    # Start priority

    def _prioritize(self, starting: Worker) -> None:
        """Give the starting app the CPU: suspend the workers still warming and hold the next warm-ups.

        A board has 4 cores; a warm-up importing next to an app start slows the start down. Workers already
        ready sleep on their channel and cost nothing. Everything resumes once the app reaches App.run() plus
        a short tail, when it ends, or after start_priority_s at most.
        """
        if self.config.start_priority_s <= 0:
            return
        self._warm_gate.clear()
        for slot in self.slots.values():
            worker = slot.worker
            if worker is not None and worker is not starting and worker.state == WorkerState.WARMING and worker.suspend():
                self._suspended.append(worker)
        if self._resume_timer is not None:
            self._resume_timer.cancel()
        self._resume_timer = asyncio.get_running_loop().call_later(self.config.start_priority_s, self._resume_warm_ups)
        if self._suspended:
            _log(f"suspended the warm-up of {', '.join(w.app_name for w in self._suspended)} while {starting.app_name} starts")

    def _resume_warm_ups(self) -> None:
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        for worker in self._suspended:
            worker.resume()
        self._suspended.clear()
        self._warm_gate.set()

    def readiness(self) -> Message:
        """Which selected apps would start from a warm worker right now."""
        selected = self._selected()
        ready = [name for name in selected if (worker := self.slots[name].worker) is not None and worker.state == WorkerState.READY]
        return {"apps": len(selected), "ready": ready, "not_ready": [name for name in selected if name not in ready]}

    # Events

    def _publish(self, event: Message) -> None:
        event.setdefault("t_event", time.time())
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    # /app

    def _point_link(self, target: Path) -> None:
        """Point the current-app symlink at target, atomically: /app never dangles and never names two apps."""
        link = self.config.current_link
        tmp = link.with_name(f".{link.name}.{os.getpid()}.tmp")
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        os.symlink(target, tmp)
        os.replace(tmp, link)

    # Recently started apps, warmed first after a restart of the container

    def _load_recent(self) -> list[str]:
        try:
            recent = json.loads(self.config.state_file.read_text()).get("recent", [])
            return [str(name) for name in recent]
        except (OSError, ValueError, AttributeError):
            return []

    def _save_recent(self) -> None:
        try:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"recent": self._recent}))
            os.replace(tmp, self.config.state_file)
        except OSError as e:
            _log(f"cannot save {self.config.state_file}: {e}")

    # Control socket

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        splitter = protocol.LineSplitter()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                try:
                    lines = splitter.feed(chunk)
                except protocol.ProtocolError as e:
                    await self._reply(writer, protocol.error(protocol.ERR_BAD_REQUEST, str(e)))
                    break
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        request = protocol.decode(line)
                    except protocol.ProtocolError as e:
                        await self._reply(writer, protocol.error(protocol.ERR_BAD_REQUEST, str(e)))
                        continue
                    if request.get("cmd") == "events" or (request.get("cmd") == "logs" and request.get("follow")):
                        await self._stream(request, writer)
                        return
                    reply = await self.dispatch(request)
                    if "id" in request:
                        reply["id"] = request["id"]
                    await self._reply(writer, reply)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, message: Message) -> None:
        writer.write(protocol.encode(message))
        await writer.drain()

    async def dispatch(self, request: Message) -> Message:
        """Execute one request of the control protocol and build its reply."""
        cmd = request.get("cmd")
        try:
            if cmd == "ping":
                return {"ok": True, "v": protocol.PROTOCOL_VERSION, "version": __version__, "readiness": self.readiness()}
            if cmd == "start":
                return await self.start(
                    _required_str(request, "app"),
                    env=_optional_env(request),
                    prepare=str(request.get("prepare", "auto")),
                    mode=str(request.get("mode", "auto")),
                )
            if cmd == "stop":
                return await self.stop()
            if cmd == "restart":
                app = request.get("app")
                return await self.restart(str(app) if app else None)
            if cmd == "prepare":
                return await self.prepare(_required_str(request, "app"), env=_optional_env(request))
            if cmd == "warm":
                return self.warm(_required_str(request, "app"))
            if cmd == "rescan":
                return {"ok": True, "apps": self.rescan()}
            if cmd == "status":
                return self.status()
            if cmd == "logs":
                tail = int(request.get("tail", 200))
                lines = list(self.active.log)[-tail:] if self.active is not None else []
                return {"ok": True, "app": self.active.slot.info.name if self.active else None, "lines": lines}
            if cmd == "shutdown":
                self.request_shutdown()
                return {"ok": True}
            return protocol.error(protocol.ERR_BAD_REQUEST, f"unknown command {cmd!r}")
        except ValueError as e:
            return protocol.error(protocol.ERR_BAD_REQUEST, str(e))
        except AppNotFound as e:
            return protocol.error(protocol.ERR_NOT_FOUND, str(e))
        except PrepareFailed as e:
            return protocol.error(protocol.ERR_PREPARE_FAILED, str(e))
        except WorkerError as e:
            return protocol.error(protocol.ERR_SPAWN_FAILED, str(e))
        except Exception as e:  # noqa: BLE001 - the supervisor outlives a bug in one request
            _log(f"{cmd} failed: {traceback.format_exc()}")
            return protocol.error(protocol.ERR_INTERNAL, repr(e))

    async def _stream(self, request: Message, writer: asyncio.StreamWriter) -> None:
        """Streaming requests: `events`, and `logs` with follow, until the client goes away."""
        if request.get("cmd") == "events":
            events: asyncio.Queue[Message | None] = asyncio.Queue()
            self._subscribers.add(events)
            try:
                await self._reply(writer, {"ok": True, "id": request.get("id"), "streaming": "events"})
                while (event := await events.get()) is not None:
                    await self._reply(writer, event)
            except ConnectionError:
                pass
            finally:
                self._subscribers.discard(events)
            return
        lines: asyncio.Queue[str | None] = asyncio.Queue()
        active = self.active
        if active is None:
            await self._reply(writer, {"ok": True, "id": request.get("id"), "streaming": "logs", "app": None})
            await self._reply(writer, {"event": "end"})
            return
        self._log_followers.add(lines)
        try:
            await self._reply(writer, {"ok": True, "id": request.get("id"), "streaming": "logs", "app": active.slot.info.name})
            for line in list(active.log)[-int(request.get("tail", 200)) :]:
                await self._reply(writer, {"line": line})
            while (line := await lines.get()) is not None:
                await self._reply(writer, {"line": line})
            await self._reply(writer, {"event": "end"})
        except ConnectionError:
            pass
        finally:
            self._log_followers.discard(lines)


def _required_str(request: Message, key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} is required")
    return value


def _optional_env(request: Message) -> dict[str, str] | None:
    env = request.get("env")
    if env is None:
        return None
    if not isinstance(env, dict):
        raise ValueError("'env' must be an object of strings")
    return {str(key): "" if value is None else str(value) for key, value in env.items()}  # pyright: ignore[reportUnknownVariableType]


async def serve(config: Config | None = None) -> None:
    """Run the supervisor until it is told to stop."""
    await Supervisor(config or Config.from_env()).serve()
