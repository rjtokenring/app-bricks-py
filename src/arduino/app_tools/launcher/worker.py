# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Warm app process of arduino-app-launcher: imports in advance, then runs /app/python/main.py once.

The supervisor starts it by path with the interpreter of the app venv, `<app>/.cache/.venv/bin/python
.../launcher/worker.py`, never as `-m`: imports then resolve as for `python main.py` in run.sh, the app venv
first and the image site-packages after it, and an `arduino` package the app installs in its venv cannot
replace this file. For the same reason it only uses the standard library.

Life of a worker, driven by messages on the control channel (fd --control-fd, one JSON object per line):

1. `hello` to the supervisor.
2. `warm {modules}` from the supervisor: import them one by one, then `ready`. A `run` arriving meanwhile
   cuts the warm-up short. In immediate mode the supervisor sends `run` straight away.
3. `run`: the supervisor has stopped the previous app and pointed /app at this one. Move into /app and
   execute main.py as the `__main__` module, the way `python /app/python/main.py` would.
4. `reload`, while main.py runs: stop the app inside this process, check that nothing of it is left, forget its
   modules and execute the edited main.py again, imports kept. `reload_ok` when it did, `reload_fallback {reason}`
   when the process is not clean enough: the supervisor then restarts the app with a new worker.
5. The process ends with the app. Apart from a reload it is never reused: the next start gets a new worker.
"""

import sys

# The folder of this file is sys.path[0]: drop it before anything is imported, it is not the app's.
del sys.path[0]

if sys.platform == "win32":
    raise ImportError("the arduino-app-launcher worker needs a POSIX system")

import argparse  # noqa: E402
import atexit  # noqa: E402
import builtins  # noqa: E402
import gc  # noqa: E402
import importlib  # noqa: E402
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import linecache  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

PROTOCOL_VERSION = 1
"""Must match arduino.app_tools.launcher.protocol.PROTOCOL_VERSION."""

FEATURES = ["reload"]
"""What this worker supports beyond the protocol version, announced in `hello`."""

RELOAD_SIGNAL = signal.SIGUSR1
"""Interrupts the main thread for a reload: a signal is what wakes it from a sleep or a blocking call."""

RELOAD_THREAD_GRACE_S = 1.0
"""How long the threads of the stopped app get to end before a reload gives up on the process."""

PERSISTENT_THREAD_PREFIXES = ("Bridge.",)
"""Threads that may outlive a run: the router connection is process-wide and kept across reloads."""

APP_STARTING_BANNER = "======== App is starting ============================"
"""Printed before main.py runs, as run.sh does: arduino-app-cli and App Lab read the app log."""

EXIT_LINK_MISMATCH = 70
"""Exit code when /app does not point at this worker's app at run time."""

Message = dict[str, Any]


class Channel:
    """The control channel to the supervisor: a stream socket inherited as a file descriptor."""

    def __init__(self, fd: int) -> None:
        # Not inherited by what the app runs: the supervisor must see the channel close with this process
        os.set_inheritable(fd, False)
        self._sock = socket.socket(fileno=fd)
        self._buffer = b""
        self._eof = False
        # While the app runs the main thread and the control thread both send
        self._send_lock = threading.Lock()

    def send(self, message: Message) -> None:
        try:
            with self._send_lock:
                self._sock.sendall(json.dumps(message, separators=(",", ":"), default=str).encode() + b"\n")
        except OSError:
            pass  # The supervisor is gone: it will not miss the message

    def _pop_line(self) -> Message | None:
        end = self._buffer.find(b"\n")
        if end < 0:
            return None
        line, self._buffer = self._buffer[:end], self._buffer[end + 1 :]
        try:
            message = json.loads(line)
        except ValueError:
            return {}
        return message if isinstance(message, dict) else {}  # pyright: ignore[reportUnknownVariableType]

    def recv(self, timeout: float | None = None) -> Message | None:
        """Next message; None at end of stream, or when the timeout expires first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            message = self._pop_line()
            if message is not None:
                return message
            if self._eof:
                return None
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([self._sock], [], [], remaining)
            if not readable:
                return None
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                self._eof = True
            self._buffer += chunk

    @property
    def closed(self) -> bool:
        return self._eof and not self._buffer

    def close(self) -> None:
        self._sock.close()


class Quit(Exception):
    """The supervisor asked this worker to end, or went away, before the app ran."""


def _is_within(path: str, folder: str) -> bool:
    real = os.path.realpath(path)
    return real == folder or real.startswith(folder + os.sep)


def _module_origin(spec: importlib.machinery.ModuleSpec) -> str | None:
    if spec.origin and spec.origin not in ("built-in", "frozen", "namespace"):
        return spec.origin
    locations = spec.submodule_search_locations
    return next(iter(locations), None) if locations else None


def _uss_kb() -> int | None:
    """Memory this process does not share with any other, from /proc/self/smaps_rollup."""
    try:
        with open("/proc/self/smaps_rollup") as f:
            private = 0
            for line in f:
                if line.startswith(("Private_Clean:", "Private_Dirty:")):
                    private += int(line.split()[1])
            return private
    except (OSError, ValueError, IndexError):
        return None


def _set_oom_score_adj(value: int) -> int | None:
    """Make an idle worker the first thing the kernel kills under memory pressure; returns the previous value."""
    try:
        with open("/proc/self/oom_score_adj") as f:
            previous = int(f.read().strip())
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(value))
        return previous
    except (OSError, ValueError):
        return None


def warm(channel: Channel, modules: list[str], app_real: str) -> Message | None:
    """Import the modules one by one; returns a `run` that arrived meanwhile, None once they are all done.

    Raises:
        Quit: if the supervisor sends `quit` or closes the channel.
    """
    started = time.monotonic()
    imported: list[str] = []
    failed: list[str] = []
    for name in modules:
        pending = channel.recv(timeout=0)
        if pending is None and channel.closed:
            raise Quit()
        if pending is not None:
            if pending.get("cmd") == "run":
                return pending
            if pending.get("cmd") == "quit":
                raise Quit()
        if name in sys.modules:
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError) as e:
            failed.append(f"{name}: {e!r}")
            continue
        if spec is None:
            continue  # Not installed in this venv: the app does not need it, or will say so itself
        origin = _module_origin(spec)
        if origin is not None and _is_within(origin, app_real):
            continue  # The app's own code is imported only when the app runs
        try:
            importlib.import_module(name)
            imported.append(name)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - best effort, a module may even call sys.exit()
            failed.append(f"{name}: {e!r}")
    gc.collect()
    channel.send({
        "event": "ready",
        "warm_ms": round((time.monotonic() - started) * 1000, 1),
        "imported": imported,
        "failed": failed,
        "uss_kb": _uss_kb(),
    })
    return None


def _trace_app_run(channel: Channel) -> None:
    """Report when the app reaches App.run(): the moment its bricks start, which start timings measure up to.

    Once per run: tracing an App already traced re-arms the report for the next run.
    """
    app_module = sys.modules.get("arduino.app_utils.app")
    controller: Any = getattr(app_module, "AppController", None)
    original: Any = getattr(controller, "run", None)
    if controller is None or original is None:
        return
    if getattr(original, "_launcher_traced", False):
        setattr(original, "_reported", False)
        return

    def run(self: object, *args: Any, **kwargs: Any) -> object:
        if not getattr(run, "_reported", False):
            setattr(run, "_reported", True)
            channel.send({"event": "app_run", "t": time.time()})
        return original(self, *args, **kwargs)

    setattr(run, "_launcher_traced", True)
    run.__doc__ = original.__doc__
    run.__wrapped__ = original  # pyright: ignore[reportFunctionMemberAccess]
    controller.run = run


APP_MODULE = "arduino.app_utils.app"


class _TraceOnImport:
    """Meta path finder that installs the App.run() trace once the app imports it, when the worker did not in advance."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> importlib.machinery.ModuleSpec | None:
        if fullname != APP_MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        loader: Any = spec.loader if spec is not None else None
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module
        channel = self._channel

        def exec_and_trace(module: types.ModuleType) -> None:
            exec_module(module)
            _trace_app_run(channel)

        loader.exec_module = exec_and_trace
        return spec


def _install_trace(channel: Channel) -> None:
    if APP_MODULE in sys.modules:
        _trace_app_run(channel)
    elif not any(isinstance(finder, _TraceOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TraceOnImport(channel))


def _shadowed_local_modules(roots: list[str], app_real: str) -> list[str]:
    """Modules already imported under a name the app's own folders provide: in the app they would not be the app's."""
    shadowed: list[str] = []
    for root in roots:
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for entry in entries:
            name = entry[:-3] if entry.endswith(".py") else entry
            module = sys.modules.get(name)
            if module is None or not name.isidentifier():
                continue
            location = getattr(module, "__file__", None)
            if location is None or not _is_within(location, app_real):
                shadowed.append(name)
    return sorted(set(shadowed))


def _exec_fresh_interpreter(argv: list[str]) -> None:
    """Replace this process with a fresh interpreter: same pid, same environment, nothing imported in advance."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *argv])


class Reload(BaseException):
    """Raised into the main thread to end the run for a reload. App.run() shuts the app down on its way out."""


def _exec_main(main_py: str) -> None:
    """Execute main.py as the `__main__` module, as `python main.py` does.

    The module stays `__main__` for good, unlike with runpy.run_path, which puts the worker module back as soon as
    main.py returns: threads and atexit handlers of the app still run after that. Like the interpreter, it drops
    `__file__` and `__cached__` from it once main.py is done. An uncaught exception is reported by sys.excepthook,
    the app_utils one when the app imports it, without the frames of this file.
    """
    module = types.ModuleType("__main__")
    module.__file__ = main_py
    module.__loader__ = importlib.machinery.SourceFileLoader("__main__", main_py)
    module.__spec__ = None
    module.__package__ = None
    module.__dict__["__cached__"] = None
    module.__builtins__ = builtins  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules["__main__"] = module
    try:
        with open(main_py, "rb") as f:
            code = compile(f.read(), main_py, "exec")
        exec(code, module.__dict__)  # noqa: S102 - running the app is the point
    except (SystemExit, KeyboardInterrupt, Reload):
        raise
    except BaseException as exc:  # noqa: BLE001 - reported like an uncaught exception
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
            tb = tb.tb_next
        # The default hook prints exc.__traceback__, whatever traceback it is given
        sys.excepthook(type(exc), exc.with_traceback(tb), tb)
        raise SystemExit(1) from None
    finally:
        # As the interpreter does for a script once it is done: atexit handlers no longer see them
        module.__dict__.pop("__file__", None)
        module.__dict__.pop("__cached__", None)


def _run_streamlit(main_py: str) -> None:
    """`python -m streamlit run --server.port 7000 main.py`, in this process when Streamlit imports."""
    argv = ["run", "--server.port", "7000", main_py]
    try:
        import runpy

        importlib.import_module("streamlit.web.cli")
    except ImportError:
        _exec_fresh_interpreter(["-m", "streamlit", *argv])
        return
    sys.argv = ["streamlit", *argv]
    runpy.run_module("streamlit", run_name="__main__", alter_sys=True)


@dataclass
class Baseline:
    """Process state right before main.py runs, put back after a run that ends in a reload."""

    threads: set[int | None]
    signals: dict[int, Any]
    path: list[str]
    environ: dict[str, str]
    argv: list[str]
    cwd: str
    stdio: tuple[Any, Any, Any]
    logging_root: tuple[list[Any], int] | None
    atexit_callbacks: int

    @classmethod
    def capture(cls) -> "Baseline":
        signals: dict[int, Any] = {}
        for signum in signal.valid_signals():
            if signum in (signal.SIGKILL, signal.SIGSTOP):
                continue
            try:
                handler = signal.getsignal(signum)
            except (OSError, ValueError):
                continue
            if handler is not None:  # None: a handler not installed from Python, which Python cannot put back
                signals[signum] = handler
        logging_module: Any = sys.modules.get("logging")
        return cls(
            threads={thread.ident for thread in threading.enumerate()},
            signals=signals,
            path=list(sys.path),
            environ=dict(os.environ),
            argv=list(sys.argv),
            cwd=os.getcwd(),
            stdio=(sys.stdin, sys.stdout, sys.stderr),
            logging_root=(list(logging_module.root.handlers), logging_module.root.level) if logging_module else None,
            atexit_callbacks=_atexit_callbacks(),
        )

    def restore(self) -> None:
        for signum, handler in self.signals.items():
            if signal.getsignal(signum) is not handler:
                try:
                    signal.signal(signum, handler)
                except (OSError, ValueError):
                    pass
        sys.path[:] = self.path
        if dict(os.environ) != self.environ:
            os.environ.clear()
            os.environ.update(self.environ)
        sys.argv = list(self.argv)
        os.chdir(self.cwd)
        sys.stdin, sys.stdout, sys.stderr = self.stdio
        logging_module: Any = sys.modules.get("logging")
        if logging_module is not None:
            handlers, level = self.logging_root or ([], logging_module.WARNING)
            root = logging_module.root
            for handler in root.handlers:
                if handler not in handlers:
                    handler.close()
            root.handlers[:] = handlers
            root.setLevel(level)


def _atexit_callbacks() -> int:
    count: Callable[[], int] | None = getattr(atexit, "_ncallbacks", None)
    return count() if count is not None else 0


@dataclass
class ReloadState:
    """Shared by the main thread, which runs the app, and the control thread, which receives `reload`."""

    channel: Channel
    handler: Callable[[int, types.FrameType | None], None] | None = None
    in_main: bool = False
    """main.py is executing: only then can a reload end it."""
    requested_at: float | None = None
    raised: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    """Reentrant: the signal handler runs on the main thread, possibly while that thread holds it."""


def _install_reload_handler(state: ReloadState) -> None:
    def on_reload_signal(signum: int, frame: types.FrameType | None) -> None:
        with state.lock:
            # Not in main.py any more: the main thread reports the request it missed, see _run_reloadable
            if state.requested_at is None or state.raised or not state.in_main:
                return
            state.raised = True
        raise Reload()

    state.handler = on_reload_signal
    signal.signal(RELOAD_SIGNAL, on_reload_signal)


def _request_reload(state: ReloadState) -> None:
    """On the control thread: interrupt the main thread, wherever it is, with Reload."""
    with state.lock:
        if state.requested_at is not None:
            return  # The reload under way answers
        if not state.in_main:
            reason = "main.py has already returned"
        elif signal.getsignal(RELOAD_SIGNAL) is not state.handler:
            reason = f"the app handles {RELOAD_SIGNAL.name} itself"
        else:
            reason = None
            state.requested_at = time.monotonic()
    if reason is not None:
        state.channel.send({"event": "reload_fallback", "reason": reason})
        return
    main_ident = threading.main_thread().ident
    assert main_ident is not None
    signal.pthread_kill(main_ident, RELOAD_SIGNAL)


def _control_loop(state: ReloadState) -> None:
    """Read the channel while the app runs: the supervisor sends `reload` there."""
    while True:
        message = state.channel.recv()
        if message is None:
            return  # The supervisor is gone: the app runs on, as it did before reloads existed
        if message.get("cmd") == "reload":
            _request_reload(state)


def _stray_threads(baseline: Baseline) -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.ident not in baseline.threads
        and not thread.name.startswith(PERSISTENT_THREAD_PREFIXES)
        # Threads started outside Python and seen by it once: nothing tells whether they still run
        and type(thread).__name__ != "_DummyThread"
    ]


def _purge_app_modules(roots: tuple[str, ...]) -> int:
    """Forget the modules imported from the app's own folders, so that the next run imports the edited ones."""
    prefixes = tuple(root + os.sep for root in roots)
    names: list[str] = []
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if location is None:
            locations = getattr(module, "__path__", None)
            location = next(iter(locations), None) if locations else None
        if isinstance(location, str) and location.startswith(prefixes):
            names.append(name)
    for name in names:
        module = sys.modules.pop(name, None)
        # A .pyc is checked against the source mtime in whole seconds and its size: a source edited within the
        # second of its import, keeping its size, would run the old code
        cached = getattr(module, "__cached__", None)
        if isinstance(cached, str):
            try:
                os.remove(cached)
            except OSError:
                pass
    importlib.invalidate_caches()
    linecache.clearcache()
    return len(names)


def _reset_for_reload(baseline: Baseline, roots: tuple[str, ...]) -> tuple[list[str], int]:
    """Undo the run that just ended. Returns what makes the process unfit for another run, and how many modules went."""
    problems: list[str] = []
    if APP_MODULE in sys.modules:
        try:
            reset: Callable[[], list[str]] = importlib.import_module("arduino.app_utils._reload").reset
        except (ImportError, AttributeError):
            problems.append("the arduino library of this venv cannot reload")
        else:
            try:
                problems.extend(reset())
            except Exception as e:  # noqa: BLE001 - any failure only means: restart instead
                problems.append(f"the library reset failed: {e!r}")
    # Signals first: from here on a SIGTERM must end the process at once, as it does an idle worker
    baseline.restore()
    deadline = time.monotonic() + RELOAD_THREAD_GRACE_S
    while (stray := _stray_threads(baseline)) and time.monotonic() < deadline:
        time.sleep(0.02)
    if stray:
        problems.append(f"threads still running: {', '.join(sorted(stray))}")
    if problems:
        return problems, 0
    purged = _purge_app_modules(roots)
    gc.collect()
    return [], purged


def _run_reloadable(channel: Channel, main_py: str, roots: tuple[str, ...]) -> None:
    """Run main.py; on a reload, run it again in this process when nothing of the previous run is left."""
    state = ReloadState(channel)
    _install_reload_handler(state)
    threading.Thread(target=_control_loop, args=(state,), name="launcher-control", daemon=True).start()
    baseline = Baseline.capture()
    while True:
        with state.lock:
            state.in_main = True
        caught: float | None = None
        try:
            _exec_main(main_py)
        except Reload:
            caught = time.monotonic()
        finally:
            with state.lock:
                state.in_main = False
                missed = caught is None and state.requested_at is not None
            if missed:
                channel.send({"event": "reload_fallback", "reason": "main.py has already returned"})
        if caught is None:
            return
        problems, purged = _reset_for_reload(baseline, roots)
        if problems:
            channel.send({"event": "reload_fallback", "reason": "; ".join(problems)})
            # The supervisor restarts the app with a new worker, and stops this one first
            while True:
                time.sleep(3600)
        reset_done = time.monotonic()
        requested_at = state.requested_at if state.requested_at is not None else caught
        channel.send({
            "event": "reload_ok",
            "t": time.time(),
            "pid": os.getpid(),
            "path": "reload",
            "shutdown_ms": round((caught - requested_at) * 1000, 1),
            "reset_ms": round((reset_done - caught) * 1000, 1),
            "purged": purged,
            "atexit_added": _atexit_callbacks() - baseline.atexit_callbacks,
        })
        with state.lock:
            state.requested_at = None
            state.raised = False
        print(APP_STARTING_BANNER, flush=True)
        _install_trace(channel)


def run(channel: Channel, request: Message, app_real: str, link: str, warmed: bool) -> None:
    """Move into the app as /app sees it and run it."""
    if os.path.realpath(link) != app_real:
        channel.send({"event": "error", "code": "link_mismatch", "link": os.path.realpath(link), "app": app_real})
        raise SystemExit(EXIT_LINK_MISMATCH)

    python_dir = os.path.join(link, "python")
    bricks_dir = os.path.join(link, "bricks")
    main_py = os.path.join(python_dir, "main.py")

    # As run.sh: cwd /app, the script folder first on sys.path, then /app/bricks through PYTHONPATH
    os.chdir(link)
    os.environ["PWD"] = link
    sys.path.insert(0, python_dir)
    if os.path.isdir(bricks_dir):
        sys.path.insert(1, bricks_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), bricks_dir]))
    sys.argv = [main_py]

    shadowed = _shadowed_local_modules([python_dir, bricks_dir], app_real)
    streamlit = bool(request.get("streamlit"))
    path = "streamlit" if streamlit else ("exec" if shadowed else ("warm" if warmed else "immediate"))
    channel.send({"event": "started", "t": time.time(), "pid": os.getpid(), "path": path, "shadowed": shadowed})

    if streamlit:
        _run_streamlit(main_py)
        return
    print(APP_STARTING_BANNER, flush=True)
    _install_trace(channel)
    if shadowed:
        # An app module has the name of one imported in advance: only a fresh interpreter runs it as python main.py would
        channel.close()
        _exec_fresh_interpreter([main_py])
    _run_reloadable(channel, main_py, (python_dir, bricks_dir, os.path.join(app_real, "python"), os.path.join(app_real, "bricks")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--app", required=True, help="real path of the app folder")
    parser.add_argument("--link", default="/app", help="path the app is reached at once it runs")
    args = parser.parse_args()
    app_real = os.path.realpath(args.app)

    channel = Channel(args.control_fd)
    channel.send({
        "event": "hello",
        "v": PROTOCOL_VERSION,
        "features": FEATURES,
        "pid": os.getpid(),
        "executable": sys.executable,
        "prefix": sys.prefix,
    })

    request = channel.recv()
    try:
        if request is None or request.get("cmd") == "quit":
            raise Quit()
        warmed = request.get("cmd") == "warm"
        if warmed:
            previous_oom = _set_oom_score_adj(1000)
            modules = [str(name) for name in request.get("modules", [])]
            request = warm(channel, modules, app_real)
            while request is None or request.get("cmd") != "run":
                request = channel.recv()
                if request is None or request.get("cmd") == "quit":
                    raise Quit()
            if previous_oom is not None:
                _set_oom_score_adj(previous_oom)
        elif request.get("cmd") != "run":
            raise Quit()
    except Quit:
        channel.close()
        sys.exit(0)
    run(channel, request, app_real, args.link, warmed)


if __name__ == "__main__":
    main()
