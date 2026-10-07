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
4. The process ends with the app. It is never reused: the next run of the app gets a new worker.
"""

import sys

# The folder of this file is sys.path[0]: drop it before anything is imported, it is not the app's.
del sys.path[0]

if sys.platform == "win32":
    raise ImportError("the arduino-app-launcher worker needs a POSIX system")

import argparse  # noqa: E402
import builtins  # noqa: E402
import gc  # noqa: E402
import importlib  # noqa: E402
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from typing import Any  # noqa: E402

PROTOCOL_VERSION = 1
"""Must match arduino.app_tools.launcher.protocol.PROTOCOL_VERSION."""

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

    def send(self, message: Message) -> None:
        try:
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
    """Report when the app reaches App.run(): the moment its bricks start, which start timings measure up to."""
    app_module = sys.modules.get("arduino.app_utils.app")
    controller: Any = getattr(app_module, "AppController", None)
    original: Any = getattr(controller, "run", None)
    if controller is None or original is None or getattr(original, "_launcher_traced", False):
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
    else:
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
    except (SystemExit, KeyboardInterrupt):
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
    _exec_main(main_py)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--app", required=True, help="real path of the app folder")
    parser.add_argument("--link", default="/app", help="path the app is reached at once it runs")
    args = parser.parse_args()
    app_real = os.path.realpath(args.app)

    channel = Channel(args.control_fd)
    channel.send({"event": "hello", "v": PROTOCOL_VERSION, "pid": os.getpid(), "executable": sys.executable, "prefix": sys.prefix})

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
