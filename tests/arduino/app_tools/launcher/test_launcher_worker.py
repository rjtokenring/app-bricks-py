# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The worker runs main.py as `python main.py` would: same module, paths, signals, exit codes."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher needs a POSIX system")

WORKER_PY = Path(__file__).parents[4] / "src" / "arduino" / "app_tools" / "launcher" / "worker.py"

PROBE = """
import atexit, json, os, signal, sys

PRELOADED = sorted(name for name in ("fractions", "statistics") if name in sys.modules)
OUT = os.environ["PROBE_OUT"]


def dump(tag):
    main = sys.modules["__main__"]
    data = {
        "name": __name__,
        "file": globals().get("__file__"),
        "argv": sys.argv,
        "path0": sys.path[0],
        "cwd": os.getcwd(),
        "main_file": getattr(main, "__file__", None),
        "main_is_this_module": main.__dict__ is globals(),
        "dunders": sorted(k for k in globals() if k.startswith("__") and k != "__annotations__"),
        "spec": repr(__spec__),
        "package": __package__,
        "cached": globals().get("__cached__", "unset"),
        "loader": type(__loader__).__name__,
        "sigterm": str(signal.getsignal(signal.SIGTERM)),
        "sigint": str(signal.getsignal(signal.SIGINT)),
        "sigpipe": str(signal.getsignal(signal.SIGPIPE)),
        "preloaded": PRELOADED,
    }
    with open(OUT + tag, "w") as f:
        json.dump(data, f)


dump(".top")
atexit.register(dump, ".atexit")
mode = os.environ.get("PROBE_MODE", "return")
if mode == "exit3":
    sys.exit(3)
if mode == "raise":
    raise RuntimeError("boom")
if mode == "local":
    import fractions
    with open(OUT + ".local", "w") as f:
        f.write(fractions.__file__)
if mode == "app_run":
    from arduino.app_utils import App
    open(OUT + ".running", "w").close()
    App.run()
"""


def make_app(tmp_path: Path, files: dict[str, str] | None = None) -> tuple[Path, Path]:
    """An app folder, and a symlink standing in for /app pointing at it."""
    app = tmp_path / "apps" / "probe"
    (app / "python").mkdir(parents=True)
    (app / "python" / "main.py").write_text(PROBE)
    (app / "app.yaml").write_text("name: probe\nbricks: []\n")
    for relative, content in (files or {}).items():
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    link = tmp_path / "app"
    link.symlink_to(app)
    return app, link


class WorkerProc:
    """A worker driven by hand, the way the supervisor does."""

    def __init__(self, app: Path, link: Path, env: dict[str, str]) -> None:
        self.parent, child = socket.socketpair()
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER_PY), "--control-fd", str(child.fileno()), "--app", str(app), "--link", str(link)],
            pass_fds=(child.fileno(),),
            cwd=str(app),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child.close()
        self.parent.settimeout(30)
        self._buffer = b""

    def send(self, message: dict[str, Any]) -> None:
        self.parent.sendall(json.dumps(message).encode() + b"\n")

    def event(self) -> dict[str, Any] | None:
        while b"\n" not in self._buffer:
            chunk = self.parent.recv(65536)
            if not chunk:
                return None
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def wait(self) -> tuple[int, str, str]:
        out, err = self.proc.communicate(timeout=30)
        self.parent.close()
        return self.proc.returncode, out, err


def probe_env(tmp_path: Path, mode: str, tag: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update({"PROBE_OUT": str(tmp_path / tag), "PROBE_MODE": mode, "PYTHONUNBUFFERED": "1"})
    return env


def run_classic(tmp_path: Path, link: Path, mode: str) -> tuple[int, str, str]:
    """What run.sh does: cd /app; python /app/python/main.py."""
    result = subprocess.run(
        [sys.executable, str(link / "python" / "main.py")],
        cwd=str(link),
        env=probe_env(tmp_path, mode, "classic"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def run_worker(tmp_path: Path, app: Path, link: Path, mode: str, modules: list[str] | None = None) -> tuple[int, str, str, list[dict[str, Any]]]:
    worker = WorkerProc(app, link, probe_env(tmp_path, mode, "worker"))
    events = [worker.event()]
    worker.send({"cmd": "warm", "modules": modules or []})
    events.append(worker.event())
    worker.send({"cmd": "run"})
    while (event := worker.event()) is not None:
        events.append(event)
    code, out, err = worker.wait()
    return code, out, err, [e for e in events if e]


def read(tmp_path: Path, name: str) -> dict[str, Any]:
    return json.loads((tmp_path / name).read_text())


def test_main_runs_as_with_python_main_py(tmp_path: Path):
    app, link = make_app(tmp_path)
    classic_code, _, _ = run_classic(tmp_path, link, "return")
    code, out, _, events = run_worker(tmp_path, app, link, "return")
    assert (classic_code, code) == (0, 0)
    assert [e["event"] for e in events] == ["hello", "ready", "started"]
    assert events[2]["path"] == "warm"
    assert "App is starting" in out

    classic, worker = read(tmp_path, "classic.top"), read(tmp_path, "worker.top")
    for key in (
        "name",
        "file",
        "argv",
        "main_file",
        "main_is_this_module",
        "dunders",
        "spec",
        "package",
        "cached",
        "loader",
        "sigterm",
        "sigint",
        "sigpipe",
    ):
        assert worker[key] == classic[key], key
    assert worker["file"] == str(link / "python" / "main.py"), "the app sees itself under /app"
    assert worker["path0"] == str(link / "python")
    assert os.path.realpath(worker["path0"]) == os.path.realpath(classic["path0"])
    assert os.path.realpath(worker["cwd"]) == os.path.realpath(classic["cwd"]) == str(app.resolve())


def test_atexit_handlers_still_see_the_app_as_main(tmp_path: Path):
    app, link = make_app(tmp_path)
    run_classic(tmp_path, link, "return")
    run_worker(tmp_path, app, link, "return")
    classic, worker = read(tmp_path, "classic.atexit"), read(tmp_path, "worker.atexit")
    assert worker["main_is_this_module"] and classic["main_is_this_module"]
    # Once the script is done the interpreter drops __file__ and __cached__ from __main__: so does the worker
    for key in ("file", "main_file", "cached", "dunders"):
        assert worker[key] == classic[key], key
    assert classic["file"] is None


def test_exit_codes_match(tmp_path: Path):
    app, link = make_app(tmp_path)
    assert run_classic(tmp_path, link, "exit3")[0] == 3
    assert run_worker(tmp_path, app, link, "exit3")[0] == 3


def test_an_uncaught_exception_is_reported_without_the_worker_frames(tmp_path: Path):
    app, link = make_app(tmp_path)
    classic_code, _, classic_err = run_classic(tmp_path, link, "raise")
    code, _, err, _ = run_worker(tmp_path, app, link, "raise")
    assert classic_code == code == 1
    assert "RuntimeError: boom" in err
    assert "worker.py" not in err, err
    assert err.splitlines()[1] == classic_err.splitlines()[1], "the traceback starts at main.py"


def test_modules_are_imported_in_advance(tmp_path: Path):
    app, link = make_app(tmp_path)
    code, _, _, events = run_worker(tmp_path, app, link, "return", modules=["fractions", "statistics", "no_such_module_anywhere"])
    assert code == 0
    ready = events[1]
    assert ready["event"] == "ready"
    assert ready["imported"] == ["fractions", "statistics"]
    assert read(tmp_path, "worker.top")["preloaded"] == ["fractions", "statistics"]


def test_the_app_own_modules_are_never_imported_in_advance(tmp_path: Path):
    app, link = make_app(tmp_path, {"python/helper.py": "raise SystemExit('imported in advance')\n"})
    code, _, _, events = run_worker(tmp_path, app, link, "return", modules=["helper"])
    assert code == 0
    assert events[1]["imported"] == []


def test_an_app_module_named_like_a_preloaded_one_gets_a_fresh_interpreter(tmp_path: Path):
    app, link = make_app(tmp_path, {"python/fractions.py": "LOCAL = True\n"})
    code, _, _, events = run_worker(tmp_path, app, link, "local", modules=["fractions"])
    assert code == 0
    assert events[2]["path"] == "exec" and events[2]["shadowed"] == ["fractions"]
    # A fresh `python /app/python/main.py`: as for any script, sys.path[0] is the real folder of main.py
    assert os.path.realpath((tmp_path / "worker.local").read_text()) == str((app / "python" / "fractions.py").resolve()), (
        "the app gets its own module"
    )


def test_the_worker_refuses_to_run_another_app(tmp_path: Path):
    app, link = make_app(tmp_path)
    other = tmp_path / "apps" / "other"
    other.mkdir()
    link.unlink()
    link.symlink_to(other)
    code, _, _, events = run_worker(tmp_path, app, link, "return")
    assert code == 70
    assert events[-1]["code"] == "link_mismatch"
    assert not (tmp_path / "worker.top").exists()


def test_a_run_cuts_the_warm_up_short(tmp_path: Path):
    app, link = make_app(tmp_path)
    worker = WorkerProc(app, link, probe_env(tmp_path, "return", "worker"))
    assert worker.event()["event"] == "hello"
    worker.send({"cmd": "warm", "modules": ["fractions"] * 2000 + ["statistics"]})
    worker.send({"cmd": "run"})
    events = []
    while (event := worker.event()) is not None:
        events.append(event)
    code, _, _ = worker.wait()
    assert code == 0
    assert "started" in [e["event"] for e in events]


def test_quit_ends_an_idle_worker(tmp_path: Path):
    app, link = make_app(tmp_path)
    worker = WorkerProc(app, link, probe_env(tmp_path, "return", "worker"))
    worker.event()
    worker.send({"cmd": "warm", "modules": []})
    assert worker.event()["event"] == "ready"
    worker.send({"cmd": "quit"})
    assert worker.wait()[0] == 0
    assert not (tmp_path / "worker.top").exists()


def test_sigterm_inside_app_run_exits_143(tmp_path: Path):
    app, link = make_app(tmp_path)
    worker = WorkerProc(app, link, probe_env(tmp_path, "app_run", "worker"))
    worker.event()
    worker.send({"cmd": "warm", "modules": ["arduino.app_utils"]})
    ready = worker.event()
    assert ready["imported"] == ["arduino.app_utils"]
    worker.send({"cmd": "run"})
    assert worker.event()["event"] == "started"
    assert worker.event()["event"] == "app_run", "the worker reports when the app reaches App.run()"
    deadline = time.monotonic() + 20
    while not (tmp_path / "worker.running").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    os.kill(worker.proc.pid, signal.SIGTERM)
    code, out, _ = worker.wait()
    assert code == 143, out


RELOADABLE = """
import json, os, signal, sys, threading, time
import helper
VERSION = {version}
OUT = os.environ["PROBE_OUT"]
record = {{
    "version": VERSION,
    "helper": helper.VALUE,
    "pid": os.getpid(),
    "main_is_this_module": sys.modules["__main__"].__dict__ is globals(),
    "sighup": str(signal.getsignal(signal.SIGHUP)),
    "path0": sys.path[0],
    "cwd": os.getcwd(),
    "env": os.environ.get("PROBE_SET_BY_APP"),
}}
with open(OUT + ".runs", "a") as f:
    f.write(json.dumps(record) + "\\n")
# What a run may leave behind, which the next run must not see
signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["PROBE_SET_BY_APP"] = str(VERSION)
sys.path.insert(0, "/nowhere")
os.chdir("/")
{tail}
"""

APP_RUN_TAIL = "from arduino.app_utils import App\nApp.run()\n"


def reloadable_app(tmp_path: Path, version: int, tail: str = APP_RUN_TAIL) -> tuple[Path, Path]:
    return make_app(tmp_path, {"python/main.py": RELOADABLE.format(version=version, tail=tail), "python/helper.py": f"VALUE = {version}\n"})


def edit(app: Path, version: int, tail: str = APP_RUN_TAIL) -> None:
    (app / "python" / "main.py").write_text(RELOADABLE.format(version=version, tail=tail))
    (app / "python" / "helper.py").write_text(f"VALUE = {version}\n")


def runs(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "worker.runs"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_for(condition: Any, timeout: float = 20.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def next_event(worker: WorkerProc, name: str) -> dict[str, Any]:
    while (event := worker.event()) is not None:
        if event.get("event") == name:
            return event
    raise AssertionError(f"no {name} event")


def started_worker(tmp_path: Path, app: Path, link: Path, modules: list[str], env: dict[str, str] | None = None) -> WorkerProc:
    worker = WorkerProc(app, link, {**probe_env(tmp_path, "return", "worker"), **(env or {})})
    assert "reload" in next_event(worker, "hello")["features"]
    worker.send({"cmd": "warm", "modules": modules})
    next_event(worker, "ready")
    worker.send({"cmd": "run"})
    next_event(worker, "started")
    return worker


def test_a_reload_runs_the_edited_app_in_the_same_process(tmp_path: Path):
    app, link = reloadable_app(tmp_path, 1)
    worker = started_worker(tmp_path, app, link, ["arduino.app_utils"])
    next_event(worker, "app_run")
    wait_for(lambda: runs(tmp_path))

    edit(app, 2)
    worker.send({"cmd": "reload"})
    reloaded = next_event(worker, "reload_ok")
    assert reloaded["pid"] == worker.proc.pid and reloaded["purged"] >= 1, reloaded
    assert next_event(worker, "app_run"), "the trace reports the App.run() of every run"
    first, second = wait_for(lambda: len(runs(tmp_path)) == 2 and runs(tmp_path))

    assert (first["version"], second["version"]) == (1, 2)
    assert second["helper"] == 2, "the app's own modules are imported again"
    assert second["pid"] == first["pid"]
    assert second["main_is_this_module"]
    for key in ("sighup", "path0", "cwd", "env"):
        assert second[key] == first[key], f"{key}: the second run starts from the state of the first"

    os.kill(worker.proc.pid, signal.SIGTERM)
    code, out, _ = worker.wait()
    assert code == 143
    assert out.count("App is starting") == 2


def test_an_app_without_app_run_reloads_too(tmp_path: Path):
    loop = "while True:\n    time.sleep(0.05)\n"
    app, link = reloadable_app(tmp_path, 1, loop)
    worker = started_worker(tmp_path, app, link, [])
    wait_for(lambda: runs(tmp_path))
    edit(app, 2, loop)
    worker.send({"cmd": "reload"})
    next_event(worker, "reload_ok")
    assert [run["version"] for run in wait_for(lambda: len(runs(tmp_path)) == 2 and runs(tmp_path))] == [1, 2]
    os.kill(worker.proc.pid, signal.SIGTERM)
    worker.wait()


def test_a_thread_left_running_makes_the_reload_fall_back(tmp_path: Path):
    tail = "threading.Thread(target=time.sleep, args=(300,), name='stray', daemon=True).start()\n" + APP_RUN_TAIL
    app, link = reloadable_app(tmp_path, 1, tail)
    worker = started_worker(tmp_path, app, link, ["arduino.app_utils"])
    next_event(worker, "app_run")
    worker.send({"cmd": "reload"})
    fallback = next_event(worker, "reload_fallback")
    assert "stray" in fallback["reason"]
    assert len(runs(tmp_path)) == 1, "main.py does not run again"
    # The app is already stopped: a SIGTERM ends the process at once
    started = time.monotonic()
    os.kill(worker.proc.pid, signal.SIGTERM)
    code, _, _ = worker.wait()
    assert code == -signal.SIGTERM and time.monotonic() - started < 2.0


FAKE_CAMERA = """
from arduino.app_peripherals.device_registry import DeviceRegistry

REGISTRY = DeviceRegistry()
LEAKED = []


class FakeCamera:
    \"\"\"Claims its device as Camera() does: the claim goes when the object is collected.\"\"\"

    def __init__(self):
        key = REGISTRY.select(lambda: ["fake-cam"])
        if key is None:
            raise RuntimeError("fake-cam already in use")
        REGISTRY.bind(key, self)

    def stop(self):
        pass
"""


def fake_camera_env(tmp_path: Path) -> dict[str, str]:
    """A library module outside the app, so a reload keeps it, as it keeps arduino.app_peripherals."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "fakecam.py").write_text(FAKE_CAMERA)
    return {"PYTHONPATH": str(lib)}


def test_a_device_claimed_by_the_ended_run_is_free_for_the_next(tmp_path: Path):
    tail = "import fakecam\ncamera = fakecam.FakeCamera()\n" + APP_RUN_TAIL
    app, link = reloadable_app(tmp_path, 1, tail)
    worker = started_worker(tmp_path, app, link, ["arduino.app_utils"], fake_camera_env(tmp_path))
    next_event(worker, "app_run")
    edit(app, 2, tail)
    worker.send({"cmd": "reload"})
    next_event(worker, "reload_ok")
    assert next_event(worker, "app_run"), "the second run opened the device again"
    wait_for(lambda: len(runs(tmp_path)) == 2)
    time.sleep(0.3)  # app_run is reported on entering App.run(), before loop() takes SIGTERM over
    os.kill(worker.proc.pid, signal.SIGTERM)
    assert worker.wait()[0] == 143


def test_a_device_still_referenced_makes_the_reload_fall_back(tmp_path: Path):
    tail = "import fakecam\ncamera = fakecam.FakeCamera()\nfakecam.LEAKED.append(camera)\n" + APP_RUN_TAIL
    app, link = reloadable_app(tmp_path, 1, tail)
    worker = started_worker(tmp_path, app, link, ["arduino.app_utils"], fake_camera_env(tmp_path))
    next_event(worker, "app_run")
    worker.send({"cmd": "reload"})
    assert next_event(worker, "reload_fallback")["reason"] == "devices still claimed: fake-cam"
    os.kill(worker.proc.pid, signal.SIGTERM)
    worker.wait()


def test_a_reload_after_main_py_returned_falls_back(tmp_path: Path):
    tail = "threading.Thread(target=time.sleep, args=(300,), name='keeps-it-alive').start()\n"
    app, link = reloadable_app(tmp_path, 1, tail)
    worker = started_worker(tmp_path, app, link, [])
    wait_for(lambda: runs(tmp_path))
    time.sleep(0.2)
    worker.send({"cmd": "reload"})
    assert next_event(worker, "reload_fallback")["reason"] == "main.py has already returned"
    os.killpg(worker.proc.pid, signal.SIGKILL)
    worker.wait()


def test_an_immediate_worker_reports_app_run_too(tmp_path: Path):
    app, link = make_app(tmp_path)
    worker = WorkerProc(app, link, probe_env(tmp_path, "app_run", "worker"))
    worker.event()
    worker.send({"cmd": "run"})
    started = worker.event()
    assert started["event"] == "started" and started["path"] == "immediate"
    assert worker.event()["event"] == "app_run", "traced once the app imports arduino.app_utils"
    deadline = time.monotonic() + 20
    while not (tmp_path / "worker.running").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    os.kill(worker.proc.pid, signal.SIGTERM)
    assert worker.wait()[0] == 143
