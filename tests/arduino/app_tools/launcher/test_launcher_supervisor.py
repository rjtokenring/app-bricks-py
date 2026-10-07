# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The supervisor end to end: real workers, a fake run.sh, apps that record each run."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the launcher needs a POSIX system")

if sys.platform != "win32":
    from arduino.app_tools.launcher.client import Client
    from arduino.app_tools.launcher.server import Config, PrepareFailed, Supervisor

RECORDING_APP = """
import json, os, sys, time
VERSION = "{version}"
os.makedirs("data", exist_ok=True)
ENV = {{k: os.environ.get(k) for k in ("APP_HOME", "VIDEO_DEVICE", "APP_SHUTDOWN_GRACE_PERIOD_S")}}
with open(os.path.join("data", "runs.jsonl"), "a") as f:
    f.write(json.dumps({{"version": VERSION, "pid": os.getpid(), "file": __file__, "env": ENV}}) + "\\n")
{tail}
"""

LOOP = "while True:\n    time.sleep(0.05)\n"

STUBBORN_TAIL = """
import signal, subprocess
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(["sleep", "300"])
with open(os.path.join("data", "child.pid"), "w") as f:
    f.write(str(child.pid))
while True:
    time.sleep(0.05)
"""

FAKE_RUN_SH = """#!/bin/sh
echo "$1 $APP_DIR SKIP_DEVICE_PROVISIONING=$SKIP_DEVICE_PROVISIONING" >> "$FAKE_PREPARE_LOG"
mkdir -p "$APP_DIR/.cache/.venv/bin"
printf '#!/bin/sh\\nexec "%s" "$@"\\n' "$FAKE_PYTHON" > "$APP_DIR/.cache/.venv/bin/python"
chmod +x "$APP_DIR/.cache/.venv/bin/python"
if [ -f "$APP_DIR/slow_prepare" ]; then sleep 1; fi
if [ -f "$APP_DIR/fail_prepare" ]; then exit 1; fi
"""


class Env:
    def __init__(self, root: Path, sock_dir: Path) -> None:
        self.root = root
        self.apps = root / "apps"
        self.apps.mkdir()
        self.prepare_log = root / "prepare.log"
        self.prepare_log.write_text("")
        run_sh = root / "run.sh"
        run_sh.write_text(FAKE_RUN_SH)
        run_sh.chmod(0o755)
        launcher = root / "launcher"
        self.config = Config(
            apps_dir=self.apps,
            socket_path=sock_dir / "l.sock",
            app_link=str(launcher / "current"),
            current_link=launcher / "current",
            idle_dir=launcher / "none",
            state_file=launcher / "state.json",
            run_sh=run_sh,
            stop_timeout_s=1.0,
            settle_s=0.0,
            warm_concurrency=2,
            warm_timeout_s=60.0,
            run_timeout_s=30.0,
            replacement_delay_s=0.2,
            mem_reserve_mb=0,
            max_failures=1,
            preload=("fractions",),
            interpreter_fallback=sys.executable,
        )

    def app(self, name: str, version: str = "1", tail: str = LOOP, files: dict[str, str] | None = None) -> Path:
        app = self.apps / name
        (app / "python").mkdir(parents=True, exist_ok=True)
        (app / "python" / "main.py").write_text(RECORDING_APP.format(version=version, tail=tail))
        (app / "app.yaml").write_text(f"name: {name}\nbricks: []\n")
        for relative, content in (files or {}).items():
            path = app / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return app

    def runs(self, name: str) -> list[dict[str, Any]]:
        log = self.apps / name / "data" / "runs.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def link_target(self) -> str:
        return os.path.realpath(self.config.current_link)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    sock_dir = Path(tempfile.mkdtemp(prefix="lch", dir="/tmp"))  # AF_UNIX paths are short
    environment = Env(tmp_path, sock_dir)
    monkeypatch.setenv("FAKE_PREPARE_LOG", str(environment.prepare_log))
    monkeypatch.setenv("FAKE_PYTHON", sys.executable)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    yield environment
    shutil.rmtree(sock_dir, ignore_errors=True)


async def wait_until(condition: Callable[[], Any], timeout: float = 30.0, what: str = "condition") -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def worker_state(supervisor: "Supervisor", name: str) -> str | None:
    slot = supervisor.slots.get(name)
    return str(slot.worker.state) if slot is not None and slot.worker is not None else None


def pid_gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            return any(line.startswith("State:") and "Z" in line.split()[1] for line in f)
    except FileNotFoundError:
        return True


def run_scenario(env: Env, scenario: Callable[["Supervisor"], Awaitable[None]], config: "Config | None" = None) -> None:
    async def main() -> None:
        supervisor = Supervisor(config or env.config)
        await supervisor.open()
        try:
            await scenario(supervisor)
        finally:
            await supervisor.close()

    asyncio.run(main())


def test_an_app_starts_from_its_warm_worker(env: Env):
    app = env.app("a")

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "a") == "ready", what="a ready")
        assert env.link_target() == str(env.config.idle_dir.resolve()), "no app runs yet"
        reply = await supervisor.start("a")
        assert reply["ok"] and reply["path"] == "warm" and reply["worker"]["was_ready"]
        runs = await wait_until(lambda: env.runs("a"), what="a run")
        assert runs[0]["pid"] == reply["pid"]
        assert runs[0]["file"] == f"{env.config.app_link}/python/main.py", "the app sees itself under the link"
        assert env.link_target() == str(app.resolve())
        assert runs[0]["env"]["APP_SHUTDOWN_GRACE_PERIOD_S"] == "1.0", "the app shutdown fits the stop timeout"

    run_scenario(env, scenario)


def test_switching_stops_the_running_app_first_and_rewarms_it(env: Env):
    env.app("a")
    b = env.app("b")

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "a") == "ready" and worker_state(supervisor, "b") == "ready", what="both ready")
        first = await supervisor.start("a")
        await wait_until(lambda: env.runs("a"))
        second = await supervisor.start("b")
        assert second["stopped"]["signal"] == "SIGTERM" and not second["stopped"]["killed"]
        assert pid_gone(first["pid"]), "a is gone before b starts"
        assert env.link_target() == str(b.resolve())
        assert supervisor.active is not None and supervisor.active.slot.info.name == "b"
        await wait_until(lambda: worker_state(supervisor, "a") == "ready", what="a warm again")
        assert supervisor.slots["a"].last_exit["signal"] == "SIGTERM"

    run_scenario(env, scenario)


def test_a_restart_runs_the_edited_main_py(env: Env):
    app = env.app("a")

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "a") == "ready")
        first = await supervisor.start("a")
        await wait_until(lambda: env.runs("a"))
        # The next worker is warming while the app runs
        await wait_until(lambda: worker_state(supervisor, "a") == "ready", what="spare worker")
        (app / "python" / "main.py").write_text(RECORDING_APP.format(version="2", tail=LOOP))
        second = await supervisor.restart(None)
        assert second["worker"]["was_ready"], "the restart used the spare worker"
        assert second["pid"] != first["pid"] and pid_gone(first["pid"])
        runs = await wait_until(lambda: len(env.runs("a")) == 2 and env.runs("a"))
        assert [run["version"] for run in runs] == ["1", "2"]

    run_scenario(env, scenario)


def test_an_app_that_ignores_sigterm_is_killed_with_its_children(env: Env):
    app = env.app("stubborn", tail=STUBBORN_TAIL)

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "stubborn") == "ready")
        await supervisor.start("stubborn")
        child_pid_file = app / "data" / "child.pid"
        await wait_until(child_pid_file.exists, what="child")
        child = int(child_pid_file.read_text())
        started = time.monotonic()
        reply = await supervisor.stop()
        elapsed = time.monotonic() - started
        assert reply["stopped"]["killed"] and reply["stopped"]["signal"] == "SIGKILL"
        assert 1.0 <= elapsed < 4.0, elapsed
        await wait_until(lambda: pid_gone(child), timeout=5, what="child killed")
        assert env.link_target() == str(env.config.idle_dir.resolve())

    run_scenario(env, scenario)


def test_an_app_that_exits_frees_the_slot(env: Env):
    env.app("short", tail="sys.exit(5)\n")

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "short") == "ready")
        events: asyncio.Queue[Any] = asyncio.Queue()
        supervisor._subscribers.add(events)  # pyright: ignore[reportPrivateUsage]
        await supervisor.start("short")
        exited = await wait_until(lambda: supervisor.active is None and supervisor.slots["short"].last_exit, what="exit")
        assert exited["exit_code"] == 5
        assert env.link_target() == str(env.config.idle_dir.resolve())
        names = []
        while not events.empty():
            names.append(events.get_nowait()["event"])
        assert "app_started" in names and "app_exited" in names
        await wait_until(lambda: worker_state(supervisor, "short") == "ready", what="rewarmed")

    run_scenario(env, scenario)


def test_prepare_runs_once_for_an_app_without_a_venv(env: Env):
    app = env.app("a")

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "a") == "ready")
        assert (app / ".cache" / ".venv" / "bin" / "python").exists()
        assert env.prepare_log.read_text().splitlines() == [f"prepare {app.resolve()} SKIP_DEVICE_PROVISIONING=1"]
        assert supervisor.slots["a"].worker.spec.interpreter == str(app.resolve() / ".cache" / ".venv" / "bin" / "python")
        await supervisor.start("a")
        await supervisor.stop()
        assert len(env.prepare_log.read_text().splitlines()) == 1, "the dependencies did not change"
        (app / "python" / "requirements.txt").write_text("psutil\n")
        await supervisor.start("a")
        assert len(env.prepare_log.read_text().splitlines()) == 2

    run_scenario(env, scenario)


def test_a_failed_prepare_is_reported(env: Env):
    env.app("broken", files={"fail_prepare": ""})

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: supervisor.slots["broken"].warm_error, what="warm error")
        with pytest.raises(PrepareFailed):
            await supervisor.start("broken")
        reply = await supervisor.dispatch({"cmd": "start", "app": "broken"})
        assert reply["error"]["code"] == "prepare_failed"

    run_scenario(env, scenario)


def test_a_worker_dying_while_warming_is_reported_and_the_app_still_starts(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env.app("a")
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "crashmod.py").write_text("import os\nprint('crashing on import', flush=True)\nos._exit(3)\n")
    monkeypatch.setenv("PYTHONPATH", str(modules))
    config = env.config
    config.preload = ("crashmod",)

    async def scenario(supervisor: Supervisor) -> None:
        error = await wait_until(lambda: supervisor.slots["a"].warm_error, what="warm error")
        assert "exited" in error
        assert supervisor.slots["a"].failures == 1
        reply = await supervisor.start("a")
        assert reply["ok"] and reply["worker"]["mode"] == "immediate"
        await wait_until(lambda: env.runs("a"))

    run_scenario(env, scenario, config)


def test_the_env_comes_from_the_compose_file_or_the_request(env: Env):
    app = env.app(
        "a", files={".cache/app-compose.yaml": "services:\n  main:\n    environment:\n      VIDEO_DEVICE: /dev/video1\n      APP_HOME: /host/a\n"}
    )

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: worker_state(supervisor, "a") == "ready")
        await supervisor.start("a")
        first = await wait_until(lambda: env.runs("a"))
        assert first[0]["env"]["VIDEO_DEVICE"] == "/dev/video1" and first[0]["env"]["APP_HOME"] == "/host/a"
        reply = await supervisor.start("a", env={"VIDEO_DEVICE": "/dev/video2"})
        assert not reply["worker"]["was_ready"], "a different environment needs a different worker"
        runs = await wait_until(lambda: len(env.runs("a")) == 2 and env.runs("a"))
        assert runs[1]["env"]["VIDEO_DEVICE"] == "/dev/video2"
        assert runs[1]["env"]["APP_HOME"] == str(app.resolve()), "APP_HOME defaults to the app folder"

    run_scenario(env, scenario)


def test_the_control_socket(env: Env):
    env.app("a")

    async def scenario(supervisor: Supervisor) -> None:
        client = Client(env.config.socket_path)
        await wait_until(lambda: worker_state(supervisor, "a") == "ready")
        assert (await asyncio.to_thread(client.call, "ping"))["ok"]
        status = await asyncio.to_thread(client.call, "status")
        assert [app["name"] for app in status["apps"]] == ["a"] and status["apps"][0]["worker"]["state"] == "ready"
        started = await asyncio.to_thread(client.call, "start", app="a")
        assert started["ok"] and started["id"] == 3
        assert (await asyncio.to_thread(client.call, "status"))["active"]["app"] == "a"
        await wait_until(lambda: env.runs("a"))
        assert (await asyncio.to_thread(client.call, "stop"))["stopped"]["signal"] == "SIGTERM"
        assert (await asyncio.to_thread(client.call, "start", app="missing"))["error"]["code"] == "not_found"
        assert (await asyncio.to_thread(client.call, "start"))["error"]["code"] == "bad_request"
        assert (await asyncio.to_thread(client.call, "nope"))["error"]["code"] == "bad_request"

    run_scenario(env, scenario)


def test_a_warm_request_during_a_warm_up_does_not_spawn_a_second_worker(env: Env):
    env.app("a", files={"slow_prepare": ""})

    async def scenario(supervisor: Supervisor) -> None:
        await wait_until(lambda: supervisor.slots["a"].warming, what="warm-up begun")
        supervisor.warm("a")
        supervisor.warm("a")
        await wait_until(lambda: worker_state(supervisor, "a") == "ready")
        await asyncio.sleep(1.0)
        workers = [p for p in Path("/proc").iterdir() if p.name.isdigit() and _is_worker_of(p, env.apps / "a")]
        assert len(workers) == 1, workers

    run_scenario(env, scenario)


def _is_worker_of(proc: Path, app: Path) -> bool:
    try:
        cmdline = (proc / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(part.endswith(b"worker.py") for part in cmdline) and str(app.resolve()).encode() in cmdline
