# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""run.sh of python-apps-base, as arduino-app-launcher calls it: on any app folder, not only /app."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUN_SH = Path(__file__).parents[2] / "containers" / "bricks" / "python-apps-base" / "scripts" / "run.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None, reason="needs a POSIX sh")

# The tools run.sh calls, replaced by loggers. The fake `uv venv` writes an activate script that names
# another venv, as the venvs created at /app/.cache/.venv do when reached from another path.
FAKE_TOOLS = {
    "uv": """#!/bin/sh
echo "uv $* VIRTUAL_ENV=${VIRTUAL_ENV:-none}" >> "$FAKE_LOG"
if [ "$1" = venv ]; then mkdir -p "$2/bin" && echo "export VIRTUAL_ENV=/app/.cache/.venv" > "$2/bin/activate"; fi
if [ "$1" = pip ] && [ "$2" = show ]; then exit 1; fi
""",
    "python": """#!/bin/sh
echo "python $* VIRTUAL_ENV=${VIRTUAL_ENV:-none} PWD=$(pwd)" >> "$FAKE_LOG"
""",
    "bash": """#!/bin/sh
echo "bash $*" >> "$FAKE_LOG"
""",
}

STREAMLIT_APP_YAML = "name: t\nbricks:\n- arduino:streamlit_ui: {}\n"


def _run(app: Path, tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    run_sh = tmp_path / "run.sh"
    run_sh.write_text(RUN_SH.read_text().replace("\r\n", "\n"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in FAKE_TOOLS.items():
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {
        **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "FAKE_LOG": str(log),
        "APP_DIR": str(app),
        **(extra_env or {}),
    }
    result = subprocess.run(["sh", str(run_sh), *args], env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    return result, log.read_text().splitlines()


def _app(tmp_path: Path, files: dict[str, str] | None = None, app_yaml: str = "name: t\nbricks: []\n") -> Path:
    app = tmp_path / "apps" / "one"
    (app / "python").mkdir(parents=True)
    (app / "python" / "main.py").write_text("print('hi')\n")
    (app / "app.yaml").write_text(app_yaml)
    for relative, content in (files or {}).items():
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return app


def test_app_dir_names_the_app_folder(tmp_path: Path):
    app = _app(tmp_path, {"python/requirements.txt": "requests\n"})  # Something to install: the app gets a venv
    result, calls = _run(app, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    venv = app / ".cache" / ".venv"
    assert f"uv venv {venv} --system-site-packages --relocatable VIRTUAL_ENV=none" in calls
    assert calls[-1] == f"python {app}/python/main.py VIRTUAL_ENV={venv} PWD={app}"


def test_the_venv_is_the_app_one_whatever_its_activate_script_says(tmp_path: Path):
    app = _app(tmp_path, {"python/requirements.txt": "requests\n"})
    _, calls = _run(app, tmp_path)
    install = [c for c in calls if c.startswith("uv pip install")]
    assert install and all(c.endswith(f"VIRTUAL_ENV={app}/.cache/.venv") for c in install), calls


def test_device_provisioning_can_be_skipped(tmp_path: Path):
    app = _app(tmp_path)
    _, calls = _run(app, tmp_path)
    assert "bash /provision-alsa-devices.sh" in calls
    _, calls = _run(app, tmp_path, extra_env={"SKIP_DEVICE_PROVISIONING": "1"})
    assert not any(c.startswith("bash /provision") for c in calls), calls


def test_prepare_leaves_no_temporary_folder(tmp_path: Path):
    app = _app(tmp_path)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    result, calls = _run(app, tmp_path, "prepare", extra_env={"TMPDIR": str(tmpdir), "SKIP_DEVICE_PROVISIONING": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(c.startswith("python -m compileall") for c in calls)
    assert list(tmpdir.iterdir()) == []


def test_streamlit_runs_as_a_module(tmp_path: Path):
    app = _app(tmp_path, app_yaml=STREAMLIT_APP_YAML)
    _, calls = _run(app, tmp_path)
    assert calls[-1].startswith(f"python -m streamlit run --server.port 7000 {app}/python/main.py"), calls
