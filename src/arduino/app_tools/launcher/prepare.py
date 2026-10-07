# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Dependencies of an app: the launcher leaves them to run.sh, the same script the classic container starts with."""

import sys

if sys.platform == "win32":
    raise ImportError("arduino-app-launcher needs a POSIX system")

import asyncio  # noqa: E402
import os  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402

from .appinfo import AppInfo  # noqa: E402

DEFAULT_RUN_SH = Path("/run.sh")


async def run_prepare(app: AppInfo, run_sh: Path, on_output: Callable[[bytes], None]) -> int:
    """Run `run.sh prepare` on an app folder: create its venv if missing, install what it requires, check its sources.

    The app is named by APP_DIR, its real path, since /app may point to another app meanwhile. Device provisioning
    is skipped: the launcher did it once for the container, and redoing it would rewrite files the running app uses.

    Returns:
        int: the exit code of run.sh.
    """
    env = {**os.environ, "APP_DIR": str(app.path), "SKIP_DEVICE_PROVISIONING": "1"}
    env.pop("VIRTUAL_ENV", None)
    proc = await asyncio.create_subprocess_exec(
        "sh",
        str(run_sh),
        "prepare",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(app.path),
        env=env,
    )
    assert proc.stdout is not None
    while True:
        chunk = await proc.stdout.read(65536)
        if not chunk:
            break
        on_output(chunk)
    return await proc.wait()
