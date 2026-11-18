# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The image installs the runtime and the model where the server loads them, readable by the user it runs as.

BuildKit applies a ``COPY``/``ADD --chmod`` mode to the parent directories the instruction creates
as well, so ``--chmod=644`` onto a new directory leaves it without the execute bit and the arduino
user cannot reach the files: EACCES at load. The Dockerfile is read as text, without Docker.
"""

import re
from pathlib import Path, PurePosixPath

from whistle_server import DEFAULT_LIB, DEFAULT_MODEL

DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"
# The directories of the files the server loads when the image runs it with no arguments
LOADED_DIRS = [PurePosixPath(DEFAULT_LIB.as_posix()).parent, PurePosixPath(DEFAULT_MODEL.as_posix()).parent]


def _final_stage_copies() -> list[tuple[list[str], PurePosixPath]]:
    """The COPY and ADD instructions of the last stage, as (flags, destination)."""
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text(encoding="utf-8"))
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    last_from = max(i for i, line in enumerate(lines) if line.upper().startswith("FROM "))
    copies: list[tuple[list[str], PurePosixPath]] = []
    for line in lines[last_from + 1 :]:
        words = line.split()
        if words[0].upper() in ("COPY", "ADD"):
            copies.append(([w for w in words[1:] if w.startswith("--")], PurePosixPath(words[-1])))
    return copies


def _under(dest: PurePosixPath, directory: PurePosixPath) -> bool:
    return dest == directory or directory in dest.parents


def test_the_image_installs_the_files_where_the_server_loads_them():
    destinations = [dest for _, dest in _final_stage_copies()]
    for directory in LOADED_DIRS:
        assert any(_under(dest, directory) for dest in destinations), f"nothing is copied to {directory}"


def test_the_files_are_copied_without_chmod():
    for flags, dest in _final_stage_copies():
        if any(_under(dest, directory) for directory in LOADED_DIRS):
            assert not [f for f in flags if f.startswith("--chmod")], f"{dest}: --chmod would also apply to the directories it creates"
