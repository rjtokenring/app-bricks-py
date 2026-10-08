# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Which modules a worker imports in advance for an app, and which names belong to the app itself."""

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path

SKIPPED_DIRS = {"__pycache__", ".cache", ".git", "node_modules"}

DEFAULT_PRELOAD = ("numpy", "yaml", "arduino.app_utils")
"""Imported by every worker, whatever its app: libraries most apps load, cheap enough to keep for the others.

Not cv2 (~0.3 s and ~23 MiB per worker on an UNO Q) nor PIL: most bricks that need them import them anyway, and an
app that opens a camera itself gets CAMERA_MODULES.
"""

CAMERA_PACKAGE = "arduino.app_peripherals.camera"

CAMERA_MODULES = (f"{CAMERA_PACKAGE}.v4l_camera", f"{CAMERA_PACKAGE}.csi_camera")
"""The backends Camera() imports to open a local camera, cv2 among their dependencies. Neither touches a device at
import, so both are safe to import on a board that has only one kind of camera."""

WEB_UI_BRICK = "arduino:web_ui"

WEB_UI_MODULES = ("arduino.app_bricks.web_ui", "fastapi", "fastapi_socketio", "starlette", "uvicorn", "socketio", "engineio")
"""The web stack of the web_ui brick: imported in advance only for an app that declares the brick in app.yaml."""


def _python_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.py")):
        if not SKIPPED_DIRS.intersection(path.relative_to(root).parts):
            yield path


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _module_level_imports(statements: list[ast.stmt]) -> Iterator[str]:
    """Absolute imports run when the module is imported: function bodies and `if TYPE_CHECKING:` are left out."""
    for node in statements:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _module_level_imports(node.body)
            yield from _module_level_imports(node.orelse)
        elif isinstance(node, ast.Try | ast.TryStar):
            yield from _module_level_imports(node.body)
            for handler in node.handlers:
                yield from _module_level_imports(handler.body)
            yield from _module_level_imports(node.orelse)
            yield from _module_level_imports(node.finalbody)
        elif isinstance(node, ast.With | ast.ClassDef):
            yield from _module_level_imports(node.body)


def scan_imports(roots: Iterable[Path]) -> list[str]:
    """Absolute module-level imports of the Python files under the roots, in first-seen order.

    Files that do not parse are skipped: the app reports its own syntax errors when it runs.
    """
    seen: dict[str, None] = {}
    for root in roots:
        for path in _python_files(root):
            try:
                tree = ast.parse(path.read_bytes(), filename=str(path))
            except (SyntaxError, ValueError, OSError):
                continue
            for name in _module_level_imports(tree.body):
                seen.setdefault(name, None)
    return list(seen)


def local_module_names(*roots: Path) -> set[str]:
    """Top-level names the app's own folders provide on sys.path: its python/ and bricks/ folders.

    A module with one of these names must come from the app, never from a library imported in advance.
    """
    names: set[str] = set()
    for root in roots:
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in SKIPPED_DIRS or entry.name.startswith("."):
                continue
            if entry.is_dir():
                if entry.name.isidentifier():
                    names.add(entry.name)
            elif entry.suffix == ".py" and entry.stem.isidentifier():
                names.add(entry.stem)
    return names


def uses_camera(required_devices: Iterable[str], scanned: Iterable[str]) -> bool:
    """Whether an app opens a camera: one of its bricks requires it, or its own code imports the camera package."""
    return "camera" in required_devices or any(name == CAMERA_PACKAGE or name.startswith(CAMERA_PACKAGE + ".") for name in scanned)


def excluded_modules(brick_ids: Iterable[str]) -> tuple[str, ...]:
    """Modules a worker must not import in advance for an app with these bricks: the web stack without web_ui."""
    return () if WEB_UI_BRICK in brick_ids else WEB_UI_MODULES


def _is_excluded(name: str, excluded: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in excluded)


def warm_candidates(
    preload: Iterable[str],
    brick_modules: Iterable[str],
    scanned: Iterable[str],
    local_names: set[str],
    excluded: Iterable[str] = (),
) -> list[str]:
    """The modules a worker tries to import, in order: the preload list, the bricks, then what the app imports.

    Names whose top-level package is one of the app's own modules are dropped, and so are the excluded modules
    with their submodules, from every group.
    """
    excluded = tuple(excluded)
    seen: dict[str, None] = {}
    for group in (preload, brick_modules, scanned):
        for name in group:
            name = name.strip()
            if name and name.partition(".")[0] not in local_names and not _is_excluded(name, excluded):
                seen.setdefault(name, None)
    return list(seen)
