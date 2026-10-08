# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""What the launcher knows about an app folder: where it is, what it declares, and when its worker is stale."""

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

VENV_PYTHON = Path(".cache") / ".venv" / "bin" / "python"
"""The app interpreter, relative to the app folder: the venv run.sh creates."""

LAUNCHER_CACHE = Path(".cache") / "launcher"
"""Where the launcher keeps its own per-app records, relative to the app folder."""

STREAMLIT_BRICK = "arduino:streamlit_ui"


class AppNotFound(Exception):
    """The reference names no app folder inside the apps directory."""


@dataclass(frozen=True)
class AppInfo:
    """An app folder, read once: the supervisor reloads it when the folder changes."""

    name: str
    path: Path
    """Real path of the app folder."""
    brick_ids: tuple[str, ...]
    streamlit: bool

    @property
    def python_dir(self) -> Path:
        return self.path / "python"

    @property
    def bricks_dir(self) -> Path:
        return self.path / "bricks"

    @property
    def venv_python(self) -> Path:
        return self.path / VENV_PYTHON

    @property
    def launcher_cache(self) -> Path:
        return self.path / LAUNCHER_CACHE


def is_app_dir(path: Path) -> bool:
    """An app folder has an app.yaml and a python/main.py, as run.sh expects."""
    return (path / "app.yaml").is_file() and (path / "python" / "main.py").is_file()


def discover_apps(apps_dir: Path) -> list[str]:
    """Names of the app folders directly inside apps_dir, sorted."""
    try:
        entries = sorted(apps_dir.iterdir())
    except OSError:
        return []
    return [entry.name for entry in entries if not entry.name.startswith(".") and entry.is_dir() and is_app_dir(entry)]


def resolve_app_path(apps_dir: Path, ref: str) -> Path:
    """Resolve an app name, or a path, to the real path of an app folder inside apps_dir.

    Raises:
        AppNotFound: if the reference points outside apps_dir or to a folder that is not an app.
    """
    if not ref or ref in (".", ".."):
        raise AppNotFound(f"no app named {ref!r}")
    candidate = Path(ref) if ("/" in ref or os.sep in ref) else apps_dir / ref
    real = Path(os.path.realpath(candidate))
    root = Path(os.path.realpath(apps_dir))
    if real.parent != root:
        raise AppNotFound(f"{ref!r} is not an app folder of {apps_dir}")
    if not is_app_dir(real):
        raise AppNotFound(f"{ref!r} has no app.yaml and python/main.py")
    return real


def parse_brick_ids(app_yaml: object) -> list[str]:
    """The brick ids an app.yaml declares, in order.

    Entries are a bare id (`- arduino:web_ui`) or a one-key mapping to the brick settings
    (`- arduino:llm: {model: ...}`).
    """
    if not isinstance(app_yaml, dict):
        return []
    bricks: Any = app_yaml.get("bricks")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(bricks, list):
        return []
    ids: list[str] = []
    for entry in bricks:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            ids.extend(str(key) for key in entry)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return ids


def is_streamlit_app(app_yaml_text: str) -> bool:
    """Same test as run.sh (`grep -q arduino:streamlit_ui app.yaml`), so both start the same apps with Streamlit."""
    return STREAMLIT_BRICK in app_yaml_text


def load_app(path: Path) -> AppInfo:
    """Read an app folder."""
    text = (path / "app.yaml").read_text(encoding="utf-8", errors="replace")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        parsed = None
    return AppInfo(name=path.name, path=path, brick_ids=tuple(parse_brick_ids(parsed)), streamlit=is_streamlit_app(text))


@cache
def brick_module_index() -> dict[str, str]:
    """Map the id of every installed brick to its module, from the `id:` of each brick_config.yaml.

    The id is not always the package name: arduino:video_object_detection lives in
    arduino.app_bricks.video_objectdetection.
    """
    index: dict[str, str] = {}
    try:
        spec = importlib.util.find_spec("arduino.app_bricks")
    except (ImportError, ValueError):
        return index
    if spec is None or not spec.submodule_search_locations:
        return index
    for location in spec.submodule_search_locations:
        for config in sorted(Path(location).glob("*/brick_config.yaml")):
            try:
                brick_id = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("id")
            except (OSError, yaml.YAMLError, AttributeError):
                continue
            if isinstance(brick_id, str):
                index.setdefault(brick_id, f"arduino.app_bricks.{config.parent.name}")
    return index


def brick_modules(brick_ids: tuple[str, ...] | list[str]) -> list[str]:
    """Modules of the bricks an app declares; ids with no installed brick, e.g. app-local ones, are skipped."""
    index = brick_module_index()
    return [index[brick_id] for brick_id in brick_ids if brick_id in index]


def _compose_main_environment(compose_file: Path) -> dict[str, str]:
    try:
        compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        environment: Any = compose["services"]["main"]["environment"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return {}
    env: dict[str, str] = {}
    if isinstance(environment, dict):
        for key, value in environment.items():  # pyright: ignore[reportUnknownVariableType]
            env[str(key)] = "" if value is None else str(value)  # pyright: ignore[reportUnknownArgumentType]
    elif isinstance(environment, list):
        for item in environment:  # pyright: ignore[reportUnknownVariableType]
            key, _, value = str(item).partition("=")  # pyright: ignore[reportUnknownArgumentType]
            env[key] = value
    return env


def compose_env(app_path: Path) -> dict[str, str]:
    """The environment arduino-app-cli gave the main container of this app the last time it ran it.

    Read from the compose files the CLI leaves in the app cache, the overrides on top. It stands in
    for the environment a start request should carry, until arduino-app-cli sends it.
    """
    env = _compose_main_environment(app_path / ".cache" / "app-compose.yaml")
    env.update(_compose_main_environment(app_path / ".cache" / "app-compose-overrides.yaml"))
    return env


def _file_token(label: str, path: Path) -> bytes:
    """A labelled file content to feed a digest with, or a marker when the file is missing."""
    try:
        content = path.read_bytes()
    except OSError:
        content = b"<missing>"
    return label.encode() + b"\0" + content + b"\0"


def deps_fingerprint(app_path: Path) -> str:
    """Digest of what run.sh installs into the app venv: requirements of the app and of its bricks, private wheels.

    A wheel counts with the same name before and after run.sh renames it to .whl.installed, otherwise
    every prepare would change the fingerprint it is recorded under.
    """
    digest = hashlib.sha256()
    digest.update(_file_token("python/requirements.txt", app_path / "python" / "requirements.txt"))
    bricks_dir = app_path / "bricks"
    if bricks_dir.is_dir():
        for requirements in sorted(bricks_dir.glob("*/requirements.txt")):
            digest.update(_file_token(f"bricks/{requirements.parent.name}/requirements.txt", requirements))
    libs_dir = app_path / "python-libraries"
    if libs_dir.is_dir():
        wheels: list[tuple[str, int]] = []
        for wheel in libs_dir.iterdir():
            name = wheel.name.removesuffix(".installed")
            if name.endswith(".whl"):
                try:
                    wheels.append((name, wheel.stat().st_size))
                except OSError:
                    continue
        for name, size in sorted(wheels):
            digest.update(f"wheel {name} {size}\0".encode())
    return digest.hexdigest()


def _deps_record(app: AppInfo) -> Path:
    return app.launcher_cache / "deps.sha"


def _has_requirements(path: Path) -> bool:
    try:
        return any(line.strip() for line in path.read_text().splitlines())
    except OSError:
        return False


def needs_venv(app: AppInfo) -> bool:
    """Whether the app runs in its own venv, by the rule of run.sh: a venv that exists is kept, otherwise one is made
    only for what the app adds to the image, i.e. requirements, private wheels, brick requirements or Streamlit."""
    if app.venv_python.parent.parent.is_dir() or (app.path / "python-libraries").is_dir():
        return True
    if _has_requirements(app.python_dir / "requirements.txt"):
        return True
    if app.bricks_dir.is_dir() and any(_has_requirements(path) for path in app.bricks_dir.glob("*/requirements.txt")):
        return True
    return app.streamlit


def needs_prepare(app: AppInfo) -> bool:
    """Whether run.sh prepare has to run before the app gets a worker: never prepared, its dependencies changed, or
    the venv they go into is missing. An app that adds nothing to the image has no venv and needs none."""
    if needs_venv(app) and not app.venv_python.exists():
        return True
    try:
        recorded = _deps_record(app).read_text().strip()
    except OSError:
        return True
    return recorded != deps_fingerprint(app.path)


def record_prepared(app: AppInfo) -> None:
    """Remember the dependencies a successful prepare installed."""
    record = _deps_record(app)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(deps_fingerprint(app.path) + "\n")


def app_interpreter(app: AppInfo, fallback: str) -> str:
    """The interpreter of the app venv, as run.sh runs main.py with; fallback when the app has no venv."""
    return str(app.venv_python) if app.venv_python.exists() else fallback


def installed_packages(app: AppInfo) -> list[str]:
    """What is installed in the app venv: its distributions and .pth files, by name, sorted.

    Not the folder modification time: the interpreter itself changes it, writing __pycache__ there.
    """
    names: list[str] = []
    for site_packages in sorted((app.path / ".cache" / ".venv" / "lib").glob("python*/site-packages")):
        try:
            names.extend(entry.name for entry in site_packages.iterdir() if entry.name.endswith((".dist-info", ".egg-info", ".egg-link", ".pth")))
        except OSError:
            continue
    return sorted(names)


def app_fingerprint(app: AppInfo, env: dict[str, str], interpreter: str, extra: str = "") -> str:
    """Digest of everything a warm worker depends on: when it changes, the worker is replaced.

    main.py and the app's own modules are not part of it: a worker reads them only when it runs the app.
    """
    digest = hashlib.sha256()
    digest.update(_file_token("app.yaml", app.path / "app.yaml"))
    digest.update(deps_fingerprint(app.path).encode() + b"\0")
    digest.update("\0".join(installed_packages(app)).encode())
    digest.update(b"\0" + interpreter.encode() + b"\0")
    for key in sorted(env):
        digest.update(f"{key}={env[key]}\0".encode())
    digest.update(extra.encode())
    return digest.hexdigest()
