# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import dataclasses
from pathlib import Path

import pytest

from arduino.app_tools.launcher import appinfo


def make_app(apps_dir: Path, name: str, app_yaml: str = "name: t\nbricks:\n- arduino:web_ui: {}\n", files: dict[str, str] | None = None) -> Path:
    app = apps_dir / name
    (app / "python").mkdir(parents=True)
    (app / "python" / "main.py").write_text("print('hi')\n")
    (app / "app.yaml").write_text(app_yaml)
    for relative, content in (files or {}).items():
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return app


def test_apps_are_the_folders_with_app_yaml_and_main_py(tmp_path: Path):
    make_app(tmp_path, "b")
    make_app(tmp_path, "a")
    (tmp_path / "not-an-app").mkdir()
    (tmp_path / "half" / "python").mkdir(parents=True)
    (tmp_path / "half" / "app.yaml").write_text("name: x\n")
    make_app(tmp_path, ".hidden")
    assert appinfo.discover_apps(tmp_path) == ["a", "b"]


def test_an_app_resolves_by_name_or_path(tmp_path: Path):
    app = make_app(tmp_path, "one")
    assert appinfo.resolve_app_path(tmp_path, "one") == app.resolve()
    assert appinfo.resolve_app_path(tmp_path, str(app)) == app.resolve()


@pytest.mark.parametrize("ref", ["", ".", "..", "../one", "missing", "one/python"])
def test_nothing_outside_the_apps_folder_resolves(tmp_path: Path, ref: str):
    apps = tmp_path / "apps"
    apps.mkdir()
    make_app(apps, "one")
    make_app(tmp_path, "outside")
    with pytest.raises(appinfo.AppNotFound):
        appinfo.resolve_app_path(apps, ref if ref != "../one" else str(apps / ".." / "outside"))


def test_brick_ids_come_bare_or_with_settings():
    parsed = {"bricks": ["arduino:vlm", {"arduino:llm": {"model": "x"}}, {"arduino:web_ui": {}}, 3]}
    assert appinfo.parse_brick_ids(parsed) == ["arduino:vlm", "arduino:llm", "arduino:web_ui"]
    assert appinfo.parse_brick_ids(None) == []
    assert appinfo.parse_brick_ids({"bricks": None}) == []


def test_streamlit_apps_are_told_as_run_sh_does():
    assert appinfo.is_streamlit_app("bricks:\n- arduino:streamlit_ui: {}\n")
    assert not appinfo.is_streamlit_app("bricks:\n- arduino:web_ui: {}\n")


def test_brick_ids_map_to_their_package_not_their_name():
    # Ids that differ from the package folder, as on the board apps
    modules = appinfo.brick_modules(["arduino:video_object_detection", "arduino:video_image_classification", "arduino:web_ui", "custom:mine"])
    assert modules == [
        "arduino.app_bricks.video_objectdetection",
        "arduino.app_bricks.video_imageclassification",
        "arduino.app_bricks.web_ui",
    ]


def test_the_env_comes_from_the_cli_compose_files(tmp_path: Path):
    app = make_app(
        tmp_path,
        "one",
        files={
            ".cache/app-compose.yaml": (
                "services:\n  main:\n    environment:\n      APP_HOME: /h/one\n      VIDEO_DEVICE: /dev/video1\n      EMPTY: null\n      PORT: 7000\n"
            ),
            ".cache/app-compose-overrides.yaml": (
                "services:\n  main:\n    environment:\n      - VIDEO_DEVICE=/dev/video2\n  other:\n    environment:\n      X: y\n"
            ),
        },
    )
    assert appinfo.compose_env(app) == {"APP_HOME": "/h/one", "VIDEO_DEVICE": "/dev/video2", "EMPTY": "", "PORT": "7000"}


def test_no_compose_files_no_env(tmp_path: Path):
    assert appinfo.compose_env(make_app(tmp_path, "one")) == {}


def test_a_wheel_counts_the_same_once_run_sh_marks_it_installed(tmp_path: Path):
    app = make_app(tmp_path, "one", files={"python-libraries/lib-1.0-py3-none-any.whl": "wheel"})
    before = appinfo.deps_fingerprint(app)
    (app / "python-libraries" / "lib-1.0-py3-none-any.whl").rename(app / "python-libraries" / "lib-1.0-py3-none-any.whl.installed")
    assert appinfo.deps_fingerprint(app) == before


def test_the_dependencies_fingerprint_follows_the_requirements(tmp_path: Path):
    app = make_app(tmp_path, "one", files={"python/requirements.txt": "requests\n"})
    before = appinfo.deps_fingerprint(app)
    (app / "python" / "requirements.txt").write_text("requests\npsutil\n")
    assert appinfo.deps_fingerprint(app) != before
    changed = appinfo.deps_fingerprint(app)
    (app / "bricks" / "mine").mkdir(parents=True)
    (app / "bricks" / "mine" / "requirements.txt").write_text("numpy\n")
    assert appinfo.deps_fingerprint(app) != changed


def test_prepare_is_needed_without_a_venv_or_after_a_change(tmp_path: Path):
    app_path = make_app(tmp_path, "one", files={"python/requirements.txt": "requests\n"})
    app = appinfo.load_app(app_path)
    assert appinfo.needs_prepare(app), "no venv yet"
    app.venv_python.parent.mkdir(parents=True)
    app.venv_python.write_text("")
    assert appinfo.needs_prepare(app), "never prepared by the launcher"
    appinfo.record_prepared(app)
    assert not appinfo.needs_prepare(app)
    (app_path / "python" / "requirements.txt").write_text("requests\npsutil\n")
    assert appinfo.needs_prepare(app)


def test_an_app_that_adds_nothing_is_prepared_once_and_runs_without_a_venv(tmp_path: Path):
    app = appinfo.load_app(make_app(tmp_path, "one"))
    assert not appinfo.needs_venv(app)
    assert appinfo.needs_prepare(app), "never prepared by the launcher"
    appinfo.record_prepared(app)
    assert not appinfo.needs_prepare(app), "run.sh makes no venv for it: prepare would run at every start"


def test_a_missing_venv_needs_a_prepare_when_the_app_needs_one(tmp_path: Path):
    app_path = make_app(tmp_path, "one", files={"python/requirements.txt": "requests\n"})
    app = appinfo.load_app(app_path)
    appinfo.record_prepared(app)
    assert appinfo.needs_prepare(app), "the requirements go into a venv that does not exist"


@pytest.mark.parametrize(
    ("files", "streamlit", "expected"),
    [
        ({}, False, False),
        ({"python/requirements.txt": "\n  \n"}, False, False),
        ({"python/requirements.txt": "requests\n"}, False, True),
        ({"bricks/mine/requirements.txt": "numpy\n"}, False, True),
        ({"python-libraries/lib-1.0-py3-none-any.whl": ""}, False, True),
        ({".cache/.venv/pyvenv.cfg": ""}, False, True),
        ({}, True, True),
    ],
)
def test_needs_venv_follows_run_sh(tmp_path: Path, files: dict[str, str], streamlit: bool, expected: bool):
    app = appinfo.load_app(make_app(tmp_path, "one", files=files))
    assert appinfo.needs_venv(dataclasses.replace(app, streamlit=streamlit)) is expected


def test_the_app_fingerprint_ignores_main_py_and_follows_the_rest(tmp_path: Path):
    app_path = make_app(tmp_path, "one")
    app = appinfo.load_app(app_path)
    base = appinfo.app_fingerprint(app, {"A": "1"}, "/usr/bin/python3")
    (app_path / "python" / "main.py").write_text("print('changed')\n")
    (app_path / "python" / "helper.py").write_text("X = 1\n")
    assert appinfo.app_fingerprint(app, {"A": "1"}, "/usr/bin/python3") == base, "the app code is read when it runs"
    assert appinfo.app_fingerprint(app, {"A": "2"}, "/usr/bin/python3") != base
    assert appinfo.app_fingerprint(app, {"A": "1"}, "/other/python") != base
    (app_path / "app.yaml").write_text("name: t\nbricks: []\n")
    assert appinfo.app_fingerprint(app, {"A": "1"}, "/usr/bin/python3") != base


def test_the_interpreter_is_the_venv_one_when_there_is_a_venv(tmp_path: Path):
    app = appinfo.load_app(make_app(tmp_path, "one"))
    assert appinfo.app_interpreter(app, "/usr/local/bin/python3") == "/usr/local/bin/python3"
    app.venv_python.parent.mkdir(parents=True)
    app.venv_python.write_text("")
    assert appinfo.app_interpreter(app, "/usr/local/bin/python3") == str(app.venv_python)


def test_the_interpreter_writing_its_cache_in_the_venv_changes_nothing(tmp_path: Path):
    app = appinfo.load_app(make_app(tmp_path, "one"))
    site_packages = app.path / ".cache" / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_virtualenv.pth").write_text("import _virtualenv\n")
    base = appinfo.app_fingerprint(app, {}, "/py")
    (site_packages / "__pycache__").mkdir()
    assert appinfo.app_fingerprint(app, {}, "/py") == base
    (site_packages / "psutil-7.0.0.dist-info").mkdir()
    assert appinfo.app_fingerprint(app, {}, "/py") != base, "a package installed into the venv"
