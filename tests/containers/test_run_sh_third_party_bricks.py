# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""run.sh of python-apps-base puts on PYTHONPATH the sources of the third-party bricks the app declares.

The bricks are mounted one folder each, `<name>-<version>/src/<package>/brick_config.yaml`. A brick is
loaded when the `id` of its brick_config.yaml is listed under `bricks:` in the app.yaml of the app, and
then only its `src` is added, whatever else the folder holds. The `arduino:` ids of app.yaml are the
bricks of the library: they are never looked up among the third-party ones.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.containers.test_run_sh_venv import _app, _run  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None, reason="needs a POSIX sh")


def _brick(folder: str, package: str, brick_id: str) -> dict[str, str]:
    config = f"id: {brick_id}\nname: {package}\ndescription: |\n  id: not the id\ncategory: sensor\n"
    return {
        f"{folder}/src/{package}/__init__.py": "",
        f"{folder}/src/{package}/brick_config.yaml": config,
        f"{folder}/README.md": "docs\n",
    }


def _third_party(tmp_path: Path, *bricks: dict[str, str]) -> Path:
    root = tmp_path / "3rd-party-bricks"
    root.mkdir()
    for files in bricks:
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return root


def _pythonpath(tmp_path: Path) -> list[str]:
    return [entry for entry in (tmp_path / "calls.log.pythonpath").read_text().split(":") if entry]


SOIL = _brick("Soil_sensor-1.0.0", "soil_sensor", "acme:soil_sensor")
AIR = _brick("Air_sensor-1.5.0", "air_sensor", "acme:air_sensor")
ELEVATOR = _brick("Soil_elevator-3.0.0", "soil_elevator", "acme:soil_elevator")


def test_only_the_bricks_declared_in_app_yaml_are_on_pythonpath(tmp_path: Path):
    root = _third_party(tmp_path, SOIL, AIR, ELEVATOR)
    app_yaml = "name: t\nbricks:\n  - arduino:web_ui\n  - acme:soil_sensor\n  - acme:soil_elevator\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert sorted(_pythonpath(tmp_path)) == [f"{root}/Soil_elevator-3.0.0/src", f"{root}/Soil_sensor-1.0.0/src"]


def test_an_arduino_id_is_never_looked_up_among_the_third_party_bricks(tmp_path: Path):
    impostor = _brick("Web_ui-9.9.9", "web_ui", "arduino:web_ui")
    root = _third_party(tmp_path, impostor, SOIL)
    app_yaml = "name: t\nbricks:\n  - arduino:web_ui\n  - acme:soil_sensor\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == [f"{root}/Soil_sensor-1.0.0/src"]


def test_an_app_of_arduino_bricks_only_reads_no_third_party_brick(tmp_path: Path):
    root = _third_party(tmp_path, SOIL)
    # A folder in place of the file: reading it would make awk fail
    (root / "Unreadable-1.0.0" / "src" / "unreadable" / "brick_config.yaml").mkdir(parents=True)
    app_yaml = "name: t\nbricks:\n  - arduino:web_ui\n  - arduino:llm:\n      model: tiny\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == []
    assert "Unreadable" not in (tmp_path / "run.stderr").read_text()


def test_files_with_crlf_line_endings_and_a_byte_order_mark_are_read(tmp_path: Path):
    windows_brick = {"Win-1.0.0/src/win/brick_config.yaml": "﻿id: acme:win\r\nname: win\r\n"}
    root = _third_party(tmp_path, windows_brick, SOIL, AIR)
    app_yaml = "﻿name: t\r\nbricks:\r\n  - arduino:web_ui\r\n\r\n  - acme:soil_sensor\r\n  - acme:win\r\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert sorted(_pythonpath(tmp_path)) == [f"{root}/Soil_sensor-1.0.0/src", f"{root}/Win-1.0.0/src"]


def test_an_empty_app_yaml_declares_nothing(tmp_path: Path):
    # A brick_config.yaml shaped like an app.yaml: it must not be read in its place
    decoy = {"Decoy-1.0.0/src/decoy/brick_config.yaml": "bricks:\n  - acme:soil_sensor\n"}
    root = _third_party(tmp_path, decoy, SOIL)
    _run(_app(tmp_path, app_yaml=""), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == []


def test_the_image_ships_the_selection_script_where_run_sh_looks_for_it():
    image_dir = REPO_ROOT / "containers" / "bricks" / "python-apps-base"
    assert (image_dir / "scripts" / "third-party-bricks.awk").is_file()
    assert 'THIRD_PARTY_BRICKS_AWK="/third-party-bricks.awk"\n' in (image_dir / "scripts" / "run.sh").read_text()
    assert "COPY ./scripts/*.sh ./scripts/*.awk /\n" in (image_dir / "Dockerfile").read_text()


def test_a_flow_list_of_ids_declares_them(tmp_path: Path):
    root = _third_party(tmp_path, SOIL, AIR, ELEVATOR)
    app_yaml = 'name: t\nbricks: [arduino:web_ui, acme:soil_sensor, "acme:soil_elevator"]  # inline\n'
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert sorted(_pythonpath(tmp_path)) == [f"{root}/Soil_elevator-3.0.0/src", f"{root}/Soil_sensor-1.0.0/src"]


@pytest.mark.parametrize(
    "entry",
    [
        "- acme:soil_sensor",
        "- acme:soil_sensor: {}",
        "- acme:soil_sensor:\n    model: tiny\n    variables:\n      - acme:air_sensor",
        '  - "acme:soil_sensor"  # the sensor',
        "  - acme:soil_sensor:\r",
    ],
)
def test_every_way_of_listing_a_brick_declares_it(tmp_path: Path, entry: str):
    root = _third_party(tmp_path, SOIL, AIR)
    app_yaml = f"name: t\nbricks:\n{entry}\nicon: x\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == [f"{root}/Soil_sensor-1.0.0/src"]


def test_an_id_outside_the_bricks_list_declares_nothing(tmp_path: Path):
    root = _third_party(tmp_path, SOIL, AIR)
    app_yaml = 'name: t\ndescription: "uses acme:soil_sensor"\nbricks:\n  - arduino:web_ui\n  - acme:soil_sensor_v2\nports:\n  - acme:air_sensor\n'
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == []


def test_a_folder_without_a_brick_config_is_skipped(tmp_path: Path):
    no_config = {"Bare-0.1.0/src/bare/__init__.py": ""}
    no_id = {"Anonymous-0.1.0/src/anonymous/brick_config.yaml": "name: anonymous\n"}
    root = _third_party(tmp_path, SOIL, no_config, no_id)
    app_yaml = "name: t\nbricks:\n- acme:soil_sensor\n- bare\n- anonymous\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path, third_party_bricks=root)
    assert _pythonpath(tmp_path) == [f"{root}/Soil_sensor-1.0.0/src"]


def test_without_third_party_bricks_pythonpath_is_left_alone(tmp_path: Path):
    app_yaml = "name: t\nbricks:\n- acme:soil_sensor\n"
    _run(_app(tmp_path, app_yaml=app_yaml), tmp_path)
    assert _pythonpath(tmp_path) == []

    empty = _third_party(tmp_path)
    _run(_app(tmp_path / "second", app_yaml=app_yaml), tmp_path, third_party_bricks=empty)
    assert _pythonpath(tmp_path) == []


def test_third_party_bricks_follow_the_inherited_path_and_the_custom_bricks(tmp_path: Path):
    root = _third_party(tmp_path, ELEVATOR)
    app = _app(tmp_path, {"bricks/mine/__init__.py": ""}, app_yaml="name: t\nbricks:\n- acme:soil_elevator\n")
    _run(app, tmp_path, third_party_bricks=root, pythonpath="/opt")
    assert _pythonpath(tmp_path) == ["/opt", f"{app}/bricks", f"{root}/Soil_elevator-3.0.0/src"]
