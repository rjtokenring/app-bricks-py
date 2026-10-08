# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from arduino.app_tools.launcher import imports

MAIN = """
import os, numpy as np
import xml.etree.ElementTree
from arduino.app_bricks.web_ui import WebUI
from . import sibling
from .pkg import thing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import only_for_types
else:
    import at_runtime

try:
    import optional_lib
except ImportError:
    import fallback_lib

class Model:
    import in_class_body

def later():
    import lazy_in_function

with open(__file__):
    import in_with
"""


def test_module_level_imports_are_collected_in_order(tmp_path: Path):
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "main.py").write_text(MAIN)
    found = imports.scan_imports([tmp_path / "python"])
    assert found == [
        "os",
        "numpy",
        "xml.etree.ElementTree",
        "arduino.app_bricks.web_ui",
        "typing",
        "at_runtime",
        "optional_lib",
        "fallback_lib",
        "in_class_body",
        "in_with",
    ]


def test_files_that_do_not_parse_and_caches_are_skipped(tmp_path: Path):
    root = tmp_path / "python"
    (root / "__pycache__").mkdir(parents=True)
    (root / "__pycache__" / "cached.py").write_text("import from_cache\n")
    (root / "broken.py").write_text("import ok_before\ndef (:\n")
    (root / "fine.py").write_text("import requests\n")
    assert imports.scan_imports([root, tmp_path / "missing"]) == ["requests"]


def test_local_names_are_the_modules_and_packages_of_the_app(tmp_path: Path):
    python = tmp_path / "python"
    bricks = tmp_path / "bricks"
    (python / "pkg").mkdir(parents=True)
    (python / "__pycache__").mkdir()
    (python / "main.py").write_text("")
    (python / "prompts.py").write_text("")
    (python / "prompt.yaml").write_text("")
    (bricks / "mcp_server").mkdir(parents=True)
    (bricks / "not-a-name").mkdir()
    assert imports.local_module_names(python, bricks, tmp_path / "missing") == {"pkg", "main", "prompts", "mcp_server"}


def test_candidates_put_the_preload_first_and_never_name_an_app_module():
    candidates = imports.warm_candidates(
        preload=["numpy", "cv2", " "],
        brick_modules=["arduino.app_bricks.web_ui", "numpy"],
        scanned=["prompts", "prompts.sub", "requests", "cv2"],
        local_names={"prompts"},
    )
    assert candidates == ["numpy", "cv2", "arduino.app_bricks.web_ui", "requests"]


def test_a_local_module_hides_a_preloaded_library_of_the_same_name():
    assert imports.warm_candidates(["yaml", "numpy"], [], [], {"yaml"}) == ["numpy"]


WEB_SCAN = ["arduino.app_bricks.web_ui", "fastapi.responses", "uvicorn", "starlette.staticfiles", "fastapi_socketio", "fastapiextra", "requests"]


def test_the_web_stack_is_not_warmed_without_the_web_ui_brick():
    excluded = imports.excluded_modules(["arduino:video_object_detection", "custom:web_ui"])
    candidates = imports.warm_candidates(["numpy", "fastapi"], [], WEB_SCAN, set(), excluded)
    assert candidates == ["numpy", "fastapiextra", "requests"]


def test_the_web_stack_is_warmed_with_the_web_ui_brick():
    excluded = imports.excluded_modules(["arduino:web_ui"])
    assert excluded == ()
    assert imports.warm_candidates([], ["arduino.app_bricks.web_ui"], WEB_SCAN, set(), excluded) == WEB_SCAN
