# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The pre-compiled HTP context binaries committed with the ocr-runner must match the
repository they ship with.

A `<model>.qnn_ctx.onnx` is a graph compiled on the board for one exact combination of
model bytes, `onnxruntime` / `onnxruntime-qnn` (QAIRT) wheels and compile-time QNN
options; each carries that combination in `<model>.qnn_ctx.json`. These tests compare the
fingerprints with what the repository pins, so that bumping a wheel in requirements.txt,
replacing a model or touching the compile options without recompiling turns CI red - with
the fix spelled out - instead of shipping binaries the runner will reject at start-up.

Recompile with `python tools/compile_htp_context.py` on the target board.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[3] / "containers" / "ai" / "ocr-runner"
BASE_IMAGE_DOCKERFILE = Path(__file__).resolve().parents[3] / "containers" / "base" / "qairt-common-base" / "Dockerfile"
MODEL_DIR = RUNNER_DIR / "models" / "easyocr-onnx-w8a8"
MODELS = ("detector", "recognizer")

# SoCs the image ships binaries for, soc_id -> machine (/sys/devices/soc0/{soc_id,machine}).
# HTP context binaries are SoC-specific: they are named <model>.soc<soc_id>.qnn_ctx.onnx and
# the runner loads the one matching the board it runs on. Add an entry (and the compiled
# files) to support another SoC.
SUPPORTED_SOCS = {"675": "QCS8275"}

RECOMPILE_HINT = "-> recompile the HTP context binaries on the board: python tools/compile_htp_context.py, and commit the .qnn_ctx.onnx/.json pair"


def _pinned(package: str) -> str:
    """The `==` pin of `package` in the hash-locked requirements.txt."""
    pattern = re.compile(rf"^{re.escape(package)}==(\S+?)\s*(\\)?$", re.MULTILINE)
    match = pattern.search((RUNNER_DIR / "requirements.txt").read_text(encoding="utf-8"))
    assert match, f"{package} is not pinned with == in requirements.txt"
    return match.group(1)


def _declared_qairt_version() -> str:
    match = re.search(r"^#\s*qairt-version:\s*(\S+)", (RUNNER_DIR / "requirements.in").read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "requirements.in must declare `# qairt-version: <version>` next to the onnxruntime-qnn pin"
    return match.group(1)


def _expected_compile_options() -> dict[str, str]:
    """The compile-time subset of DEFAULT_QNN_OPTIONS, read from utils/onnx_ep.py itself."""
    stub = types.ModuleType("onnxruntime")
    stub.__version__ = "0"
    stub.set_default_logger_severity = lambda level: None
    saved = sys.modules.get("onnxruntime")
    sys.modules["onnxruntime"] = stub
    try:
        spec = importlib.util.spec_from_file_location("ocr_runner_onnx_ep_for_ctx", RUNNER_DIR / "utils" / "onnx_ep.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["onnxruntime"]
        else:
            sys.modules["onnxruntime"] = saved
    return {key: value for key, value in module.DEFAULT_QNN_OPTIONS.items() if key in module.CONTEXT_COMPILE_OPTIONS}


def _binary_stem(model: str, soc_id: str) -> str:
    return f"{model}.soc{soc_id}.qnn_ctx"


def _fingerprint(model: str, soc_id: str) -> dict:
    path = MODEL_DIR / f"{_binary_stem(model, soc_id)}.json"
    assert path.is_file(), f"{path.name} missing: every shipped context binary needs its fingerprint {RECOMPILE_HINT}"
    return json.loads(path.read_text(encoding="utf-8"))


CASES = [pytest.param(model, soc_id, id=f"{model}-soc{soc_id}") for model in MODELS for soc_id in SUPPORTED_SOCS]


@pytest.mark.parametrize(("model", "soc_id"), CASES)
def test_context_binary_is_shipped_for_each_model_and_soc(model: str, soc_id: str):
    binary = MODEL_DIR / f"{_binary_stem(model, soc_id)}.onnx"
    assert binary.is_file(), f"{binary.name} missing: the runner would compile at start-up (minutes) {RECOMPILE_HINT}"
    assert binary.stat().st_size > 1_000_000, f"{binary.name} is suspiciously small ({binary.stat().st_size} bytes)"
    assert (MODEL_DIR / f"{model}.onnx").is_file() and (MODEL_DIR / f"{model}.data").is_file()


def test_every_shipped_fingerprint_belongs_to_a_supported_soc_and_matches_its_name():
    fingerprints = sorted(MODEL_DIR.glob("*.qnn_ctx.json"))
    assert fingerprints, "no *.qnn_ctx.json shipped at all"
    for path in fingerprints:
        match = re.fullmatch(r"(detector|recognizer)\.soc(\d+)\.qnn_ctx\.json", path.name)
        assert match, f"{path.name}: shipped binaries must be named <model>.soc<soc_id>.qnn_ctx.*"
        soc_id = match.group(2)
        assert soc_id in SUPPORTED_SOCS, f"{path.name}: soc_id {soc_id} is not in SUPPORTED_SOCS; add it if the SoC is meant to be supported"
        fingerprint = json.loads(path.read_text(encoding="utf-8"))
        assert fingerprint.get("soc_id") == soc_id, f"{path.name} is named for soc_id {soc_id} but was compiled on soc_id {fingerprint.get('soc_id')}"
        assert fingerprint.get("soc_machine") == SUPPORTED_SOCS[soc_id], (
            f"{path.name}: compiled on {fingerprint.get('soc_machine')}, SUPPORTED_SOCS says soc_id {soc_id} is {SUPPORTED_SOCS[soc_id]}"
        )
        assert path.with_suffix(".onnx").is_file(), f"{path.name} has no binary next to it"


@pytest.mark.parametrize(("model", "soc_id"), CASES)
def test_fingerprint_matches_the_pinned_runtime(model: str, soc_id: str):
    fingerprint = _fingerprint(model, soc_id)
    ort, ort_qnn, qairt = _pinned("onnxruntime"), _pinned("onnxruntime-qnn"), _declared_qairt_version()

    assert fingerprint["onnxruntime"] == ort, (
        f"{model}: compiled with onnxruntime {fingerprint['onnxruntime']}, requirements.txt pins {ort} {RECOMPILE_HINT}"
    )
    assert fingerprint["onnxruntime_qnn"] == ort_qnn, (
        f"{model}: compiled with onnxruntime-qnn {fingerprint['onnxruntime_qnn']}, requirements.txt pins {ort_qnn} {RECOMPILE_HINT}"
    )
    assert fingerprint["qnn_version"] == qairt, (
        f"{model}: compiled with QAIRT {fingerprint['qnn_version']}, requirements.in declares qairt-version {qairt} {RECOMPILE_HINT}"
    )
    assert fingerprint["backend"] == "libQnnHtp.so", (
        f"{model}: compiled for backend {fingerprint['backend']!r}, the runner uses the HTP (libQnnHtp.so)"
    )
    assert fingerprint.get("backend_qairt") == qairt, (
        f"{model}: the backend library used at compile time was QAIRT {fingerprint.get('backend_qairt')}, requirements.in declares "
        f"qairt-version {qairt} {RECOMPILE_HINT}"
    )


@pytest.mark.parametrize(("model", "soc_id"), CASES)
def test_fingerprint_matches_the_compile_options(model: str, soc_id: str):
    recorded = json.loads(_fingerprint(model, soc_id)["options"])
    expected = _expected_compile_options()
    assert recorded == expected, f"{model}: compiled with {recorded}, utils/onnx_ep.py now compiles with {expected} {RECOMPILE_HINT}"


@pytest.mark.parametrize(("model", "soc_id"), CASES)
def test_fingerprint_matches_the_model_file(model: str, soc_id: str):
    fingerprint = _fingerprint(model, soc_id)
    size = (MODEL_DIR / f"{model}.onnx").stat().st_size
    assert fingerprint["model_bytes"] == str(size), (
        f"{model}: compiled from a {fingerprint['model_bytes']}-byte graph, {model}.onnx is now {size} bytes {RECOMPILE_HINT}"
    )


def test_declared_qairt_version_matches_the_documented_wheel():
    """requirements.txt documents the QAIRT bundled in the pinned wheel; keep the two statements in sync."""
    qairt = _declared_qairt_version()
    lock = (RUNNER_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert f"QAIRT {qairt}" in lock, f"requirements.txt header does not mention QAIRT {qairt}; regenerate it from requirements.in"


def test_plugin_qairt_matches_the_base_image_qairt():
    """When the runner is configured to use the base image's QAIRT (Dockerfile sets
    EASYOCR_QNN_BACKEND_PATH=/usr/lib/libQnnHtp.so and drops the wheel's copy), the wheel must be
    the release built with that exact QAIRT. QNP_VER in qairt-common-base is `major.minor.patch.build`;
    the wheel reports `major.minor.patch`. While the wheel's own QAIRT is in use the check does not apply."""
    dockerfile = (RUNNER_DIR / "Dockerfile").read_text(encoding="utf-8")
    if not re.search(r"^ENV EASYOCR_QNN_BACKEND_PATH=/usr/lib/libQnnHtp\.so", dockerfile, re.MULTILINE):
        pytest.skip("the runner uses the QAIRT bundled in the onnxruntime-qnn wheel, not the base image's")
    match = re.search(r"^ENV QNP_VER=(\S+)", BASE_IMAGE_DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, "QNP_VER not found in the qairt-common-base Dockerfile"
    base_qairt = ".".join(match.group(1).split(".")[:3])
    declared = _declared_qairt_version()
    assert declared == base_qairt, (
        f"requirements.in declares qairt-version {declared} but the base image ships QAIRT {match.group(1)}: pick the onnxruntime-qnn "
        f"release built with QAIRT {base_qairt} (github.com/onnxruntime/onnxruntime-qnn/releases), update the pins {RECOMPILE_HINT}"
    )
