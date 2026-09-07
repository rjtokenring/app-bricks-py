# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""ONNX Runtime / QNN plumbing of the ocr-runner: context binary lookup, fingerprinting,
DSP library path selection and the quantized I/O wrapper.

onnxruntime is not a test dependency, so the modules are loaded standalone by file path
with a stub `onnxruntime` (and, where needed, `onnxruntime_qnn`) module in sys.modules.
Only the pure-Python paths are exercised: everything that opens a real session is out of
scope here and is what tools/check_ep.py measures on the board.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

RUNNER_DIR = Path(__file__).resolve().parents[3] / "containers" / "ai" / "ocr-runner"
MODEL_DIR = RUNNER_DIR / "models" / "easyocr-onnx-w8a8"


def _stub_onnxruntime(version: str = "1.29.0") -> types.ModuleType:
    ort = types.ModuleType("onnxruntime")
    ort.__version__ = version

    class SessionOptions:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.entries[key] = value

    ort.SessionOptions = SessionOptions
    ort.RunOptions = type("RunOptions", (), {"add_run_config_entry": lambda self, k, v: None})
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    ort.set_default_logger_severity = lambda level: None
    return ort


def _stub_plugin(version: str = "2.5.0", qnn_version: str = "2.49.40", lib_dir: Path | None = None) -> types.ModuleType:
    plugin = types.ModuleType("onnxruntime_qnn")
    plugin.__version__ = version
    plugin.build_and_package_info = types.SimpleNamespace(qnn_version=qnn_version)
    if lib_dir is not None:
        plugin.get_qnn_htp_path = lambda: str(lib_dir / "libQnnHtp.so")
        plugin.get_library_path = lambda: str(lib_dir / "libonnxruntime_providers_qnn.so")
    return plugin


def _load(name: str, monkeypatch, plugin: types.ModuleType | None = None):
    """Load a runner module by path with stubbed runtime dependencies."""
    monkeypatch.setitem(sys.modules, "onnxruntime", _stub_onnxruntime())
    if plugin is not None:
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", plugin)
    else:
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)  # makes `import onnxruntime_qnn` fail
    spec = importlib.util.spec_from_file_location(f"ocr_runner_{name}", RUNNER_DIR / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def onnx_ep(monkeypatch):
    # Each test gets a fresh module: it keeps process-wide state (_adsp_checked).
    monkeypatch.delenv("EASYOCR_QNN_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("EASYOCR_QNN_CONTEXT_STRICT", raising=False)
    monkeypatch.delenv("EASYOCR_QNN_ADSP_PATH", raising=False)
    monkeypatch.delenv("EASYOCR_QNN_KEEP_ADSP_PATH", raising=False)
    monkeypatch.delenv("ADSP_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("CDSP_LIBRARY_PATH", raising=False)
    return _load("onnx_ep", monkeypatch, plugin=_stub_plugin())


def _make_model(directory: Path, name: str = "recognizer") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{name}.onnx"
    model.write_bytes(b"onnx" * 8)
    return model


# --- context binary lookup ---------------------------------------------------------------


def test_shipped_context_binary_is_found_beside_read_only_model(onnx_ep, tmp_path, monkeypatch):
    """A pre-compiled binary next to the model must be picked up even when that directory
    cannot be written (the normal case inside an image), and fresh compiles go elsewhere."""
    model_dir = tmp_path / "models"
    model = _make_model(model_dir)
    shipped = model_dir / "recognizer.qnn_ctx.onnx"
    shipped.write_bytes(b"ctx")

    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setattr(onnx_ep, "_writable", lambda d: not Path(d).samefile(model_dir) if Path(d).exists() else True)

    assert onnx_ep._find_context_binary(str(model)) == str(shipped)
    write_path = Path(onnx_ep._context_write_path(str(model)))
    assert write_path.parent == cache_home / "easyocr-onnx" / "qnn-context"
    assert write_path.name == "recognizer.qnn_ctx.onnx"


def test_recompiled_binary_shadows_stale_shipped_one(onnx_ep, tmp_path, monkeypatch):
    """When a shipped binary was rejected and could not be deleted, the one recompiled into
    the writable cache must win on the next start - otherwise every start recompiles."""
    model_dir = tmp_path / "models"
    model = _make_model(model_dir)
    (model_dir / "recognizer.qnn_ctx.onnx").write_bytes(b"stale")

    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setattr(onnx_ep, "_writable", lambda d: not Path(d).samefile(model_dir) if Path(d).exists() else True)

    recompiled = Path(onnx_ep._context_write_path(str(model)))
    recompiled.parent.mkdir(parents=True, exist_ok=True)
    recompiled.write_bytes(b"fresh")

    assert onnx_ep._find_context_binary(str(model)) == str(recompiled)


def test_context_binary_next_to_writable_model_is_both_read_and_written(onnx_ep, tmp_path):
    model = _make_model(tmp_path / "models")
    expected = tmp_path / "models" / "recognizer.qnn_ctx.onnx"

    assert onnx_ep._find_context_binary(str(model)) is None
    assert onnx_ep._context_write_path(str(model)) == str(expected)
    expected.write_bytes(b"ctx")
    assert onnx_ep._find_context_binary(str(model)) == str(expected)


def test_explicit_context_dir_wins(onnx_ep, tmp_path, monkeypatch):
    model = _make_model(tmp_path / "models")
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("EASYOCR_QNN_CONTEXT_DIR", str(explicit))

    assert onnx_ep._context_write_path(str(model)) == str(explicit / "recognizer.qnn_ctx.onnx")


# --- fingerprint -------------------------------------------------------------------------


def _options(**overrides: str) -> dict[str, str]:
    options = {
        "backend_path": "/opt/venv/lib/python3.13/site-packages/onnxruntime_qnn/libQnnHtp.so",
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "0",
        "enable_htp_fp16_precision": "1",
        "offload_graph_io_quantization": "1",
        "profiling_level": "off",
    }
    options.update(overrides)
    return options


def test_fingerprint_records_the_qairt_version(onnx_ep, tmp_path):
    model = _make_model(tmp_path)
    fingerprint = onnx_ep._context_fingerprint(str(model), _options())

    assert fingerprint["onnxruntime"] == "1.29.0"
    assert fingerprint["onnxruntime_qnn"] == "2.5.0"
    assert fingerprint["qnn_version"] == "2.49.40"
    assert fingerprint["backend"] == "libQnnHtp.so"
    assert fingerprint["model_bytes"] == str(model.stat().st_size)
    # Runtime-only knobs are not part of the compiled binary's identity.
    assert "htp_performance_mode" not in json.loads(fingerprint["options"])
    assert json.loads(fingerprint["options"])["htp_graph_finalization_optimization_mode"] == "0"


def test_matching_fingerprint_passes(onnx_ep, tmp_path, capsys):
    model = _make_model(tmp_path)
    cache = tmp_path / "recognizer.qnn_ctx.onnx"
    cache.write_bytes(b"ctx")
    onnx_ep._write_fingerprint(str(cache), str(model), _options())

    assert (tmp_path / "recognizer.qnn_ctx.json").is_file()
    assert onnx_ep._check_fingerprint(str(cache), str(model), _options()) is True
    assert "WARNING" not in capsys.readouterr().out


def test_changed_qairt_version_is_reported_and_strict_mode_recompiles(onnx_ep, tmp_path, monkeypatch, capsys):
    model = _make_model(tmp_path)
    cache = tmp_path / "recognizer.qnn_ctx.onnx"
    cache.write_bytes(b"ctx")
    onnx_ep._write_fingerprint(str(cache), str(model), _options())

    # A newer onnxruntime-qnn wheel brings a different QAIRT: the binary no longer matches.
    sys.modules["onnxruntime_qnn"].build_and_package_info.qnn_version = "2.51.0"

    assert onnx_ep._check_fingerprint(str(cache), str(model), _options()) is True  # loaded, but loudly
    out = capsys.readouterr().out
    assert "WARNING" in out and "qnn_version: '2.49.40' -> '2.51.0'" in out

    monkeypatch.setenv("EASYOCR_QNN_CONTEXT_STRICT", "1")
    assert onnx_ep._check_fingerprint(str(cache), str(model), _options()) is False


def test_changed_compile_option_is_a_mismatch_but_runtime_option_is_not(onnx_ep, tmp_path, capsys):
    model = _make_model(tmp_path)
    cache = tmp_path / "recognizer.qnn_ctx.onnx"
    cache.write_bytes(b"ctx")
    onnx_ep._write_fingerprint(str(cache), str(model), _options())

    onnx_ep._check_fingerprint(str(cache), str(model), _options(htp_performance_mode="balanced"))
    assert "WARNING" not in capsys.readouterr().out

    onnx_ep._check_fingerprint(str(cache), str(model), _options(htp_graph_finalization_optimization_mode="3"))
    assert "htp_graph_finalization_optimization_mode" in capsys.readouterr().out


def test_missing_fingerprint_is_tolerated(onnx_ep, tmp_path, capsys):
    model = _make_model(tmp_path)
    cache = tmp_path / "recognizer.qnn_ctx.onnx"
    cache.write_bytes(b"ctx")

    assert onnx_ep._check_fingerprint(str(cache), str(model), _options()) is True
    assert "no fingerprint" in capsys.readouterr().out


# --- ADSP_LIBRARY_PATH -------------------------------------------------------------------


def _fake_wheel_dir(root: Path) -> Path:
    wheel = root / "site-packages" / "onnxruntime_qnn"
    wheel.mkdir(parents=True)
    (wheel / "libQnnHtp.so").write_bytes(b"")
    (wheel / "libQnnHtpV73Skel.so").write_bytes(b"")
    return wheel


def test_adsp_path_follows_the_skels_shipped_with_the_backend(onnx_ep, tmp_path, monkeypatch, capsys):
    """The base image exports the skels of its own QAIRT; the wheel's libQnnHtp.so needs the
    skels of the wheel's QAIRT, which sit next to it."""
    wheel = _fake_wheel_dir(tmp_path)
    monkeypatch.setenv("ADSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp")
    monkeypatch.setenv("CDSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp")

    onnx_ep._prepare_adsp_path(str(wheel / "libQnnHtp.so"))

    assert os.environ["ADSP_LIBRARY_PATH"] == str(wheel)
    assert os.environ["CDSP_LIBRARY_PATH"] == str(wheel)
    assert "was /usr/lib/rfsa/adsp" in capsys.readouterr().out


def test_adsp_path_override_and_keep_are_honoured(onnx_ep, tmp_path, monkeypatch):
    wheel = _fake_wheel_dir(tmp_path)

    monkeypatch.setenv("ADSP_LIBRARY_PATH", "/usr/lib/rfsa/adsp")
    monkeypatch.setenv("EASYOCR_QNN_KEEP_ADSP_PATH", "1")
    onnx_ep._prepare_adsp_path(str(wheel / "libQnnHtp.so"))
    assert os.environ["ADSP_LIBRARY_PATH"] == "/usr/lib/rfsa/adsp"

    onnx_ep._adsp_checked = False
    monkeypatch.delenv("EASYOCR_QNN_KEEP_ADSP_PATH")
    monkeypatch.setenv("EASYOCR_QNN_ADSP_PATH", "/dsp/custom")
    onnx_ep._prepare_adsp_path(str(wheel / "libQnnHtp.so"))
    assert os.environ["ADSP_LIBRARY_PATH"] == "/dsp/custom"


def test_inherited_adsp_path_without_skels_is_dropped_for_bare_backend(onnx_ep, tmp_path, monkeypatch, capsys):
    empty = tmp_path / "other-project"
    empty.mkdir()
    monkeypatch.setenv("ADSP_LIBRARY_PATH", str(empty))

    onnx_ep._prepare_adsp_path("libQnnHtp.so")  # bare name: no directory to take skels from

    assert "ADSP_LIBRARY_PATH" not in os.environ
    assert "unset ADSP_LIBRARY_PATH" in capsys.readouterr().out


# --- profile summary ---------------------------------------------------------------------


def test_summarize_profile_groups_kernel_time_by_provider(onnx_ep, tmp_path):
    events = [
        {"cat": "Node", "name": "QNN_0_kernel_time", "dur": 3000, "args": {"provider": "QNNExecutionProvider"}},
        {"cat": "Node", "name": "Dq_1_kernel_time", "dur": 200, "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Node", "name": "Dq_2_kernel_time", "dur": 400, "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Node", "name": "Dq_1_fence_before", "dur": 999, "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Session", "name": "model_run", "dur": 5000},
    ]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(events * 2), encoding="utf-8")

    breakdown = onnx_ep.summarize_profile(str(profile), runs=2)

    assert breakdown == {"QNNExecutionProvider": (1, 3000.0), "CPUExecutionProvider": (2, 600.0)}


# --- quantized I/O wrapper ---------------------------------------------------------------


@pytest.fixture
def model_io(monkeypatch):
    onnx_ep = _load("onnx_ep", monkeypatch, plugin=_stub_plugin())
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.onnx_ep", onnx_ep)
    return _load("model_io_processing", monkeypatch, plugin=_stub_plugin())


def test_quantize_dequantize_roundtrip_matches_metadata_params(model_io):
    scale, zero_point = 0.003921568859368563, 0  # detector input, from metadata.json
    values = np.linspace(0.0, 1.0, 11, dtype=np.float32)

    quantized = model_io.quantize(values, zero_point, scale, np.dtype(np.uint8))
    assert quantized.dtype == np.uint8
    assert quantized[0] == 0 and quantized[-1] == 255

    restored = model_io.dequantize(quantized, zero_point, scale)
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, values, atol=scale / 2)


def test_quantize_clips_to_the_dtype_range(model_io):
    values = np.array([-1.0, 2.0], dtype=np.float32)
    assert model_io.quantize(values, 0, 1 / 255, np.dtype(np.uint8)).tolist() == [0, 255]


def test_quantization_metadata_is_read_from_the_tracked_json(model_io):
    """The container ships metadata.json in git: this is what the uint8 w8a8 graphs need
    to turn real images into input tensors and logits back into probabilities."""
    detector = model_io._load_quantization_metadata(str(MODEL_DIR / "detector.onnx"))
    recognizer = model_io._load_quantization_metadata(str(MODEL_DIR / "recognizer.onnx"))

    assert detector["inputs"]["image"] == pytest.approx((0.003921568859368563, 0.0))
    assert detector["outputs"]["results"] == pytest.approx((0.004232470877468586, 12.0))
    assert recognizer["inputs"]["image"] == pytest.approx((0.003920610062777996, 0.0))
    assert recognizer["outputs"]["output_preds"] == pytest.approx((0.2553488612174988, 127.0))


def test_quantization_metadata_is_empty_without_the_json(model_io, tmp_path):
    assert model_io._load_quantization_metadata(str(tmp_path / "float.onnx")) == {"inputs": {}, "outputs": {}}


def test_layout_detection_and_shape_resolution(model_io):
    assert model_io._detect_layout((1, 3, 608, 800)) == "NCHW"
    assert model_io._detect_layout((1, 64, 800, 1)) == "NHWC"
    with pytest.raises(ValueError):
        model_io._detect_layout((1, 3, 3, 3))

    assert model_io.ONNXModel._resolve_shape(["batch", 1, 64, 800]) == (1, 1, 64, 800)
    with pytest.raises(ValueError, match="static shapes"):
        model_io.ONNXModel._resolve_shape([1, 1, 64, "width"])
