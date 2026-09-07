# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Report what ONNX Runtime can see on this machine, and how much of each EasyOCR graph
the Hexagon NPU actually takes.

Run this first when bringing the pipeline up on a new Snapdragon target:

    python tools/check_ep.py

A partial offload is normal and supported: ONNX Runtime hands QNN the nodes it can run
and leaves the rest on the CPU EP. So the question is never "NPU or not" but "how much,
and is the leftover expensive". This tool answers, in order:

  1. is the QNN EP installed at all (plugin wheel or bundled build)?
  2. does ORT enumerate an NPU device?
  3. for each model: how many QNN partitions, how many nodes stayed on the CPU, and how
     the runtime splits between the two.

The last one is what matters. A graph cut into many partitions ping-pongs tensors between
NPU and CPU, and that round-tripping can cost more than the ops themselves.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort

from utils.constants import DETECTOR_MODEL_PATH, RECOGNIZER_MODEL_PATH
from utils.model_io_processing import ONNXModel
from utils.onnx_ep import QNN_EP_NAME, _plugin_module, _qnn_provider_options, _register_plugin

TIMED_RUNS = 5


def report_environment() -> None:
    print(f"onnxruntime      {ort.__version__}")
    print(f"available EPs    {ort.get_available_providers()}")

    plugin = _plugin_module()
    if plugin is None:
        print("plugin EP        onnxruntime-qnn NOT installed")
        if QNN_EP_NAME not in ort.get_available_providers():
            print("\n  -> no QNN anywhere. On a Snapdragon target:")
            print("       pip install -r requirements.txt")
    else:
        print(f"plugin EP        onnxruntime-qnn {getattr(plugin, '__version__', 'unknown')}")
        try:
            print(f"  library        {plugin.get_library_path()}")
            print(f"  HTP backend    {plugin.get_qnn_htp_path()}")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not resolve the plugin paths: {exc}")

    print(f"provider options {_qnn_provider_options(plugin)}")

    # Plugin EPs only appear in get_ep_devices() once their library is registered, and
    # registration otherwise happens lazily at the first session. Do it now, or this
    # listing claims there is no NPU on a board where the NPU works fine.
    if plugin is not None:
        _register_plugin(plugin)

    if hasattr(ort, "get_ep_devices"):
        devices = ort.get_ep_devices()
        print(f"EP devices       {len(devices)}")
        for device in devices:
            marker = " <-- QNN" if device.ep_name == QNN_EP_NAME else ""
            print(f"  {device.ep_name}{marker}")
        if plugin is not None and not any(d.ep_name == QNN_EP_NAME for d in devices):
            print("  (no QNN device listed; the EP may still attach via the legacy path)")
    print()


def probe(label: str, model_path: str) -> None:
    if not os.path.isfile(model_path):
        print(f"{label}: {model_path} missing - run python tools/download_models.py")
        return

    # backend="auto" with verification on: exactly what the pipeline does in production,
    # CPU fallback and partial offload included, but with the placement actually measured
    # instead of inferred from get_providers().
    try:
        model = ONNXModel(model_path, backend="auto", verify_placement=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: could not open the model at all -> {exc}")
        print()
        return

    height, width = model.image_shape
    frame = np.zeros((1, height, width, model.channels), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(TIMED_RUNS):
        model(frame)
    wall_ms = (time.perf_counter() - start) / TIMED_RUNS * 1000

    placement = model.placement or {}
    print(f"{label}: {wall_ms:.1f} ms/inference ({height}x{width})")

    total_us = sum(duration for _, duration in placement.values()) or 1.0
    for provider, (count, duration) in sorted(placement.items(), key=lambda item: -item[1][1]):
        unit = "partitions" if provider == QNN_EP_NAME else "nodes"
        print(f"    {provider:<24} {count:>4} {unit:<11} {duration / 1000:7.2f} ms  {duration / total_us * 100:5.1f}%")

    qnn_partitions = placement.get(QNN_EP_NAME, (0, 0.0))[0]
    cpu_nodes = sum(count for provider, (count, _) in placement.items() if provider != QNN_EP_NAME)
    if qnn_partitions == 0:
        print("    -> NOTHING on the NPU: the whole graph ran on the CPU.")
        print("       Check the QNN errors ORT printed above. Common causes, in order:")
        print("         * ADSP_LIBRARY_PATH pointing at another project's DSP libraries")
        print("         * QAIRT/skel version mismatch between the wheel and /usr/lib/rfsa/adsp")
        print("         * an htp_arch or soc_model the device does not actually have")
        print("       EASYOCR_QNN_OP_TRACE=1 dumps the op mapping once the backend comes up.")
    elif cpu_nodes == 0:
        print("    -> whole graph on the NPU.")
    else:
        print(
            f"    -> partial offload: {qnn_partitions} NPU partition(s) with {cpu_nodes} node(s) "
            "left on the CPU. Fine as long as the CPU share above is small; if the graph is "
            "cut into many partitions, the NPU<->CPU round-trips are the cost to watch."
        )
    print()


def main() -> int:
    report_environment()
    probe("detector  ", os.environ.get("EASYOCR_DETECTOR_MODEL", DETECTOR_MODEL_PATH))
    probe("recognizer", os.environ.get("EASYOCR_RECOGNIZER_MODEL", RECOGNIZER_MODEL_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
