# ocr-runner: EasyOCR on ONNX Runtime + Qualcomm QNN

Model runner behind the `arduino:ocr` brick. CRAFT text detector + CRNN recognizer from
[Qualcomm AI Hub's EasyOCR](https://aihub.qualcomm.com/models/easyocr) (ai-hub-models
v0.61.0, w8a8), executed with ONNX Runtime on the Hexagon NPU through the QNN execution
provider. Pre/post-processing is numpy/OpenCV only: no torch, no `easyocr` package.

This used to be a TFLite/LiteRT runner. It was ported because the TFLite recognizer could
not be delegated to the NPU through `libQnnTFLiteDelegate.so` (dynamic-shaped tensor in
the exported graph), so it ran on the CPU at ~250 ms per text box. With ONNX Runtime both
graphs land on the NPU: detector ~20 ms, recognizer ~15 ms per box (QCS8275 / IQ8).

## Layout

| path | role |
| --- | --- |
| `inference.py` | the pipeline; `inference_callback` and `apply_config` are what `aihub-models-runner` calls |
| `utils/onnx_ep.py` | ORT session factory: QNN plugin EP, CPU fallback, HTP context binaries, fingerprints |
| `utils/model_io_processing.py` | `ONNXModel`: NHWC float in/out over the NCHW uint8 graphs, using `metadata.json` |
| `utils/constants.py` | model paths, thresholds, character set |
| `utils/orientation.py` | rotated text: read each cutout at 90/180/270 too and keep the most confident reading (`rotation` setting) |
| `utils/{bbox,image,post}_processing.py`, `utils/metadata.py` | runtime-agnostic EasyOCR ports (unchanged from the TFLite version) |
| `models/easyocr-onnx-w8a8/` | `.onnx` + `.data` graphs, `metadata.json`, and the compiled `*.soc<id>.qnn_ctx.onnx` / `.json`, one pair per SoC |
| `tools/compile_htp_context.py` | run on the board: compiles both graphs for the HTP and writes the context binaries |
| `requirements.in` / `requirements.txt` | the complete package list / its hash lock for linux aarch64 + CPython 3.13, installed with `--no-deps` |

## Client configuration

The brick sends a `{"config": {...}}` message before each frame; `apply_config` in
`inference.py` applies it and unknown keys are ignored:

| key | value | effect |
| --- | --- | --- |
| `allowlist` | string of characters, `""` to clear | only these characters can be decoded (CTC logits of the others are zeroed) |
| `rotation` | list of angles among 90, 180, 270, `[]` to clear | cutouts are also recognized rotated and the most confident reading wins (EasyOCR's `rotation_info`, see `utils/orientation.py`). 90/270 only on cutouts taller than wide, 180 on all, and a rotated reading must beat the upright one by 0.1 of confidence: without both guards rotated horizontal lines read as confident garbage. One extra recognizer pass per applicable angle per box; detection runs once |

Both settings are process-wide and persist until the next config message, which is why
the brick restates them on every call.

## Everything is pinned

Three things decide whether a compiled HTP graph is reusable: the model bytes, the QAIRT
release, and the compile options. Each is fixed in git:

* **Models**: tracked in the repository (`models/easyocr-onnx-w8a8/`), the
  `easyocr-onnx-w8a8.zip` of ai-hub release v0.61.0 unpacked as is.
* **Runtime**: `requirements.in` lists every package the image installs (no transitive
  resolution: sympy, mpmath, coloredlogs and humanfriendly, ~80 MB the wheels declare but
  never import, stay out); `requirements.txt` is its `uv pip compile --no-deps
  --generate-hashes` lock for the container target, installed with `pip install --no-deps
  --require-hashes`. A rebuild can neither pick a newer `onnxruntime` nor a different
  `onnxruntime-qnn` wheel nor an extra package. The QNN plugin wheel is self-contained:
  `onnxruntime-qnn 2.5.0` ships **QAIRT 2.49.40** (`libQnnHtp.so`, `libQnnHtpPrepare.so`,
  `libQnnHtpV68..V81Skel.so`, ~190 MB). That is *not* the QAIRT of the base image (2.45.41,
  used by the LiteRT delegate in the other runners), so `utils/onnx_ep.py` re-points
  `ADSP_LIBRARY_PATH` at the wheel's own skels - host library and DSP skel must come from
  the same release or the backend fails with `QNN_DEVICE_ERROR_INVALID_CONFIG`. This is safe with the base image's own fastrpc build
  (quic/fastrpc 1.0.6): it reads `ADSP_LIBRARY_PATH` with `getenv()` at every file open,
  for the CDSP domain too, and always appends the yaml-derived `/usr/share/qcom/...` DSP
  payload path after it; `CDSP_LIBRARY_PATH` is not referenced anywhere in its sources. The
  two QAIRTs do not mix: the wheel's `libQnnHtp.so` loads `libQnnHtpPrepare.so`,
  `libQnnSystem.so` and the stub by absolute path from its own directory (`dladdr`) and
  checks their build ids, so the 2.45 copies in `/usr/lib` are never picked up. The only
  system library it takes is `libcdsprpc.so`, which is the point. To see it on a board:
  `LD_DEBUG=libs python -c "import inference" 2>&1 | grep -E 'QnnHtp|QnnSystem|cdsprpc'`.
  Using a system QAIRT instead of the wheel's (`EASYOCR_QNN_BACKEND_PATH=/usr/lib/libQnnHtp.so`)
  only works if it is **at least as new** as the one the EP was built against: the EP picks
  the backend's interface only when the QNN API major matches and minor/patch are >= its
  own. Verified on the 21q: the host's QAIRT 2.46 (`qairt-libs`) is refused with
  `QNN SetupBackend failed Unable to find a valid interface for /usr/lib/libQnnHtp.so`, and
  so would be the base image's 2.45. A newer system QAIRT would load, but the context
  binaries are tied to the QAIRT that compiled them and would need recompiling.
* **Why the wheel's QAIRT and not the base image's** (measured on the 21q, EasyOCR
  recognizer, one release of `onnxruntime-qnn` at a time, each with its bundled QAIRT):

  | onnxruntime-qnn | onnxruntime | QAIRT | recognizer on the HTP |
  | --- | --- | --- | --- |
  | 2.1.1 | 1.24.4 | 2.45.41 (= base image) | garbage, every confidence 0.00 |
  | 2.2.0 | 1.24.4 | 2.46.0 | garbage |
  | 2.3.0 | 1.29.0 | 2.47.0 | digits misread, detector no longer on the NPU |
  | 2.4.0 | 1.29.0 | 2.48.40 | correct |
  | 2.5.0 | 1.29.0 | 2.49.40 | correct (pinned) |

  With 2.1.1 the same recognizer reads correctly on the CPU EP, so it is the HTP graph those
  QAIRT releases produce that is wrong, not the model. Using the base image's QAIRT 2.45.41
  is therefore not an option even though 2.1.1 loads it fine. To use a single, system QAIRT
  and drop the wheel's copy (~190 MB), `QNP_VER` in `qairt-common-base` has to move to the
  QAIRT of the pinned wheel; the Dockerfile has the two lines to enable then (`ENV
  EASYOCR_QNN_BACKEND_PATH=/usr/lib/libQnnHtp.so` plus deleting the wheel's `libQnn*.so`),
  `utils/onnx_ep.py` already resolves the backend either way and warns when the backend's
  QAIRT (`AISW_VERSION` string in the library) is not the plugin's, and
  `test_plugin_qairt_matches_the_base_image_qairt` starts enforcing the lockstep as soon as
  that `ENV` line is present.
* **Compile options**: `DEFAULT_QNN_OPTIONS` in `utils/onnx_ep.py`. The ones that shape the
  binary (`htp_graph_finalization_optimization_mode`, `enable_htp_fp16_precision`,
  `offload_graph_io_quantization`, `soc_model`, `htp_arch`, `vtcm_mb`) are part of its
  fingerprint.

To bump any of them, edit `requirements.in` and regenerate (command in the file header),
or replace the model files, and then **recompile the context binaries**.

CI enforces it: `tests/containers/ai/test_ocr_runner_context_binaries.py` compares every
committed `*.qnn_ctx.json` with the `==` pins in `requirements.txt`, the `# qairt-version:`
line in `requirements.in`, the compile-time subset of `DEFAULT_QNN_OPTIONS` and the size of
the `.onnx` it was compiled from, and checks they were compiled on the supported SoC
(`soc_id` 675, QCS8275). A wheel bump, a model swap or an option change without recompiled
binaries fails the test suite with the recompile command in the message. What CI cannot
check is whether the binaries actually run on a board: that is the runtime `soc_id` check.

## HTP context binaries (the 7-minute problem)

The first QNN session on a model compiles the graph for the HTP. Graph finalization is
single-threaded inside QNN: on the 21q board (QCS8275, HTP v75: the backend loads `libQnnHtpV75Stub.so`) the detector
takes 2.1 s and the recognizer 130 s with `htp_graph_finalization_optimization_mode=0`
(6.5 min with mode 3). The compiled result is written next to the model as
`<model>.soc<soc_id>.qnn_ctx.onnx` plus a `<model>.soc<soc_id>.qnn_ctx.json` fingerprint,
and every later start loads it in 0.25 s.

The SoC id in the name (`/sys/devices/soc0/soc_id`, 675 for the QCS8275) is what makes a
**multi-SoC image** possible: ship one pair per supported SoC side by side and the runner
loads the one matching the board it is on. Supporting another SoC means running the
compile script once on such a board and committing its pair; `SUPPORTED_SOCS` in
`tests/containers/ai/test_ocr_runner_context_binaries.py` lists the SoCs that must have
binaries. The unsuffixed `<model>.qnn_ctx.onnx` is only written and looked up when the SoC
cannot be identified.

The binaries are compiled once on the target and **committed** (`detector.soc675.qnn_ctx.onnx`
21.3 MB, `recognizer.soc675.qnn_ctx.onnx` 10.5 MB, compiled 2026-09-08 on the 21q board with
`onnxruntime-qnn` 2.5.0 / QAIRT 2.49.40), so no container ever compiles. Measured
placement: detector 1 QNN partition, 100 % on the NPU; recognizer 1 QNN partition plus 4
`DequantizeLinear` nodes on the CPU (1.9 % of the time). To regenerate them:

```bash
# on the board, from this directory, with requirements.txt installed
python tools/compile_htp_context.py
```

The script (docstring has the details, including how to run it inside the runner image)
opens each graph on QNN exactly like the runner does, writes the binary and its
fingerprint, measures how much of the graph the NPU really executes and fails if it ran
nothing, then reopens the model from the binary and reports the warm start time. Commit
`models/easyocr-onnx-w8a8/*.soc<id>.qnn_ctx.onnx` and `.json` and rebuild the image.

Lookup order at start-up (`_find_context_binary`):

1. `EASYOCR_QNN_CONTEXT_DIR` if set, otherwise the model directory when writable, otherwise
   `$XDG_CACHE_HOME/easyocr-onnx/qnn-context` (the fallback also used for *writing* when the
   model directory is read-only);
2. the model directory - where the **committed, pre-compiled binaries** live. This works
   with a read-only model directory, so binaries baked into the image are used as-is.

In each directory `<model>.soc<soc_id>.qnn_ctx.onnx` is tried first, then the unsuffixed
name.

A binary is only valid for the SoC/HTP architecture, QAIRT release, model file and compile
options that produced it. The `.json` records `onnxruntime`, `onnxruntime_qnn`,
`qnn_version` (the bundled QAIRT), the backend library, the model size, the compile
options, and the SoC it was compiled on (`soc_id` and `soc_machine` from
`/sys/devices/soc0`, e.g. `675` / `QCS8275`). On load it is compared with the current
setup:

* a **different `soc_id`** rejects the binary outright, with an `ERROR` line naming both
  SoCs: HTP code compiled for one SoC does not run on another. The same SoC on a board from
  another vendor has the same `soc_id`, so it passes. The brick's compose file mounts
  `/sys/devices/soc0` read-only into the container for this check; without it the runner
  logs that the SoC cannot be identified and loads the binary unchecked;
* every other difference is logged by name and the binary is still loaded
  (`EASYOCR_QNN_CONTEXT_STRICT=1` recompiles instead).

A binary QNN itself rejects is deleted and recompiled, and if it cannot be deleted
(read-only image layer) the recompiled one lands in the cache directory and shadows it from
then on.

Without a usable binary the first start of a container pays the compile, well past the
brick's 30 s connection timeout and the compose healthcheck. That is the failure mode a
fingerprint mismatch must never degrade into silently; see the section above for what
invalidates the binaries.

## Running outside Docker (on the board)

```bash
uv venv .venv --python 3.13 && source .venv/bin/activate
uv pip install --require-hashes -r requirements.txt   # aarch64 only; on x86_64 use requirements.in (CPU)
python tools/compile_htp_context.py                    # compiles, then reports the measured NPU/CPU split
EASYOCR_QNN_VERIFY=1 python -c "import inference"      # loads both models like the runner and reports the placement
```

Partial offload is normal (4 `DequantizeLinear` nodes stay on the CPU in the recognizer,
~2 % of the time); many partitions are not. If the backend does not come up at all
(`QNN_DEVICE_ERROR_INVALID_CONFIG`, "Failed to create device"), the usual causes in order
are: an `ADSP_LIBRARY_PATH` that does not hold the skels matching `libQnnHtp.so` (ORT's own
warning `Using existing ADSP_LIBRARY_PATH setting of ...` names the directory in use; the
runner switches to the wheel's silently), FastRPC permissions (`/dev/fastrpc-cdsp`
and `/dev/dma_heap/system` not passed to the container), and an `htp_arch` / `soc_model`
forced through the environment that the SoC does not have. `EASYOCR_ORT_LOG_LEVEL=0` makes
ORT print the QNN error verbatim.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `EASYOCR_EP` | `auto` | `auto` (QNN, CPU fallback) \| `qnn` (fail unless the NPU executes nodes) \| `cpu` |
| `EASYOCR_EP_DETECTOR` / `EASYOCR_EP_RECOGNIZER` | - | per-model override |
| `EASYOCR_DETECTOR_MODEL` / `EASYOCR_RECOGNIZER_MODEL` | `models/easyocr-onnx-w8a8/*.onnx` | model paths |
| `EASYOCR_QNN_CONTEXT_CACHE` | `1` | use/write the compiled HTP graph |
| `EASYOCR_QNN_CONTEXT_DIR` | model directory | where to read/write that cache |
| `EASYOCR_QNN_CONTEXT_STRICT` | `0` | `1` = recompile instead of loading a binary whose fingerprint differs |
| `EASYOCR_QNN_STRICT` | `0` | `1` = refuse to run unless the whole graph is on the NPU |
| `EASYOCR_QNN_VERIFY` | `0` | `1` = measure the real node placement at start-up (implied by `EASYOCR_EP=qnn`) |
| `EASYOCR_QNN_BACKEND_PATH` | wheel's `libQnnHtp.so` if present, else `libQnnHtp.so` via the loader | HTP backend library |
| `EASYOCR_QNN_ADSP_PATH` | wheel directory | `ADSP_LIBRARY_PATH` for the DSP skel libraries |
| `EASYOCR_QNN_KEEP_ADSP_PATH` | `0` | `1` = keep the inherited `ADSP_LIBRARY_PATH` untouched |
| `EASYOCR_QNN_PERF_MODE` | `sustained_high_performance` | `burst`, `balanced`, `power_saver`, ... See the note on `burst` below |
| `EASYOCR_QNN_FINALIZATION_MODE` | `0` | `0` fastest compile ... `3` slowest compile / fastest run (invalidates binaries) |
| `EASYOCR_QNN_FP16` / `EASYOCR_QNN_OFFLOAD_IO_QUANT` | `1` / `1` | `enable_htp_fp16_precision` / `offload_graph_io_quantization` overrides (invalidate binaries) |
| `EASYOCR_QNN_SOC_MODEL` / `EASYOCR_QNN_HTP_ARCH` / `EASYOCR_QNN_VTCM_MB` | - | target-specific tuning (invalidates binaries) |
| `EASYOCR_PARALLEL_INIT` | `0` | `1` = compile the two models on two threads |
| `EASYOCR_QNN_PROFILING` | `off` | `basic` \| `detailed` (QNN-internal profiling) |
| `EASYOCR_QNN_OP_TRACE` | `0` | `1` = dump the ONNX-op to QNN-op mapping |
| `EASYOCR_ORT_PROFILE` | `0` | `1` = write an ORT profile showing the QNN/CPU node split |
| `EASYOCR_QNN_RPC_LATENCY` | - | per-run RPC control latency, microseconds |
| `EASYOCR_ORT_LOG_LEVEL` | `3` | ORT log severity: `3` errors only, `2` warnings, `0` verbose (prints the partitioning) |
| `EASYOCR_ORT_THREADS` | - | intra-op threads for the CPU provider |
| `EASYOCR_DEBUG` | `0` | `1` = stage-by-stage timings and applied config on stdout. One INFO summary line per frame is always logged |

## Why not `burst`

`htp_performance_mode=burst` is the QNN mode with the highest clocks, but with `burst` ORT
also asks fastrpc for RPC polling QoS (`rpc_polling_time=9999`, i.e. `RPC_POLL_QOS`). The
container ships its own fastrpc 1.0.6, built from source in `qairt-common-base`, and there
`manage_poll_qos` fails; QNN rejects the **whole** power configuration, DCVS included, ORT
logs `Unable to set HTP power configurations` and the HTP stays at default clocks. Measured
in the container on the 21q: detector invoke 103 ms and 65 ms per recognizer call with
`burst` or `default`, 22 ms and 15 ms with `sustained_high_performance`, which asks for no
polling. On the host, whose fastrpc is 1.0.15 (`qcom-fastrpc1`), `burst` works and is no
faster than `sustained_high_performance` for this workload. Hence the default. If the base
image moves to a fastrpc that accepts `RPC_POLL_QOS`, `burst` becomes an option again; the
runtime-only nature of this option means switching it never invalidates the context binaries.

## Models

`models/easyocr-onnx-w8a8`: `uint8` I/O, NCHW, static shapes - detector `[1, 3, 608, 800]`,
recognizer `[1, 1, 64, 800]`. Each graph is `<model>.onnx` plus external weights
`<model>.data`; `metadata.json` carries the per-tensor scale/zero-point of the graph
boundaries (ONNX keeps none), which `ONNXModel` needs to feed images and read logits.
**Keep the three together.**

ai-hub also publishes a float export. It is the same size on disk (w8a8 is a QDQ graph:
weights quantized in value, still stored as float32) but ~2x slower on the CPU and only
runs on the HTP as emulated fp16, so it is not used.
