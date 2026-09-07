# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Find out why the QNN HTP backend will not come up, and which configuration fixes it.

    python tools/qnn_doctor.py

Bringing the Hexagon NPU up on aarch64 Linux fails in a handful of well-known ways, and
the QNN error messages rarely name the actual cause. `QNN_DEVICE_ERROR_INVALID_CONFIG:
Invalid config values` at device creation, for instance, is usually one of:

  * ADSP_LIBRARY_PATH pointing at another project's DSP libraries, so the DSP loads skel
    libraries from a different QAIRT version than the one the wheel expects
  * SoC auto-detection failing on non-Android Linux, leaving an empty device config that
    QNN then rejects - fixed by naming htp_arch and soc_model explicitly
  * a genuine mismatch between the wheel's QAIRT and the board's DSP firmware

and `remote_handle64_open failed` / `untrusted app trying to offload to signed remote
process` is a different problem entirely: FastRPC permissions, usually needing root or
membership of the right group.

This script collects the facts (skel libraries, SoC id, permissions), infers what the
settings should be, then actually tries to open a model across a matrix of configurations
and reports which ones get the NPU running.
"""

from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort

from utils.constants import RECOGNIZER_MODEL_PATH
from utils.onnx_ep import QNN_EP_NAME, _plugin_module

# Where the DSP looks for its skel libraries, in the order QNN documents for Linux.
SKEL_SEARCH_PATHS = ("/usr/lib/rfsa/adsp", "/dsp", "/opt", "/vendor/lib/rfsa/adsp")
SKEL_PATTERN = re.compile(r"libQnnHtpV(\d+)Skel\.so$")


def find_skels() -> dict[str, list[str]]:
    """Locate libQnnHtpV<arch>Skel.so on the usual DSP search paths, plus the wheel's own."""
    found: dict[str, list[str]] = {}

    candidates = list(SKEL_SEARCH_PATHS)
    for variable in ("ADSP_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.extend(part for part in value.split(";" if ";" in value else ":") if part)

    plugin = _plugin_module()
    if plugin is not None:
        try:
            candidates.append(os.path.dirname(plugin.get_qnn_htp_path()))
        except Exception:  # noqa: BLE001
            pass

    for directory in dict.fromkeys(candidates):
        if not os.path.isdir(directory):
            continue
        skels = sorted(os.path.basename(p) for p in glob.glob(os.path.join(directory, "libQnnHtpV*Skel.so")))
        if skels:
            found[directory] = skels
    return found


def infer_arch(skels: dict[str, list[str]]) -> str | None:
    """The HTP architecture number, read off the skel library names."""
    arches = {match.group(1) for names in skels.values() for name in names if (match := SKEL_PATTERN.search(name))}
    if len(arches) == 1:
        return arches.pop()
    return None


def read_soc() -> dict[str, str]:
    facts = {}
    for name in ("soc_id", "machine", "family", "revision"):
        path = f"/sys/devices/soc0/{name}"
        try:
            with open(path, encoding="utf-8") as handle:
                facts[name] = handle.read().strip()
        except OSError:
            pass
    return facts


def report_facts() -> tuple[dict[str, list[str]], str | None]:
    print("=" * 78)
    print("environment")
    print("=" * 78)
    print(f"  onnxruntime            {ort.__version__}")
    plugin = _plugin_module()
    print(f"  onnxruntime-qnn        {getattr(plugin, '__version__', 'NOT INSTALLED')}")
    if plugin is not None:
        try:
            print(f"  HTP backend library    {plugin.get_qnn_htp_path()}")
        except Exception as exc:  # noqa: BLE001
            print(f"  HTP backend library    unresolved: {exc}")
    print(f"  ADSP_LIBRARY_PATH      {os.environ.get('ADSP_LIBRARY_PATH', '(unset - ORT will set its own)')}")
    print(f"  LD_LIBRARY_PATH        {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")
    if hasattr(os, "geteuid"):
        print(f"  running as root        {os.geteuid() == 0}")

    soc = read_soc()
    print(f"  SoC                    {soc if soc else '(no /sys/devices/soc0)'}")

    skels = find_skels()
    print()
    print("HTP skel libraries (what the DSP will actually load)")
    if not skels:
        print("  NONE FOUND on any search path.")
        print("  Without a libQnnHtpV<arch>Skel.so the HTP backend cannot start. Install the")
        print("  board's QAIRT/DSP runtime package, or check /usr/lib/rfsa/adsp exists.")
    for directory, names in skels.items():
        print(f"  {directory}")
        for name in names:
            print(f"      {name}")

    arch = infer_arch(skels)
    if arch:
        print(f"\n  -> inferred htp_arch: {arch}")
    elif skels:
        print("\n  -> several architectures present, cannot infer htp_arch automatically")
    print()
    return skels, arch


def try_config(label: str, model_path: str, env: dict[str, str | None], options: dict[str, str]) -> bool:
    """Open `model_path` on QNN under a specific env/provider-option combination."""
    saved = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = "qnn_doctor"
    plugin = _plugin_module()
    if plugin is not None and "backend_path" not in options:
        try:
            options = {**options, "backend_path": plugin.get_qnn_htp_path()}
        except Exception:  # noqa: BLE001
            pass

    try:
        from utils.onnx_ep import _create_qnn_session, _register_plugin

        if plugin is not None:
            _register_plugin(plugin)
        session = _create_qnn_session(model_path, plugin, options, session_options)

        model_input = session.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in model_input.shape]
        dtype = np.uint8 if "uint8" in model_input.type else np.float32
        session.run(None, {model_input.name: np.zeros(shape, dtype=dtype)})

        from utils.onnx_ep import summarize_profile

        profile_path = session.end_profiling()
        placement = summarize_profile(profile_path)
        try:
            os.remove(profile_path)
        except OSError:
            pass

        on_npu = placement.get(QNN_EP_NAME, (0, 0.0))[0]
        if on_npu:
            share = placement[QNN_EP_NAME][1] / (sum(d for _, d in placement.values()) or 1) * 100
            print(f"  [OK]   {label}: {on_npu} NPU partition(s), {share:.0f}% of runtime")
            return True
        print(f"  [CPU]  {label}: session opened but QNN executed nothing")
        return False
    except Exception as exc:  # noqa: BLE001
        first_line = str(exc).strip().splitlines()[0][:150]
        print(f"  [FAIL] {label}: {first_line}")
        return False
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    model_path = os.environ.get("EASYOCR_RECOGNIZER_MODEL", RECOGNIZER_MODEL_PATH)
    if not os.path.isfile(model_path):
        print(f"{model_path} missing - run python tools/download_models.py first")
        return 1

    skels, arch = report_facts()

    print("=" * 78)
    print(f"trying configurations on {model_path}")
    print("=" * 78)
    print("  (QNN prints its own errors between these lines; the [tag] is the verdict)\n")

    adsp_default = ";".join(path for path in SKEL_SEARCH_PATHS if os.path.isdir(path))
    soc_id = read_soc().get("soc_id")

    attempts: list[tuple[str, dict[str, str | None], dict[str, str]]] = [
        ("as-is (current environment)", {}, {}),
        ("ADSP_LIBRARY_PATH unset", {"ADSP_LIBRARY_PATH": None}, {}),
    ]
    if adsp_default:
        attempts.append((f"ADSP_LIBRARY_PATH={adsp_default}", {"ADSP_LIBRARY_PATH": adsp_default}, {}))
    if arch:
        attempts.append(
            (f"htp_arch={arch}, ADSP unset", {"ADSP_LIBRARY_PATH": None}, {"htp_arch": arch}),
        )
        if soc_id:
            attempts.append(
                (
                    f"htp_arch={arch}, soc_model={soc_id}, ADSP unset",
                    {"ADSP_LIBRARY_PATH": None},
                    {"htp_arch": arch, "soc_model": soc_id},
                ),
            )
    # A CPU-backend QNN session isolates "QNN itself is broken" from "the HTP is broken".
    attempts.append(
        (
            "QNN CPU backend (sanity check, not the NPU)",
            {},
            {"backend_path": "libQnnCpu.so"},
        ),
    )

    working = [label for label, env, options in attempts if try_config(label, model_path, env, options)]

    print()
    print("=" * 78)
    if working:
        print("configurations that reached the NPU:")
        for label in working:
            print(f"  * {label}")
        print("\nTranslate the winning one into env vars, e.g.:")
        print("  unset ADSP_LIBRARY_PATH")
        if arch:
            print(f"  export EASYOCR_QNN_HTP_ARCH={arch}")
        if soc_id:
            print(f"  export EASYOCR_QNN_SOC_MODEL={soc_id}   # only if it was needed above")
    else:
        print("nothing reached the NPU.")
        if not skels:
            print("  No HTP skel libraries on the board - install the QAIRT/DSP runtime package.")
        elif hasattr(os, "geteuid") and os.geteuid() != 0:
            print("  Try again as root. FastRPC on aarch64 Linux often refuses unprivileged")
            print("  processes ('untrusted app trying to offload to signed remote process'),")
            print("  and that failure mode is a permissions problem, not a configuration one.")
        else:
            print("  Likely a QAIRT version mismatch between the onnxruntime-qnn wheel and the")
            print("  board's DSP firmware. Compare the wheel's QAIRT with the board's skel")
            print("  libraries and align them.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
