# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Expose AI Hub QAIRT bundles as a GenieX model store without copying them.

Every ``<models>/<dir>/metadata.json`` whose ``runtime`` is accepted becomes
``<store>/models/qualcomm/<model_id>/``: one symlink per bundle file plus the
``geniex.json`` manifest GenieX reads to load it. A bare model name in a request
(``qwen3_4b_instruct_2507``) resolves to ``qualcomm/<name>`` in the store.
The models directory is only read. Bundles still being downloaded (``.download``
marker) or without a context binary are skipped.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ORG = "qualcomm"
MANIFEST = "geniex.json"
DOWNLOAD_MARKER = ".download"
DEFAULT_RUNTIMES = ("GENIEX_QAIRT",)


def _log(message: str) -> None:
    print(f"make-store: {message}", file=sys.stderr, flush=True)


def _enum_value(value: object, prefix: str) -> str:
    """Normalize a metadata.json enum: newer bundles write ``RUNTIME_GENIE``, older ones ``genie``."""
    return str(value or "").upper().removeprefix(prefix)


def build_manifest(bundle: Path, metadata: dict) -> dict | None:
    """Return the geniex.json content for *bundle*, or None when it has no context binary."""
    files = sorted(p for p in bundle.iterdir() if p.is_file() and not p.name.startswith("."))
    names = [p.name for p in files]
    # model_files is a list in recent bundles and an object keyed by graph name in older ones
    model_files = metadata.get("model_files") or []
    entries = model_files.values() if isinstance(model_files, dict) else model_files
    listed = [f.get("filename", "") for f in entries if isinstance(f, dict)]
    entry = next((n for n in listed if n.endswith(".bin") and n in names), None)
    entry = entry or next((n for n in names if n.endswith(".bin")), None)
    if entry is None:
        return None

    def info(path: Path) -> dict:
        return {"Name": path.name, "Downloaded": True, "Size": path.stat().st_size}

    precision = _enum_value(metadata.get("precision"), "PRECISION_") or "N/A"
    vision = bool(metadata.get("genie", {}).get("supports_vision"))
    return {
        "Name": f"{ORG}/{metadata['model_id']}",
        "ModelName": metadata.get("model_name") or metadata["model_id"],
        "ModelType": "vlm" if vision else "llm",
        "PluginId": "qairt",
        "ModelFile": {precision: info(bundle / entry)},
        "ExtraFiles": [info(p) for p in files if p.name != entry],
    }


def make_store(models_dir: Path, store_dir: Path, runtimes: tuple[str, ...] = DEFAULT_RUNTIMES) -> list[str]:
    """Rebuild ``<store_dir>/models/qualcomm`` from the bundles in *models_dir*; return the model names."""
    org_dir = store_dir / "models" / ORG
    shutil.rmtree(org_dir, ignore_errors=True)
    org_dir.mkdir(parents=True)

    served: list[str] = []
    for meta_path in sorted(models_dir.glob("*/metadata.json")):
        bundle = meta_path.parent
        if (bundle / DOWNLOAD_MARKER).exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log(f"skipping {bundle.name}: unreadable metadata.json ({exc})")
            continue
        if _enum_value(metadata.get("runtime"), "RUNTIME_") not in runtimes or not metadata.get("model_id"):
            continue
        manifest = build_manifest(bundle, metadata)
        if manifest is None:
            _log(f"skipping {bundle.name}: no .bin context binary")
            continue

        model_id = metadata["model_id"]
        target = org_dir / model_id
        if target.exists():
            _log(f"{model_id}: {bundle.name} replaces an earlier bundle with the same model_id")
            shutil.rmtree(target)
        target.mkdir()
        for file_info in [*manifest["ModelFile"].values(), *manifest["ExtraFiles"]]:
            os.symlink(bundle / file_info["Name"], target / file_info["Name"])
        (target / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        served.append(model_id)
    return served


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models_dir", type=Path, help="directory holding one AI Hub bundle per subdirectory")
    parser.add_argument("store_dir", type=Path, help="GenieX data directory (GENIEX_DATADIR)")
    parser.add_argument("--runtime", action="append", help="accepted metadata.json runtime, e.g. GENIE (default: GENIEX_QAIRT)")
    args = parser.parse_args()
    served = make_store(args.models_dir, args.store_dir, tuple(args.runtime or DEFAULT_RUNTIMES))
    _log(f"{len(served)} model(s): {', '.join(served) or 'none'}")


if __name__ == "__main__":
    main()
