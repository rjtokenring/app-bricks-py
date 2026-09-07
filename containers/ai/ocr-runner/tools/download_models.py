# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Fetch the pre-exported EasyOCR ONNX assets from Qualcomm AI Hub and verify them.

These are release zips Qualcomm publishes for AI Hub model v0.61.0, so no AI Hub account,
job submission or PyTorch install is needed. The Dockerfile runs this at build time; run
it by hand to work on the pipeline outside the container.

Only the w8a8 (quantized) export is fetched. The float export is the same size on disk -
w8a8 is a QDQ graph whose weights are quantized in value but still stored as float32 - so
it costs nothing to skip, and it is the variant the Hexagon NPU runs natively.

Every byte is pinned: the zip must match ZIP_SHA256 and each extracted file must match
models/easyocr-onnx-w8a8/SHA256SUMS. A compiled HTP context binary is only valid for the
exact model that produced it, so a silent upstream change to the models would silently
invalidate the pre-compiled binaries shipped next to them.

Usage:
    python tools/download_models.py
    python tools/download_models.py --from-manifest   # resolve the URL from Hugging Face instead of the pin
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

RELEASE_MANIFEST = "https://huggingface.co/qualcomm/EasyOCR/raw/main/release_assets.json"

# Pinned so a silent upstream bump cannot change the models under an existing install.
# Refresh by reading RELEASE_MANIFEST (--from-manifest), then update ZIP_SHA256 and
# SHA256SUMS - and recompile the HTP context binaries.
PINNED_RELEASE = "v0.61.0"
PRECISION = "w8a8"
PINNED_URL = (
    f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/easyocr/releases/{PINNED_RELEASE}/easyocr-onnx-{PRECISION}.zip"
)
ZIP_SHA256 = "a51144c8bc377f6514fd9878471c3a1dc1b5ff4aa32cb66367d0761c50879203"

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CHECKSUMS_FILE = "SHA256SUMS"


def url_from_manifest() -> str | None:
    """Read the current w8a8 ONNX download URL off the Hugging Face release manifest."""
    with urllib.request.urlopen(RELEASE_MANIFEST, timeout=60) as response:
        manifest = json.load(response)
    entry = manifest.get("precisions", {}).get(PRECISION, {})
    onnx = entry.get("universal_assets", {}).get("onnx", {})
    return onnx.get("download_url")


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(destination: str) -> dict[str, str]:
    """Parse the `sha256sum`-style SHA256SUMS file tracked next to the models."""
    path = os.path.join(destination, CHECKSUMS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing; it pins the model bytes and must be tracked in git")
    checksums: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(maxsplit=1)
            checksums[name.lstrip("*")] = digest.lower()
    return checksums


def verify_files(destination: str) -> None:
    """Fail unless every file listed in SHA256SUMS is present and matches."""
    for name, expected in read_checksums(destination).items():
        path = os.path.join(destination, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"{name} missing from {destination}")
        actual = sha256_of(path)
        if actual != expected:
            raise RuntimeError(f"{name}: sha256 {actual} does not match the pinned {expected}")
        print(f"  {name}: OK")


def download(url: str, destination: str, expected_sha256: str | None) -> None:
    """Download `url`, check its hash, and unpack it into `destination` (flattening the zip's top folder)."""
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as response:
        payload = response.read()
    print(f"  {len(payload) / 1e6:.1f} MB")

    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"zip sha256 {actual} does not match the pinned {expected_sha256}; refusing to unpack")
    if expected_sha256 is None:
        print(f"  zip sha256 {actual} (unpinned: --from-manifest)")

    # Only the files inside the zip are (over)written: SHA256SUMS, .gitignore and any
    # pre-compiled *.qnn_ctx.onnx binaries next to the models stay untouched.
    os.makedirs(destination, exist_ok=True)
    print(f"  unpacking into {destination}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            # The archives wrap everything in a single easyocr-onnx-<precision>/ folder.
            name = os.path.basename(member.filename)
            if not name:
                continue
            with archive.open(member) as source, open(os.path.join(destination, name), "wb") as target:
                target.write(source.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help=f"Resolve the URL from {RELEASE_MANIFEST} instead of the pinned {PINNED_RELEASE} one (skips the zip hash check).",
    )
    parser.add_argument(
        "--skip-file-checksums",
        action="store_true",
        help="Do not verify the extracted files against SHA256SUMS (only useful together with --from-manifest to update the pins).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = os.path.join(MODELS_DIR, f"easyocr-onnx-{PRECISION}")

    if args.from_manifest:
        url = url_from_manifest()
        if url is None:
            print(f"No ONNX asset published for precision {PRECISION!r}", file=sys.stderr)
            return 1
        download(url, destination, expected_sha256=None)
    else:
        download(PINNED_URL, destination, expected_sha256=ZIP_SHA256)

    if args.skip_file_checksums:
        for name in sorted(os.listdir(destination)):
            print(f"  {name}  sha256 {sha256_of(os.path.join(destination, name))}")
        return 0

    print("verifying against SHA256SUMS")
    verify_files(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
