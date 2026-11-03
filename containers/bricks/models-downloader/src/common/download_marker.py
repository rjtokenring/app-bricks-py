# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Read/write the per-model ".download" in-progress marker.

The marker lives at ``<model_dir>/.download`` and exists only while a download is
in progress (or was interrupted before completing). It is a JSON document so
external tools can match the in-progress model back to models-list.yaml:

    {
      "status": "downloading",
      "handler": "hf-handler",
      "models_repository": "llamacpp",
      "model_directory": "google/gemma-4-E2B-it-qat-q4_0-gguf",
      "file_patterns": ["*Q4_0*.gguf"]
    }

``file_patterns`` names the files being downloaded, and matters when the directory can
hold more than one model: a Hugging Face repository directory is shared by every
quantization ever downloaded from that repository, so the bare presence of a marker there
does not mean that all of them are in progress. It is optional — a handler whose model
directory holds exactly one model has nothing to disambiguate, and a marker written
before the field existed carries none — and its absence means the whole directory.

``size_mb`` is the size the download will have once complete (``common/model_size.py``),
written when the handler knows it up front. The listing reports it for a download in
progress that models-list.yaml does not size — an ad-hoc Hugging Face model — so such a
model is sized before any of its files exist. It is optional as well.

Several downloads can be in flight in one directory at once (``common/download_lock.py``).
Each adds an entry to ``downloads`` with its own ``id``, ``model_url``, ``file_patterns``
and ``size_mb``, and takes it out again when it ends. The top-level ``file_patterns`` is
the union of theirs, and ``model_url`` and ``size_mb`` are set only while there is just
one, so a reader that knows nothing of ``downloads`` still finds every file in flight
covered. ``marker_downloads`` splits a marker back into one marker per download.
"""

import json
import os

MARKER_NAME = ".download"


def marker_payload(handler="", models_repository="", model_directory="", model_url="", file_patterns=None, size_mb=None):
    """Build the marker dict from the fields that match a models-list.yaml entry.

    ``model_url`` is included only when set (some handlers download by URL), and so are
    ``file_patterns`` (only handlers sharing a directory between models need it) and
    ``size_mb`` (only a handler that knows the size before downloading can tell it).
    """
    payload = {
        "status": "downloading",
        "handler": handler or "",
        "models_repository": models_repository or "",
        "model_directory": model_directory or "",
    }
    if model_url:
        payload["model_url"] = model_url
    if file_patterns:
        payload["file_patterns"] = list(file_patterns)
    if size_mb is not None:
        payload["size_mb"] = size_mb
    return payload


def write_marker(model_dir, handler="", models_repository="", model_directory="", model_url="", file_patterns=None, size_mb=None):
    """Create ``<model_dir>/.download`` with the JSON payload and return its path."""
    path = os.path.join(model_dir, MARKER_NAME)
    _replace_marker(path, marker_payload(handler, models_repository, model_directory, model_url, file_patterns, size_mb))
    return path


def _replace_marker(path, payload):
    """Write *payload* to *path* by rename: a reader never sees half a marker."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, path)


def marker_downloads(marker):
    """The downloads *marker* stands for, each as a marker of its own.

    A marker without ``downloads`` (another handler, or one written before the field
    existed) is a single download. Each entry of ``downloads`` comes back with the fields
    every download of the directory shares filled in.
    """
    if not marker:
        return []
    entries = marker.get("downloads")
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        return [marker]
    return [{**_shared_fields(marker), **entry} for entry in entries]


def add_download(model_dir, download_id, handler="", models_repository="", model_directory="", model_url="", file_patterns=None, size_mb=None):
    """Add download *download_id* to ``<model_dir>/.download``, next to any already there.

    Read-modify-write: callers serialize on the directory's bookkeeping lock.
    """
    path = os.path.join(model_dir, MARKER_NAME)
    entry = {"id": download_id, **_download_fields(marker_payload(model_url=model_url, file_patterns=file_patterns, size_mb=size_mb))}
    entries = [_download_fields(d) for d in marker_downloads(read_marker(path))]
    _write_downloads(path, marker_payload(handler, models_repository, model_directory), [*entries, entry])
    return path


def remove_download(model_dir, download_id):
    """Take download *download_id* out of ``<model_dir>/.download``; return the ones left.

    The marker goes with the last of them. Serialized like ``add_download``.
    """
    path = os.path.join(model_dir, MARKER_NAME)
    marker = read_marker(path)
    left = [d for d in marker_downloads(marker) if d.get("id") != download_id]
    if marker is not None and not left:
        os.unlink(path)
    elif marker is not None:
        _write_downloads(path, _shared_fields(marker), [_download_fields(d) for d in left])
    return left


# What one download records, as opposed to what every download of a directory shares.
DOWNLOAD_FIELDS = ("id", "model_url", "file_patterns", "size_mb")


def _download_fields(marker):
    return {k: v for k, v in marker.items() if k in DOWNLOAD_FIELDS}


def _shared_fields(marker):
    return {k: v for k, v in marker.items() if k != "downloads" and k not in DOWNLOAD_FIELDS}


def _write_downloads(path, shared, entries):
    """Write the marker of the downloads *entries* (see the module docstring)."""
    payload = dict(shared)
    if len(entries) == 1:
        payload.update({k: v for k, v in entries[0].items() if k in ("model_url", "size_mb")})
    patterns = list(dict.fromkeys(p for entry in entries for p in entry.get("file_patterns") or []))
    if patterns:
        payload["file_patterns"] = patterns
    payload["downloads"] = entries
    _replace_marker(path, payload)


def read_marker(path):
    """Parse a marker file into a dict, or None if it is missing / unreadable.

    Backward compatible with the legacy plaintext format (a bare model directory
    name), which is returned as a payload with that value as ``model_directory``.
    """
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    if not text:
        return marker_payload()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        return data
    return marker_payload(model_directory=text)
