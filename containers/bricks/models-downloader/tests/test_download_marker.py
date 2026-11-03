# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the ".download" marker of downloads sharing a directory, and their locks."""

from common.download_lock import DirectoryDownloads
from common.download_marker import MARKER_NAME, add_download, marker_downloads, read_marker, remove_download, write_marker


def test_a_single_download_reads_like_a_marker_written_by_one(tmp_path):
    add_download(str(tmp_path), "a", handler="hf-handler", model_url="org/repo:Q4_0", file_patterns=["*Q4_0*.gguf"], size_mb=1.5)

    marker = read_marker(str(tmp_path / MARKER_NAME))
    assert marker is not None
    assert marker["model_url"] == "org/repo:Q4_0"
    assert marker["file_patterns"] == ["*Q4_0*.gguf"]
    assert marker["size_mb"] == 1.5


def test_downloads_share_the_marker_and_the_last_one_out_removes_it(tmp_path):
    add_download(str(tmp_path), "a", handler="hf-handler", model_url="org/repo:Q4_0", file_patterns=["*Q4_0*.gguf"], size_mb=1.5)
    add_download(str(tmp_path), "b", handler="hf-handler", model_url="org/repo:Q8_0", file_patterns=["*Q8_0*.gguf"], size_mb=3.0)

    marker = read_marker(str(tmp_path / MARKER_NAME))
    assert marker is not None
    # A reader that knows nothing of "downloads" still sees both in flight...
    assert marker["file_patterns"] == ["*Q4_0*.gguf", "*Q8_0*.gguf"]
    # ...and no single download's url or size passed off as the directory's.
    assert "model_url" not in marker and "size_mb" not in marker
    assert [(d["handler"], d["model_url"], d["size_mb"]) for d in marker_downloads(marker)] == [
        ("hf-handler", "org/repo:Q4_0", 1.5),
        ("hf-handler", "org/repo:Q8_0", 3.0),
    ]

    assert [d["id"] for d in remove_download(str(tmp_path), "a")] == ["b"]
    marker = read_marker(str(tmp_path / MARKER_NAME))
    assert marker is not None
    assert (marker["model_url"], marker["file_patterns"]) == ("org/repo:Q8_0", ["*Q8_0*.gguf"])

    assert remove_download(str(tmp_path), "b") == []
    assert not (tmp_path / MARKER_NAME).exists()


def test_a_legacy_marker_is_one_download_that_stays_until_discarded(tmp_path):
    """A marker without "downloads" is a single download with no id: a run killed before
    the field existed. Adding to it keeps it; only the discard of a killed run removes it."""
    write_marker(str(tmp_path), handler="hf-handler", file_patterns=["*Q3_K_S*.gguf"])
    assert marker_downloads(read_marker(str(tmp_path / MARKER_NAME))) == [read_marker(str(tmp_path / MARKER_NAME))]

    add_download(str(tmp_path), "a", handler="hf-handler", file_patterns=["*Q4_0*.gguf"])
    left = remove_download(str(tmp_path), "a")
    assert [d.get("file_patterns") for d in left] == [["*Q3_K_S*.gguf"]]
    assert (tmp_path / MARKER_NAME).is_file()


def test_a_running_download_is_seen_by_the_others_until_it_leaves(tmp_path):
    first = DirectoryDownloads(str(tmp_path), "org/repo")
    second = DirectoryDownloads(str(tmp_path), "org/repo")
    elsewhere = DirectoryDownloads(str(tmp_path), "org/other")

    assert not second.others_running()
    with first.bookkeeping():
        first.join()
    assert second.others_running()
    assert not elsewhere.others_running()
    first.leave()
    assert not second.others_running()
