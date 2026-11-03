# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Coordinate the downloads that share one model directory.

A Hugging Face repository directory holds every file ever fetched from that repository
(its quantizations, or its voices), and each download is its own process, possibly its
own container on the same models volume. The transfers run in parallel; the bookkeeping
around them (the ".download" marker, the metadata record, huggingface_hub's ".cache"
tree, the directory itself) is shared, and is coordinated with two flock(2) locks:

* the bookkeeping lock, held exclusively for the short sections that read or change
  the shared bookkeeping, never for a transfer;
* the presence lock, held shared by every download for as long as it runs. The kernel
  releases it however the process ends, SIGKILL included.

A download joins and leaves under the bookkeeping lock, together with the marker update
it goes with. So whoever holds the bookkeeping lock and finds the presence lock free
knows that every download the marker still names was killed: its leftovers can go.

The lock files live in the models root, not in the directory they guard, which can be
removed and recreated while they are held. They are never deleted: removing a lock file
lets the next process lock a new file while another still holds the old one.
"""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

LOCKS_DIR = ".download-locks"


class DirectoryDownloads:
    """The downloads running into ``<base_dir>/<directory>``, from any process."""

    def __init__(self, base_dir: str, directory: str):
        name = directory.strip("/").replace("/", "--")
        self._locks_dir = os.path.join(base_dir, LOCKS_DIR)
        self._bookkeeping_path = os.path.join(self._locks_dir, f"{name}.lock")
        self._presence_path = os.path.join(self._locks_dir, f"{name}.active")
        self._presence: IO[str] | None = None

    @contextmanager
    def bookkeeping(self) -> Iterator[None]:
        """Hold the exclusive bookkeeping lock for the duration of the block."""
        os.makedirs(self._locks_dir, exist_ok=True)
        with open(self._bookkeeping_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def join(self) -> None:
        """Count this process as running until ``leave``, or until it exits."""
        if self._presence is None:
            self._presence = open(self._presence_path, "a")
            fcntl.flock(self._presence, fcntl.LOCK_SH)

    def leave(self) -> None:
        """Stop counting this process as running; a no-op when it is not."""
        if self._presence is not None:
            self._presence.close()
            self._presence = None

    def others_running(self) -> bool:
        """True while another process has joined and not left.

        Meaningful under ``bookkeeping`` only, where nobody can join or leave meanwhile,
        and for a process that is not running itself: its own presence would count.
        """
        os.makedirs(self._locks_dir, exist_ok=True)
        with open(self._presence_path, "a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False
