# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Policy of the worker pool, kept free of I/O: which apps get a warm worker, in which order, and when there is room."""

from collections.abc import Iterable

ALL_APPS = "all"


def select_apps_to_warm(available: Iterable[str], selection: str = ALL_APPS) -> list[str]:
    """The apps that keep a warm worker.

    Args:
        available (Iterable[str]): names of the app folders found.
        selection (str): "all", or a comma-separated list of app names (unknown names are ignored).

    Returns:
        list[str]: the selected apps, in the order of `available`.
    """
    names = list(available)
    if selection.strip().lower() in ("", ALL_APPS):
        return names
    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    return [name for name in names if name in wanted]


def warm_order(names: Iterable[str], recent: Iterable[str]) -> list[str]:
    """Order in which to warm the apps: the most recently started first, then the others alphabetically."""
    pending = sorted(set(names))
    ordered = [name for name in dict.fromkeys(recent) if name in pending]
    return ordered + [name for name in pending if name not in ordered]


def memory_allows_spawn(mem_available_kb: int | None, reserve_mb: int, expected_kb: int = 0) -> bool:
    """Whether a new worker fits: MemAvailable, minus what the worker is expected to take, stays above the reserve.

    Unknown memory (no /proc/meminfo) never blocks.
    """
    if mem_available_kb is None:
        return True
    return mem_available_kb - expected_kb >= reserve_mb * 1024


def touch_recent(recent: list[str], name: str, keep: int = 32) -> list[str]:
    """Move an app to the front of the most-recently-started list."""
    return ([name] + [other for other in recent if other != name])[:keep]
