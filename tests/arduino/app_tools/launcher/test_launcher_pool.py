# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_tools.launcher import pool


def test_every_app_is_warmed_by_default():
    assert pool.select_apps_to_warm(["a", "b", "c"]) == ["a", "b", "c"]
    assert pool.select_apps_to_warm(["a", "b"], "ALL") == ["a", "b"]
    assert pool.select_apps_to_warm(["a", "b"], "") == ["a", "b"]


def test_a_selection_keeps_only_the_named_apps():
    assert pool.select_apps_to_warm(["a", "b", "c"], "c, a,missing") == ["a", "c"]


def test_recent_apps_are_warmed_first():
    assert pool.warm_order(["c", "a", "b", "d"], ["b", "gone", "d"]) == ["b", "d", "a", "c"]


def test_memory_reserve():
    assert pool.memory_allows_spawn(None, 400), "unknown memory never blocks"
    assert pool.memory_allows_spawn(500 * 1024, 400)
    assert not pool.memory_allows_spawn(399 * 1024, 400)
    assert not pool.memory_allows_spawn(500 * 1024, 400, expected_kb=200 * 1024)


def test_touch_recent_moves_to_the_front_and_bounds_the_list():
    assert pool.touch_recent(["a", "b", "c"], "c") == ["c", "a", "b"]
    assert pool.touch_recent(["a", "b"], "z", keep=2) == ["z", "a"]
