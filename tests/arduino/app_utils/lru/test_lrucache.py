# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import LRUDict


def test_lru_cache():
    cache = LRUDict(maxsize=3)

    # Test setting and getting items
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3

    # Test LRU eviction
    cache["d"] = 4  # This should evict 'a'

    for k, v in cache.items():
        print(f"{k}: {v}")

    assert "a" not in cache
    assert cache["d"] == 4

    # Access 'b' to make it recently used
    _ = cache["b"]
    cache["e"] = 5  # This should evict 'c'
    assert "c" not in cache
    assert cache["b"] == 2
    assert cache["e"] == 5
