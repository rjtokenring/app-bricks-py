# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Puts the library back in the state main.py finds at its start, after a run that ended in this process.

After reset(): no brick registered or running, no peripheral registered, no method provided on the Bridge.
The router connection stays open. Once the objects of the ended run are collected, held_devices() is empty.
"""

import sys

from .app import App
from .bridge import unprovide_all


def reset() -> list[str]:
    """Stops and forgets what the ended run left in the library.

    Returns:
        list[str]: What could not be reset, e.g. a brick still running or a method that could not be
            withdrawn; empty when main.py can run again.
    """
    return App._reset_for_reload() + unprovide_all()  # pyright: ignore[reportPrivateUsage]


def held_devices() -> list[str]:
    """The devices still claimed by a peripheral instance, e.g. a camera something still references."""
    if "arduino.app_peripherals.device_registry" not in sys.modules:
        return []
    from arduino.app_peripherals.device_registry import DeviceRegistry

    return DeviceRegistry.claimed_devices()
