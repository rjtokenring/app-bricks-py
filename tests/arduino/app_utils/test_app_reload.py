# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The library side of an app reload in the same process: after the reset, main.py finds the library as at its start."""

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

import arduino.app_utils.app as app
from arduino.app_utils import AppController, peripheral_registry
from arduino.app_utils import bridge as bridge_module
from arduino.app_utils._reload import reset
from arduino.app_utils.peripheral_registry import PeripheralRegistry


class Brick:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()

    def start(self) -> None:
        self.started.set()

    def stop(self) -> None:
        self.stopped.set()

    def loop(self) -> None:
        self.stopped.wait(0.01)


class Device:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> AppController:
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    monkeypatch.setattr("arduino.app_utils._reload.App", instance)
    return instance


@pytest.fixture
def peripherals(monkeypatch: pytest.MonkeyPatch) -> PeripheralRegistry:
    registry = PeripheralRegistry()
    monkeypatch.setattr(peripheral_registry, "Peripherals", registry)
    return registry


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    fake = MagicMock()
    monkeypatch.setattr(bridge_module, "_bridge", fake)
    monkeypatch.setattr(bridge_module, "_provided", set())
    yield fake


def test_a_running_brick_is_stopped_and_forgotten(controller: AppController, peripherals: PeripheralRegistry):
    brick = Brick()
    device = Device()
    peripherals.register(device)
    controller.start_brick(brick)
    assert brick.started.is_set()

    assert controller._reset_for_reload() == []  # pyright: ignore[reportPrivateUsage]

    assert brick.stopped.is_set()
    assert device.stops == 1, "the peripherals of the ended run are released"
    assert list(peripherals._peripherals) == [], "and forgotten"  # pyright: ignore[reportPrivateUsage]


def test_the_next_run_registers_and_starts_its_bricks(controller: AppController, peripherals: PeripheralRegistry):
    controller.register(Brick())
    controller.start_bricks()
    controller._shutdown()  # pyright: ignore[reportPrivateUsage]  # What App.run() does when the reload interrupts it
    controller._reset_for_reload()  # pyright: ignore[reportPrivateUsage]

    # The new main.py registers at its top level, before App.run(): a controller left "stopping" would refuse it
    second = Brick()
    controller.register(second)
    controller.start_bricks()
    assert second.started.is_set()
    controller.stop_bricks()


def test_bricks_registered_but_never_started_are_dropped(controller: AppController, peripherals: PeripheralRegistry):
    first = Brick()
    controller.register(first)  # The reload came before App.run()

    controller._reset_for_reload()  # pyright: ignore[reportPrivateUsage]
    controller.start_bricks()

    assert not first.started.is_set(), "a brick of the ended run never starts in the next one"


def test_provided_methods_are_withdrawn(router: MagicMock):
    bridge_module.Bridge.provide("by_call", lambda: None)

    @bridge_module.provide()
    def by_decorator() -> None: ...

    assert bridge_module.unprovide_all() == []
    assert sorted(call.args[0] for call in router.unprovide.call_args_list) == ["by_call", "by_decorator"]
    assert bridge_module.unprovide_all() == [], "withdrawn once"
    assert router.unprovide.call_count == 2


def test_a_method_unprovided_by_the_app_is_not_withdrawn_again(router: MagicMock):
    bridge_module.Bridge.provide("gone", lambda: None)
    bridge_module.Bridge.unprovide("gone")
    router.unprovide.reset_mock()
    assert bridge_module.unprovide_all() == []
    router.unprovide.assert_not_called()


def test_a_method_that_cannot_be_withdrawn_is_reported(router: MagicMock):
    router.unprovide.side_effect = ConnectionError("router gone")
    bridge_module.Bridge.provide("stuck", lambda: None)
    problems = bridge_module.unprovide_all()
    assert len(problems) == 1 and "stuck" in problems[0]


def test_reset_covers_the_app_and_the_bridge(controller: AppController, peripherals: PeripheralRegistry, router: MagicMock):
    brick = Brick()
    controller.start_brick(brick)
    bridge_module.Bridge.provide("m", lambda: None)

    assert reset() == []
    assert brick.stopped.is_set()
    router.unprovide.assert_called_once_with("m")
