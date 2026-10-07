# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Whoever stops the app sets the shutdown grace period: arduino-app-cli 5 s, arduino-app-launcher less."""

import pytest

from arduino.app_utils import app


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 5.0), ("", 5.0), ("2.5", 2.5), ("not a number", 5.0), ("0.2", app.MIN_SHUTDOWN_GRACE_PERIOD_S)],
)
def test_the_grace_period_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float):
    if value is None:
        monkeypatch.delenv("APP_SHUTDOWN_GRACE_PERIOD_S", raising=False)
    else:
        monkeypatch.setenv("APP_SHUTDOWN_GRACE_PERIOD_S", value)
    assert app._grace_period_from_env() == expected  # pyright: ignore[reportPrivateUsage]


def test_the_default_budgets_are_unchanged():
    assert app.SHUTDOWN_GRACE_PERIOD_S == 5.0
    assert app.SHUTDOWN_PERIPHERALS_BUDGET_S == 1.5
    assert app.SHUTDOWN_BRICKS_BUDGET_S == 3.0
