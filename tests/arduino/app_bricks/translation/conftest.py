# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_bricks.translation import LanguageTranslation
from arduino.app_utils import App

# The configured model uses dashes while the fake runner below reports underscores, so every test
# exercises the separator normalization the real service requires.
CONFIGURED_MODEL = "opus-en-es"
RUNNER_MODEL = "opus_en_es"

MODELS = [{"name": RUNNER_MODEL}]


@pytest.fixture(autouse=True)
def _patch_brick_lookup(monkeypatch: pytest.MonkeyPatch):
    """Avoid hitting the real service-discovery.

    This only bypasses discovery and model configuration: ``LanguageTranslation.__init__`` also lists the models
    offered by the runner, so ``requests`` must still be patched by each test (see ``make_translation``).
    """
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr(
        "arduino.app_bricks.translation.local_translation.get_brick_config",
        lambda cls: {"id": "arduino:translation", "model": CONFIGURED_MODEL},
    )
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.get_brick_configured_model", lambda _id: None)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, json_error=None):
        self.status_code = status_code
        self._json_data = json_data
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


def make_translation(monkeypatch, models=None, post=None, **kwargs):
    """Build a LanguageTranslation brick against a fake runner.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        models (list, optional): Entries returned by ``GET /translations/models``. Defaults to ``MODELS``.
        post (callable, optional): Replacement for ``requests.post``, called as ``post(url, json, **kwargs)``.
        **kwargs: Forwarded to the ``LanguageTranslation`` constructor.

    Returns:
        tuple[LanguageTranslation, list[dict]]: The brick, and the list of recorded POST calls.
    """
    calls: list[dict] = []

    def fake_get(url, **get_kwargs):
        calls.append({"method": "GET", "url": url, "kwargs": get_kwargs})
        return FakeResponse(json_data=MODELS if models is None else models)

    def fake_post(url, json=None, **post_kwargs):
        calls.append({"method": "POST", "url": url, "json": json, "kwargs": post_kwargs})
        if post is not None:
            return post(url, json, **post_kwargs)
        return FakeResponse(json_data={"translations": [{"translated_text": "translated"}] * len((json or {}).get("text", []))})

    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.requests.get", fake_get)
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.requests.post", fake_post)

    translation = LanguageTranslation(**kwargs)
    App.unregister(translation)
    calls.clear()  # drop the model listing performed during construction
    return translation, calls


def posts(calls: list[dict]) -> list[dict]:
    """Return only the recorded POST calls.

    Args:
        calls (list[dict]): The recorded calls.

    Returns:
        list[dict]: The POST calls, in order.
    """
    return [call for call in calls if call["method"] == "POST"]
