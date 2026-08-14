# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading

import pytest
from conftest import CONFIGURED_MODEL, RUNNER_MODEL, FakeResponse, make_translation, posts

from arduino.app_bricks.translation import (
    LanguageTranslation,
    TranslationError,
    TranslationModelNotAvailableError,
    TranslationRequestError,
    TranslationUnavailableError,
)
from arduino.app_utils import App, AppError


def _translations(*texts):
    return FakeResponse(json_data={"translations": [{"translated_text": text} for text in texts]})


def _patch_models(monkeypatch, response=None, raises=None):
    """Patch only the model listing, for tests that must fail during construction."""

    def fake_get(url, **kwargs):
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.requests.get", fake_get)


# --------------------------------------------------------------------------------------------------
# Model and language resolution
# --------------------------------------------------------------------------------------------------


def test_resolve_model_matches_runner_name_with_different_separators(monkeypatch):
    translation, _ = make_translation(monkeypatch)

    # The configured catalog id `opus-en-es` resolves to the runner-reported `opus_en_es`,
    # which is what translation requests must send.
    assert translation.model == RUNNER_MODEL
    assert translation.source_language == "en"
    assert translation.target_language == "es"


def test_resolve_model_prefers_entry_language_metadata(monkeypatch):
    models = [{"name": RUNNER_MODEL, "source_language": "eng", "target_language": "spa"}]
    translation, _ = make_translation(monkeypatch, models=models)

    assert translation.source_language == "eng"
    assert translation.target_language == "spa"


def test_resolve_model_accepts_entry_id_key(monkeypatch):
    translation, _ = make_translation(monkeypatch, models=[{"id": RUNNER_MODEL}])

    assert translation.model == RUNNER_MODEL


def test_resolve_model_accepts_bare_string_entry(monkeypatch):
    translation, _ = make_translation(monkeypatch, models=[RUNNER_MODEL])

    assert translation.model == RUNNER_MODEL


def test_constructor_raises_when_model_not_offered(monkeypatch):
    _patch_models(monkeypatch, response=FakeResponse(json_data=[{"name": "opus_zh_en"}]))

    with pytest.raises(TranslationModelNotAvailableError) as excinfo:
        LanguageTranslation()

    assert CONFIGURED_MODEL in str(excinfo.value)
    assert "opus_zh_en" in str(excinfo.value)  # the message lists what is available


def test_constructor_raises_when_models_listing_unreachable(monkeypatch):
    _patch_models(monkeypatch, raises=OSError("connection refused"))

    with pytest.raises(TranslationUnavailableError):
        LanguageTranslation()


def test_constructor_raises_on_models_listing_error_status(monkeypatch):
    _patch_models(monkeypatch, response=FakeResponse(status_code=500, json_data={"error": {"message": "engine down"}}))

    with pytest.raises(TranslationUnavailableError, match="engine down"):
        LanguageTranslation()


def test_explicit_model_wins_over_app_config_and_brick_default(monkeypatch):
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.get_brick_configured_model", lambda _id: "opus-zh-en")

    translation, _ = make_translation(monkeypatch, models=[{"name": "opus_en_zh"}, {"name": "opus_zh_en"}], model="opus-en-zh")

    assert translation.model == "opus_en_zh"
    assert (translation.source_language, translation.target_language) == ("en", "zh")


def test_app_configured_model_wins_over_brick_default(monkeypatch):
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.get_brick_configured_model", lambda _id: "opus-zh-en")

    translation, _ = make_translation(monkeypatch, models=[{"name": RUNNER_MODEL}, {"name": "opus_zh_en"}])

    assert translation.model == "opus_zh_en"
    assert (translation.source_language, translation.target_language) == ("zh", "en")


def test_constructor_raises_when_no_model_configured(monkeypatch):
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.get_brick_config", lambda cls: {})

    with pytest.raises(RuntimeError, match="No translation model configured"):
        LanguageTranslation()


def test_constructor_raises_when_language_pair_undecidable(monkeypatch):
    _patch_models(monkeypatch, response=FakeResponse(json_data=[{"name": "opusenes"}]))

    with pytest.raises(ValueError, match="Could not derive the language pair"):
        LanguageTranslation()


def test_explicit_languages_override_derived_pair(monkeypatch):
    translation, _ = make_translation(monkeypatch, source_language="pt", target_language="en")

    assert (translation.source_language, translation.target_language) == ("pt", "en")


def test_constructor_raises_when_address_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.resolve_address", lambda host: "")

    with pytest.raises(RuntimeError, match="Host address could not be resolved"):
        LanguageTranslation()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("opus-en-es", ("en", "es")),
        ("opus_zh_en", ("zh", "en")),
        ("opus_mt_en_es", ("en", "es")),
        ("melo-tts-en", (None, None)),
        ("pipertts_en", (None, None)),
        ("opus", (None, None)),
    ],
)
def test_parse_language_pair(name, expected):
    assert LanguageTranslation._parse_language_pair(name) == expected


# --------------------------------------------------------------------------------------------------
# translate
# --------------------------------------------------------------------------------------------------


def test_translate_posts_expected_payload_and_returns_text(monkeypatch):
    translation, calls = make_translation(monkeypatch, post=lambda url, json, **kwargs: _translations("Hola mundo"))

    assert translation.translate("Hello world") == ["Hola mundo"]

    call = posts(calls)[0]
    assert call["url"] == "http://127.0.0.1:8085/audio-analytics/v1/api/translations/translate"
    assert call["json"] == {
        "text": ["Hello world"],
        "model": RUNNER_MODEL,
        "source_language": "en",
        "target_language": "es",
    }
    assert "parameters" not in call["json"]
    assert call["kwargs"]["timeout"] == 60


def test_translate_uses_configured_timeout(monkeypatch):
    translation, calls = make_translation(monkeypatch, timeout=5)

    translation.translate("Hello world")

    assert posts(calls)[0]["kwargs"]["timeout"] == 5


def test_constructor_parameters_included_in_payload(monkeypatch):
    translation, calls = make_translation(monkeypatch, parameters={"dummy_param": "dummy_value"})

    translation.translate("Hello world")

    assert posts(calls)[0]["json"]["parameters"] == {"dummy_param": "dummy_value"}


def test_translate_preserves_order_in_a_single_request(monkeypatch):
    translation, calls = make_translation(
        monkeypatch,
        post=lambda url, json, **kwargs: _translations("Hola mundo", "¿Cómo estás?", "Buenos días"),
    )

    assert translation.translate(["Hello world", "How are you?", "Good morning"]) == [
        "Hola mundo",
        "¿Cómo estás?",
        "Buenos días",
    ]
    assert len(posts(calls)) == 1


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_translate_returns_one_empty_string_for_blank_input(monkeypatch, text):
    translation, calls = make_translation(monkeypatch)

    assert translation.translate(text) == [""]
    assert posts(calls) == []


def test_translate_returns_empty_list_for_empty_list_input(monkeypatch):
    translation, calls = make_translation(monkeypatch)

    assert translation.translate([]) == []
    assert posts(calls) == []


def test_translate_skips_blank_entries_but_keeps_positions(monkeypatch):
    translation, calls = make_translation(monkeypatch, post=lambda url, json, **kwargs: _translations("Hola", "Adiós"))

    assert translation.translate(["Hi", "  ", "Bye"]) == ["Hola", "", "Adiós"]
    assert posts(calls)[0]["json"]["text"] == ["Hi", "Bye"]


def test_translate_returns_empty_strings_when_all_entries_blank(monkeypatch):
    translation, calls = make_translation(monkeypatch)

    assert translation.translate(["", "  "]) == ["", ""]
    assert posts(calls) == []


def test_translate_raises_on_error_status(monkeypatch):
    translation, _ = make_translation(
        monkeypatch,
        post=lambda url, json, **kwargs: FakeResponse(status_code=400, json_data={"error": {"message": "bad model"}}),
    )

    with pytest.raises(TranslationRequestError, match="bad model"):
        translation.translate("Hello world")


def test_translate_raises_on_error_status_without_error_body(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(status_code=503, json_data={}))

    with pytest.raises(TranslationRequestError, match="status_code=503"):
        translation.translate("Hello world")


def test_translate_raises_when_service_unreachable(monkeypatch):
    def post(url, json, **kwargs):
        raise OSError("connection reset")

    translation, _ = make_translation(monkeypatch, post=post)

    with pytest.raises(TranslationUnavailableError):
        translation.translate("Hello world")


def test_translate_raises_when_translations_key_missing(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(json_data={}))

    with pytest.raises(TranslationRequestError, match="No 'translations' returned"):
        translation.translate("Hello world")


def test_translate_raises_on_result_count_mismatch(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: _translations("Hola"))

    with pytest.raises(TranslationRequestError, match="1 results for 2 inputs"):
        translation.translate(["Hello world", "Good morning"])


@pytest.mark.parametrize("entry", [{"translated_text": "Hola mundo"}, {"text": "Hola mundo"}, {"translation": "Hola mundo"}, "Hola mundo"])
def test_translate_accepts_alternative_entry_shapes(monkeypatch, entry):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(json_data={"translations": [entry]}))

    assert translation.translate("Hello world") == ["Hola mundo"]


def test_translate_raises_on_unexpected_entry_shape(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(json_data={"translations": [{"foo": 1}]}))

    with pytest.raises(TranslationRequestError, match="Unexpected translation entry"):
        translation.translate("Hello world")


def test_translate_raises_on_invalid_json_response(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(json_error=ValueError("not json")))

    with pytest.raises(TranslationRequestError, match="Invalid translation response"):
        translation.translate("Hello world")


def test_translate_rejects_non_string_input(monkeypatch):
    translation, _ = make_translation(monkeypatch)

    with pytest.raises(ValueError, match="text must be a string or a list of strings"):
        translation.translate(42)


def test_translate_rejects_non_string_items(monkeypatch):
    translation, _ = make_translation(monkeypatch)

    with pytest.raises(ValueError, match="All items in text must be strings"):
        translation.translate(["Hello world", 1])


# --------------------------------------------------------------------------------------------------
# Error reporting
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_class",
    [TranslationError, TranslationUnavailableError, TranslationModelNotAvailableError, TranslationRequestError],
)
def test_translation_errors_are_app_errors(error_class):
    # AppError subclasses are reported by the app excepthook with their message and hint,
    # instead of a bare traceback.
    assert issubclass(error_class, AppError)


def test_unreachable_service_error_hints_at_the_audio_service(monkeypatch):
    _patch_models(monkeypatch, raises=OSError("connection refused"))

    with pytest.raises(TranslationUnavailableError) as excinfo:
        LanguageTranslation()

    assert "arduino:genie_audio" in excinfo.value.hint


def test_model_not_available_error_hints_at_deploying_or_selecting_a_model(monkeypatch):
    _patch_models(monkeypatch, response=FakeResponse(json_data=[{"name": "opus_zh_en"}]))

    with pytest.raises(TranslationModelNotAvailableError) as excinfo:
        LanguageTranslation()

    assert "Deploy the model" in excinfo.value.hint


def test_translate_unreachable_error_hints_at_the_audio_service(monkeypatch):
    def post(url, json, **kwargs):
        raise OSError("connection reset")

    translation, _ = make_translation(monkeypatch, post=post)

    with pytest.raises(TranslationUnavailableError) as excinfo:
        translation.translate("Hello world")

    assert "arduino:genie_audio" in excinfo.value.hint


def test_unusable_payload_error_carries_a_hint(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(json_data={}))

    with pytest.raises(TranslationRequestError) as excinfo:
        translation.translate("Hello world")

    assert "unexpected payload" in excinfo.value.hint


def test_rejected_request_error_hint_mentions_the_model(monkeypatch):
    translation, _ = make_translation(
        monkeypatch,
        post=lambda url, json, **kwargs: FakeResponse(status_code=400, json_data={"error": {"message": "bad model"}}),
    )

    with pytest.raises(TranslationRequestError) as excinfo:
        translation.translate("Hello world")

    assert RUNNER_MODEL in excinfo.value.hint


# --------------------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------------------


def test_start_warms_up_the_model(monkeypatch):
    translation, calls = make_translation(monkeypatch)

    translation.start()

    warmup = posts(calls)
    assert len(warmup) == 1
    assert warmup[0]["url"].endswith("/translations/translate")
    assert warmup[0]["json"]["text"] == ["ok"]


def test_start_does_not_raise_when_warmup_fails(monkeypatch):
    def post(url, json, **kwargs):
        raise OSError("connection refused")

    translation, _ = make_translation(monkeypatch, post=post)

    translation.start()  # best-effort: a failed warmup must not break App.run()


def test_stop_closes_remote_session(monkeypatch):
    translation, calls = make_translation(monkeypatch)

    translation.stop()

    call = posts(calls)[0]
    assert call["url"] == "http://127.0.0.1:8085/audio-analytics/v1/api/translations/close"
    assert call["json"] is None


def test_stop_does_not_raise_when_close_fails(monkeypatch):
    translation, _ = make_translation(monkeypatch, post=lambda url, json, **kwargs: FakeResponse(status_code=500))

    translation.stop()


def test_stop_does_not_raise_when_close_is_unreachable(monkeypatch):
    def post(url, json, **kwargs):
        raise OSError("connection refused")

    translation, _ = make_translation(monkeypatch, post=post)

    translation.stop()


# --------------------------------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------------------------------


def test_concurrent_translations_are_serialized(monkeypatch):
    in_flight = 0
    overlapped = False
    first_request_started = threading.Event()
    release_first_request = threading.Event()
    state_lock = threading.Lock()

    def post(url, json, **kwargs):
        nonlocal in_flight, overlapped
        with state_lock:
            in_flight += 1
            if in_flight > 1:
                overlapped = True
        if not first_request_started.is_set():
            first_request_started.set()
            release_first_request.wait(timeout=2)
        with state_lock:
            in_flight -= 1
        return _translations("Hola mundo")

    translation, _ = make_translation(monkeypatch, post=post)

    results: list[list[str]] = []
    errors: list[Exception] = []

    def run():
        try:
            results.append(translation.translate("Hello world"))
        except Exception as e:  # pragma: no cover - would mean the lock rejected a caller
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(2)]
    threads[0].start()
    assert first_request_started.wait(timeout=2)
    threads[1].start()
    release_first_request.set()
    for thread in threads:
        thread.join(timeout=5)

    # Requests queue on the lock: both callers succeed, and neither gets a "busy" error.
    assert errors == []
    assert results == [["Hola mundo"], ["Hola mundo"]]
    assert not overlapped


def test_brick_instance_is_registered_with_the_app(monkeypatch):
    def fake_get(url, **kwargs):
        return FakeResponse(json_data=[{"name": RUNNER_MODEL}])

    monkeypatch.setattr("arduino.app_bricks.translation.local_translation.requests.get", fake_get)

    translation = LanguageTranslation()
    try:
        assert translation in App._waiting_queue
    finally:
        App.unregister(translation)
