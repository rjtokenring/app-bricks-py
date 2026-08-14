# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import re
import threading
import time

import requests

from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import brick, AppError, Logger

logger = Logger("LanguageTranslation")

# ISO 639-1 language code, as used in the `<family>-<source>-<target>` translation model names.
_LANGUAGE_CODE_RE = re.compile(r"^[a-z]{2}$")

_WARMUP_TEXT = "ok"

_SERVICE_HINT = "Check that the 'arduino:genie_audio' service is running and reachable or restart the app."
_PAYLOAD_HINT = "The translation service returned an unexpected payload."


class TranslationError(AppError):
    """Base class for translation errors."""


class TranslationUnavailableError(TranslationError):
    """Raised when the translation service cannot be reached."""


class TranslationModelNotAvailableError(TranslationError):
    """Raised when the configured model is not offered by the translation service."""


class TranslationRequestError(TranslationError):
    """Raised when the translation service rejects a request or returns an unusable payload."""


@brick
class LanguageTranslation:
    """Language translation brick for offline text-to-text machine translation."""

    _APP_SERVICE_NAME = "audio-analytics-runner"
    _API_PORT = 8085
    _CONTROL_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        model: str | None = None,
        source_language: str | None = None,
        target_language: str | None = None,
        parameters: dict | None = None,
        timeout: int = 60,
    ):
        """Initialize the LanguageTranslation brick.

        Args:
            model (str, optional): Translation model to use. The model also selects the translation direction
                (e.g. `opus-en-es`, `opus-es-en`, `opus-en-zh`, `opus-zh-en`). When omitted, the model is read
                from this brick's `model:` entry in `app.yaml`, falling back to the brick default.
            source_language (str, optional): Source language code (e.g. `"en"`). When omitted, it is derived from
                the model name.
            target_language (str, optional): Target language code (e.g. `"es"`). When omitted, it is derived from
                the model name.
            parameters (dict, optional): Extra model parameters forwarded verbatim to the translation service on
                every request. Keys unknown to the service are ignored.
            timeout (int): Maximum time in seconds to wait for a translation response. Default: 60.

        Raises:
            RuntimeError: If the service address cannot be resolved, or if no model is configured.
            ValueError: If the language pair cannot be derived from the model name and was not provided.
            TranslationUnavailableError: If the translation service cannot be reached.
            TranslationModelNotAvailableError: If the configured model is not available on the translation service.
        """
        # API configuration
        self.api_host = resolve_address(self._APP_SERVICE_NAME)
        if not self.api_host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self.api_port = self._API_PORT
        self.api_base_url = f"http://{self.api_host}:{self.api_port}/audio-analytics/v1/api"

        logger.debug(f"Initialized LanguageTranslation with API base URL: {self.api_base_url}")

        # Resolve the model: the explicit argument wins over the app.yaml override (per-brick `model:`),
        # which in turn wins over the brick default.
        model_name = model
        if not model_name:
            brick_config = get_brick_config(self.__class__) or {}
            brick_id = brick_config.get("id")
            override = get_brick_configured_model(brick_id) if brick_id else None
            model_name = override or brick_config.get("model")
        if not model_name:
            raise RuntimeError("No translation model configured for the LanguageTranslation brick.")

        self._timeout = timeout
        self._parameters = dict(parameters) if parameters else None

        self._model, entry_source, entry_target = self._resolve_model(model_name)
        self._source_language = source_language or entry_source
        self._target_language = target_language or entry_target
        if not self._source_language or not self._target_language:
            raise ValueError(
                f"Could not derive the language pair from translation model '{self._model}'. Pass source_language and target_language explicitly."
            )

        logger.debug(f"Using translation model '{self._model}' ({self._source_language} -> {self._target_language}).")

        # The service translates one request at a time; concurrent callers queue on this lock instead of
        # being rejected, since there is no long-lived session to conflict with.
        self._request_lock = threading.Lock()

    @property
    def model(self) -> str:
        """Translation model in use, as reported by the translation service.

        Returns:
            str: The resolved model name.
        """
        return self._model

    @property
    def source_language(self) -> str:
        """Source language code used for translation requests.

        Returns:
            str: The source language code (e.g. `"en"`).
        """
        return self._source_language

    @property
    def target_language(self) -> str:
        """Target language code used for translation requests.

        Returns:
            str: The target language code (e.g. `"es"`).
        """
        return self._target_language

    def start(self):
        """Start the LanguageTranslation brick, warming up the model on the translation service."""
        self._warmup()

    def stop(self):
        """Stop the LanguageTranslation brick, releasing the translation session on the translation service."""
        self._close_remote_session()

    def translate(self, text: str | list[str]) -> list[str]:
        """
        Translate one or more texts into the target language with a single request.

        Args:
            text (str | list[str]): The text to translate, either a single string or a list of strings. Empty or
                blank entries are not sent to the translation service and are returned as empty strings.

        Returns:
            list[str]: The translated texts, in the same order and with the same length as the input. A single
                string yields a one-item list.

        Raises:
            ValueError: If `text` is not a string or a list of strings.
            TranslationUnavailableError: If the translation service cannot be reached.
            TranslationRequestError: If the translation service rejects the request or returns an unusable payload.
        """
        if isinstance(text, str):
            items = [text]
        else:
            try:
                items = list(text)
            except TypeError:
                raise ValueError(f"text must be a string or a list of strings, got {type(text).__name__}.") from None
            for item in items:
                if not isinstance(item, str):
                    raise ValueError(f"All items in text must be strings, got {type(item).__name__}.")

        # Blank entries are never sent, but they are re-inserted at their original position so the returned
        # list always matches the input length and order.
        indexes = [index for index, item in enumerate(items) if item.strip()]
        if not indexes:
            return ["" for _ in items]

        translated = self._translate([items[index] for index in indexes])

        results = ["" for _ in items]
        for position, index in enumerate(indexes):
            results[index] = translated[position]
        return results

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Catalog ids and runner-reported names differ only by separators
        (e.g. catalog `opus-en-es` vs runner `opus_en_es`), so compare them
        stripped of `-` and `_`."""
        return name.replace("-", "").replace("_", "").lower()

    @staticmethod
    def _parse_language_pair(name: str) -> tuple[str | None, str | None]:
        """Derive the language pair from a `<family>-<source>-<target>` model name, e.g. `opus-en-es` or `opus_zh_en`.

        Args:
            name (str): The model name to parse.

        Returns:
            tuple[str | None, str | None]: The source and target language codes, or `(None, None)` when the name
                does not encode a language pair.
        """
        tokens = [token for token in re.split(r"[-_]", name.lower()) if token]
        if len(tokens) < 3:
            return None, None

        source, target = tokens[-2], tokens[-1]
        if not _LANGUAGE_CODE_RE.match(source) or not _LANGUAGE_CODE_RE.match(target):
            return None, None
        return source, target

    def _resolve_model(self, model_name: str) -> tuple[str, str | None, str | None]:
        """Fetch the models offered by the translation service and resolve `model_name` against them.

        Args:
            model_name (str): The configured model name, either a catalog id or a runner-reported name.

        Returns:
            tuple[str, str | None, str | None]: The runner-reported model name and its source and target languages.

        Raises:
            TranslationUnavailableError: If the model list cannot be fetched.
            TranslationModelNotAvailableError: If the model is not offered by the translation service.
        """
        try:
            response = requests.get(f"{self.api_base_url}/translations/models", timeout=self._CONTROL_TIMEOUT_SECONDS)
        except Exception as e:
            raise TranslationUnavailableError(f"Failed to fetch translation models: {e}.", hint=_SERVICE_HINT) from None

        if response.status_code != 200:
            raise TranslationUnavailableError(self._error_message(response, "Failed to fetch translation models."), hint=_SERVICE_HINT)

        try:
            entries = response.json() or []
        except Exception as e:
            raise TranslationUnavailableError(f"Invalid translation models response: {e}.", hint=_SERVICE_HINT) from None

        wanted = self._normalize_model_name(model_name)
        available: list[str] = []
        for entry in entries:
            # The service reports the model either under `name` or under `id`, depending on the model family.
            if isinstance(entry, dict):
                entry_name = entry.get("name") or entry.get("id")
            elif isinstance(entry, str):
                entry_name = entry
            else:
                continue
            if not entry_name:
                continue

            available.append(entry_name)
            if self._normalize_model_name(entry_name) != wanted:
                continue

            source = entry.get("source_language") if isinstance(entry, dict) else None
            target = entry.get("target_language") if isinstance(entry, dict) else None
            if not source or not target:
                source, target = self._parse_language_pair(entry_name)
            return entry_name, source, target

        raise TranslationModelNotAvailableError(
            f"Translation model '{model_name}' is not available on the runner. Available models: {', '.join(available) or 'none'}.",
            hint="Download the model on the device or select a model already available on the device.",
        )

    def _translate(self, texts: list[str]) -> list[str]:
        payload = {
            "text": texts,
            "model": self._model,
            "source_language": self._source_language,
            "target_language": self._target_language,
        }
        if self._parameters:
            payload["parameters"] = self._parameters

        url = f"{self.api_base_url}/translations/translate"
        started_at = time.perf_counter()
        with self._request_lock:
            try:
                response = requests.post(url, json=payload, timeout=self._timeout)
            except Exception as e:
                raise TranslationUnavailableError(f"Failed to reach the translation service: {e}.", hint=_SERVICE_HINT) from None

        if response.status_code != 200:
            raise TranslationRequestError(
                self._error_message(response, "Failed to translate text."),
                hint=f"Check that the text and the model parameters are valid for model '{self._model}'.",
            )

        try:
            data = response.json()
        except Exception as e:
            raise TranslationRequestError(f"Invalid translation response: {e}.", hint=_SERVICE_HINT) from None

        translations = data.get("translations") if isinstance(data, dict) else None
        if not isinstance(translations, list):
            raise TranslationRequestError("No 'translations' returned from the translation API.", hint=_PAYLOAD_HINT)
        # The response carries no identifiers, so results are correlated by position: a count mismatch means
        # we cannot tell which translation belongs to which input.
        if len(translations) != len(texts):
            raise TranslationRequestError(f"Translation API returned {len(translations)} results for {len(texts)} inputs.", hint=_PAYLOAD_HINT)

        results = [self._extract_translated_text(entry) for entry in translations]

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            f"Translation completed in {elapsed_ms:.2f} ms (model={self._model}, items={len(texts)}, input_chars={sum(len(text) for text in texts)})"
        )
        return results

    @staticmethod
    def _extract_translated_text(entry) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for key in ("translated_text", "text", "translation"):
                value = entry.get(key)
                if isinstance(value, str):
                    return value

        raise TranslationRequestError(f"Unexpected translation entry in response: {entry!r}", hint=_PAYLOAD_HINT)

    @staticmethod
    def _error_message(response, fallback: str) -> str:
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "error" in error_data:
                return error_data["error"].get("message", fallback)
        except Exception:
            pass
        return f"{fallback} (status_code={response.status_code})"

    def _warmup(self) -> None:
        """Best-effort warmup: translate a short text so the inference container loads
        the translation model before the first real translate()."""
        started_at = time.perf_counter()
        try:
            self._translate([_WARMUP_TEXT])
        except Exception as e:
            logger.warning(f"Translation warmup failed: {e}")
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(f"Translation warmup completed in {elapsed_ms:.2f} ms")

    def _close_remote_session(self) -> None:
        try:
            response = requests.post(f"{self.api_base_url}/translations/close", timeout=self._CONTROL_TIMEOUT_SECONDS)
            if response.status_code >= 400:
                logger.warning(f"Failed to close remote translation session: status_code={response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to close remote translation session: {e}")
