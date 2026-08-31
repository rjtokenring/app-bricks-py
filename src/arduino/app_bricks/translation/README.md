# Language Translation Brick

The `LanguageTranslation` brick provides a completely offline text-to-text machine translation solution for Arduino Apps.
It translates text between a fixed language pair using locally available translation models,
ensuring privacy and low-latency performance without reliance on cloud services.

## Features

- **Offline Operation:** All translations are performed locally, ensuring data privacy and eliminating network dependencies.
- **Selectable Translation Type:** The translation type *is* the model: each model translates one direction (e.g. `opus-en-es`, `opus-es-en`, `opus-en-zh`, `opus-zh-en`). Select it in the constructor, in `brick_config.yaml`, or override it per-app in `app.yaml`.
- **Single or Batch Translation:** `translate()` accepts a string or a list of strings, translates them in a single request, and always returns a `list[str]` preserving input order.

## Prerequisites

Before using the examples shown in the next sections, ensure you have the following:

- Arduino VENTUNO Q
- The selected translation model deployed on the device

## Supported Translation Types

| Model         | Translation        |
| ------------- | ------------------ |
| `opus-en-es`  | English → Spanish  |
| `opus-es-en`  | Spanish → English  |
| `opus-en-zh`  | English → Chinese  |
| `opus-zh-en`  | Chinese → English  |

## Code Example and Usage

This example shows how to translate a text into the target language of the configured model. `translate()` always returns a list, one entry per input text.

```python
from arduino.app_bricks.translation import LanguageTranslation
from arduino.app_utils import App


translation = LanguageTranslation()  # defaults to opus-en-es


def runner():
    print(translation.translate("Hello world, Arduino!")[0])
    raise StopIteration

App.run(user_loop=runner)
```

### Selecting the Translation Type

Select the model from the brick's configuration tab, or force it in the constructor:

```python
from arduino.app_bricks.translation import LanguageTranslation

translation = LanguageTranslation(model="opus-zh-en")
print(translation.translate("你好，世界。")[0])
```

The model must be downloaded and available on the board.

### Batch Translation

Passing a list to `translate()` sends every text in a single request. The returned list has the same length and order as the input, and empty or blank entries come back as empty strings.

```python
from arduino.app_bricks.translation import LanguageTranslation

translation = LanguageTranslation(model="opus-en-es")

for translated in translation.translate(["Hello world", "How are you?", "Good morning"]):
    print(translated)
```

## Configuration

`LanguageTranslation(model=None, source_language=None, target_language=None, parameters=None, timeout=60)`

- `model`: the translation model, which also selects the translation direction. When omitted, the model configured for this brick is used, falling back to the brick default `opus-en-es`.
- `source_language` and `target_language`: ISO 639-1 codes (e.g. `"en"`, `"es"`). When omitted, they are taken from the model as reported by the translation service, or derived from its `<family>-<source>-<target>` name (e.g. `opus-en-es` → `en`, `es`). Pass them explicitly when the model carries no language pair, otherwise the constructor raises `ValueError`.
- `parameters`: extra model parameters forwarded verbatim to the translation service on every request. Keys the service does not know are ignored.
- `timeout`: how long to wait, in seconds, for a response from the translation service, both when listing the available models at construction and on every translation request.

The resolved values are exposed as the read-only `model`, `source_language`, and `target_language` properties.

## Errors

- `TranslationUnavailableError`: raised when the translation service cannot be reached, either while listing the available models at construction or while translating. Fix by checking that the `arduino:genie_audio` service is running.
- `TranslationModelNotAvailableError`: raised at construction if the configured model is not offered by the translation service. The message lists the available models. Fix by deploying the model or selecting one of the available ones.
- `TranslationRequestError`: raised when the translation service rejects a request or returns an unusable payload.
- `TranslationError`: base class for all of the above, deriving from `AppError`. When one of these reaches the top level uncaught, the app prints the message and a hint on how to fix the problem, followed by the traceback.
