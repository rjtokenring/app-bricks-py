# Text To Speech (TTS) Brick

The `TextToSpeech` brick provides a completely offline text-to-speech (TTS) solution for Arduino Apps. It's designed to convert text input into spoken audio using locally available TTS engines, ensuring privacy and low-latency performance without reliance on cloud services.

## Features

- **Offline Operation:** All text-to-speech functions are performed locally, ensuring data privacy and eliminating network dependencies.
- **Multiple Language Support:** Select a language by configuring the corresponding TTS model (e.g. `melo-tts-en`, `melo-tts-es`, `melo-tts-zh`, `piper-tts-de`, `piper-tts-en`, `piper-tts-it`) in `brick_config.yaml` or override per-app in `app.yaml`.
- **Multilingual Kokoro Voices:** The `kokoro-82m` model speaks American and British English, Spanish, French, Hindi, Italian and Brazilian Portuguese with a single model: the language is the one of the voice passed to `TextToSpeech(voice=...)`. It runs on the CPU, in a service of its own that already contains the model.
- **Audio Output Formats:** Directly output synthesized speech to a Speaker instance or to WAV, PCM, or PCM audio.
- **Long Text Support:** `speak()` splits long input into sentence-aware chunks of up to 1024 characters before synthesis.
- **Streaming Playback:** `speak()` plays PCM chunks as they arrive from the local TTS service instead of waiting for the full rendered response.
- **Blocking or Background Playback:** `speak()` blocks until playback completes by default; pass `block=False` to enqueue the text and return immediately. A single background worker plays queued texts sequentially (FIFO), so `speak()` can be called repeatedly without waiting. In background mode synthesis errors are logged instead of raised; `is_speaking()` tells whether speech is still playing or queued, and `wait_until_done()` blocks until it is over. The queue holds up to 128 pending requests by default (configurable via the `max_queue_size` constructor parameter); when full, further non-blocking calls raise `TTSBusyError`.
- **Speaking a Streaming LLM Answer:** `speak_stream()` takes text in pieces, such as the chunks of `chat_stream()`, and queues each sentence as soon as it is complete, so the voice starts after the first sentence instead of after the whole answer. It returns the text it spoke.
- **Cancellable Playback:** Use `cancel()` to stop the current spoken sequence and notify the local TTS service without stopping the TTS brick or speaker.
- **Single-Session Semantics:** Each instance handles one speech session at a time. For concurrent speech, create multiple `TextToSpeech` instances.

## Prerequisites

Before using the TTS Speak Through Speaker example, shown in the next section, ensure you have the following:

- USB-C® Hub with external power supply (5V, 3A)
- USB audio device (USB speaker or USB-C → 3.5mm adapter)
- Arduino VENTUNO Q running in Network Mode or SBC Mode (USB-C port needed for the hub)

## Code Example and Usage

This example shows how to convert text into spoken audio, which will be played on a speaker.

```python
from arduino.app_bricks.tts import TextToSpeech
from arduino.app_utils import App


tts = TextToSpeech()


def runner():
    tts.speak("Hello world, Arduino!")  # Blocks until playback completes


App.run(user_loop=runner)
```

### Non-blocking playback

This example shows how to start speech playback in the background and keep working while it plays.

```python
from arduino.app_bricks.tts import TextToSpeech
from arduino.app_utils import App
import time


tts = TextToSpeech()


def runner():
    tts.speak("This sentence plays in the background.", block=False)
    tts.speak("This one is queued and plays right after.", block=False)
    while tts.is_speaking():
        time.sleep(0.1)  # Do other work while the speech plays


App.run(user_loop=runner)
```

### Speak an LLM answer while it is generated

This example speaks the answer of a local language model sentence by sentence, while the rest of the answer is still being generated.

```python
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_bricks.tts import TextToSpeech
from arduino.app_utils import App


llm = LargeLanguageModel(system_prompt="Answer in two short sentences of plain text.")
tts = TextToSpeech()


def runner():
    spoken = tts.speak_stream(llm.chat_stream("Why is the sky blue?"))  # Blocks until the answer has been spoken
    print(spoken)
    raise StopIteration


App.run(user_loop=runner)
```

Pass `block=False` to return as soon as the answer is generated, while its last sentences still play; `wait_until_done()` waits for them, and `cancel()` stops the speech and the reading of the rest of the answer.

### Save as WAV

This example shows how to convert text into spoken audio and save it as a WAV file.

```python
from arduino.app_bricks.tts import TextToSpeech

tts = TextToSpeech()

wav = tts.synthesize_wav("Hello, Arduino world!")
with open("synthesized_speech.wav", "wb") as f:
    f.write(wav)
```

### Speak with a Kokoro voice

This example speaks Italian with the `kokoro-82m` model. Select the model for the brick in the `app.yaml` of the App:

```yaml
bricks:
  - arduino:tts:
      model: kokoro-82m
```

Then choose the voice in the code. Without `voice`, the model speaks with `af_heart` (American English).

```python
from arduino.app_bricks.tts import TextToSpeech
from arduino.app_utils import App


tts = TextToSpeech(voice="if_sara")


def runner():
    tts.speak("Ciao a tutti, sono la voce di Arduino!")
    raise StopIteration


App.run(user_loop=runner)
```

The first letter of a voice name is its language, the second its gender (`f` female, `m` male):

| Language | Voices |
|---|---|
| American English | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| British English | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` |
| Spanish | `ef_dora`, `em_alex`, `em_santa` |
| French | `ff_siwis` |
| Hindi | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |
| Italian | `if_sara`, `im_nicola` |
| Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa` |

A voice speaks the text in its own language: give an Italian voice Italian text. All the other methods (`speak(block=False)`, `speak_stream()`, `synthesize_wav()`, `cancel()`) work the same with every model.

## Configuration

`TextToSpeech(speaker=None, max_queue_size=128, voice=None)`: pass a `Speaker` instance to control the audio output device (a default shared `Speaker` is used otherwise); `max_queue_size` bounds the number of pending `speak(block=False)` requests; `voice` names the voice of the `kokoro-82m` model and is ignored by the other models, which have one voice each.

The model is the `model` setting of the brick in `app.yaml` (default `pipertts_en`).

## Errors

- `TTSBusyError`: raised if you call `speak()` (blocking mode), `synthesize_pcm()`, `synthesize_pcm_stream()`, or `synthesize_wav()` while the instance already has an active session. Fix by awaiting the current session or using a separate instance. `speak(..., block=False)` raises it only when the speech queue is full: otherwise the text is queued and played when the current session ends.
- `TTSVoiceError`: raised by `TextToSpeech(voice=...)` when the `kokoro-82m` model has no voice with that name. The error hint lists the available voices.
- `TTSError`: base class for all of the above.
