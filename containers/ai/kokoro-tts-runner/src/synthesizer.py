# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Kokoro speech synthesis on ONNX Runtime: phoneme tokens in, 16-bit PCM out."""

import os
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import text as text_frontend

# Sample rate of the audio the model generates
MODEL_SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"


class Session(Protocol):
    """The part of an ONNX Runtime inference session the synthesizer runs."""

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


class UnknownVoiceError(ValueError):
    """The requested voice is not one this runner speaks with."""


def to_pcm16(samples: np.ndarray, sample_rate: int) -> bytes:
    """Mono float samples at the model's rate as 16-bit little-endian PCM at sample_rate."""
    if sample_rate != MODEL_SAMPLE_RATE and samples.size:
        count = round(samples.size * sample_rate / MODEL_SAMPLE_RATE)
        samples = np.interp(np.arange(count) * (MODEL_SAMPLE_RATE / sample_rate), np.arange(samples.size), samples)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Synthesizer:
    """Speaks text with the voices of a Kokoro model, one inference at a time."""

    def __init__(
        self,
        session: Session,
        voices: Mapping[str, np.ndarray],
        vocab: dict[str, int],
        espeak: text_frontend.Espeak = text_frontend.espeak_ipa,
    ) -> None:
        """
        Args:
            session: Inference session of the Kokoro model.
            voices: Style vectors of every voice, one per token count: shape (510, 1, 256).
            vocab: Token id of each phoneme and punctuation mark.
            espeak: Phonemizer of a piece of text without punctuation.
        """
        self._session = session
        self._voices = voices
        self._vocab = vocab
        self._espeak = espeak
        self._lock = threading.Lock()
        # Voice name -> espeak-ng language, only for the voices that have one
        self.voices = {name: language for name in sorted(voices) if (language := text_frontend.voice_language(name))}

    @classmethod
    def load(cls, model_path: Path, voices_dir: Path) -> "Synthesizer":
        """Open the model with as many threads as the cores the container may run on, and every ``<voice>.bin`` of voices_dir."""
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = len(os.sched_getaffinity(0))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        # Raw float32: one style vector of 256 values per sequence length
        voices = {path.stem: np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256) for path in sorted(voices_dir.glob("*.bin"))}
        return cls(session, voices, text_frontend.load_vocab())

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, sample_rate: int = MODEL_SAMPLE_RATE) -> Iterator[bytes]:
        """Yield the speech of each sentence of text as soon as it is generated, as 16-bit mono PCM.

        Raises:
            UnknownVoiceError: If voice is not in ``voices``. Raised by the call, before any audio.
        """
        language = self.voices.get(voice)
        if language is None:
            raise UnknownVoiceError(f"Unknown voice '{voice}'. Available voices: {', '.join(self.voices)}.")
        return self._speech(text, voice, language, speed, sample_rate)

    def _speech(self, text: str, voice: str, language: str, speed: float, sample_rate: int) -> Iterator[bytes]:
        styles = self._voices[voice]
        for sentence in text_frontend.sentences(text):
            phonemes = text_frontend.phonemize(sentence, language, self._espeak)
            for tokens in text_frontend.tokenize(phonemes, self._vocab):
                inputs = {
                    # The model wants the sequence between two pad tokens
                    "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
                    # One style vector per sequence length, the first for a single token
                    "style": styles[len(tokens) - 1].astype(np.float32),
                    "speed": np.array([speed], dtype=np.float32),
                }
                with self._lock:
                    audio = self._session.run(None, inputs)[0]
                yield to_pcm16(np.asarray(audio, dtype=np.float32).reshape(-1), sample_rate)
