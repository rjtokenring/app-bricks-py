# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Whistle speech-to-text, run in this process through the C API of the Cactus runtime (cactus.h).

The runtime library and the model ship in the app image; nothing is downloaded and
nothing leaves the board.
"""

import ctypes
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np

from arduino.app_peripherals.microphone import SpeechSegmenter, chunk_level

WHISTLE_MODEL = "whistle"
"""Catalog id of the Whistle model."""

LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
"""ISO 639-1 codes of the languages Whistle transcribes."""

SAMPLE_RATE = 16000
"""Sample rate of the audio Whistle transcribes, in Hz."""

MAX_UTTERANCE_SECONDS = 29.0
"""Longest utterance transcribed at once: one runtime call takes at most 30 s of audio."""

_LIB_PATH = Path("/usr/local/lib/cactus/libcactus.so")
_MODEL_PATH = Path("/models/ootb/cactus/whistle.cact")
_OUT_CAPACITY = 1 << 16


class WhistleEngine:
    """
    The Whistle model loaded in the Cactus runtime.

    The runtime keeps one process-wide model state (``cactus_load`` returns no handle), so
    there is a single engine per process: :meth:`shared` loads it on first use and returns
    the same instance to every caller. Transcriptions are serialized.
    """

    _shared: "WhistleEngine | None" = None
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "WhistleEngine":
        """
        The engine of this process, loaded on the first call.

        Raises:
            FileNotFoundError: If the runtime library or the model is not in the image.
            OSError: If the runtime library cannot be loaded.
            RuntimeError: If the runtime rejects the model.
        """
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(_LIB_PATH, _MODEL_PATH)
            return cls._shared

    def __init__(self, lib_path: Path, model_path: Path) -> None:
        """
        Load the runtime library and the model.

        Args:
            lib_path (Path): The Cactus runtime shared library.
            model_path (Path): The ``.cact`` model file.

        Raises:
            FileNotFoundError: If one of the files is missing.
            OSError: If the runtime library cannot be loaded.
            RuntimeError: If the runtime rejects the model.
        """
        for path in (lib_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
        lib = ctypes.CDLL(str(lib_path))
        self._load = lib.cactus_load
        self._load.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong]
        self._load.restype = ctypes.c_int
        self._last_error = lib.cactus_last_error
        self._last_error.argtypes = []
        self._last_error.restype = ctypes.c_char_p
        self._transcribe = lib.cactus_transcribe
        self._transcribe.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pcm
            ctypes.c_int,  # samples
            ctypes.c_char_p,  # language, NULL to detect it
            ctypes.c_char_p,  # keywords, one per line, or NULL
            ctypes.c_int,  # word timestamps
            ctypes.c_char_p,  # out: JSON result
            ctypes.c_int,  # out capacity
        ]
        self._transcribe.restype = ctypes.c_int
        self._lock = threading.Lock()
        self._out = ctypes.create_string_buffer(_OUT_CAPACITY)
        # Kept alive with the engine: the runtime may keep pointers into the buffer it loaded
        self._model = model_path.read_bytes()
        if self._load(self._model, len(self._model)) < 0:
            raise RuntimeError(f"The runtime rejected {model_path}: {self._error()}")

    def _error(self) -> str:
        message = cast(bytes | None, self._last_error())
        return message.decode(errors="replace") if message else "unknown error"

    def transcribe(self, pcm: np.ndarray, language: str | None = None, keywords: list[str] | None = None) -> dict[str, Any]:
        """
        Transcribe one utterance.

        Args:
            pcm (np.ndarray): Samples in -1..1 at 16 kHz mono. Audio past 30 s is ignored.
            language (str | None): ISO 639-1 code of the spoken language, or None to detect it.
            keywords (list[str] | None): Words or phrases the transcription should favour.

        Returns:
            dict[str, Any]: The runtime result: ``text``, ``language`` (the one detected or
                given) and timing fields.

        Raises:
            RuntimeError: If the runtime fails.
        """
        samples = np.ascontiguousarray(pcm[: 30 * SAMPLE_RATE], dtype=np.float32)
        lang = language.encode() if language else None
        words = "\n".join(keywords).encode() if keywords else None
        with self._lock:
            ret = self._transcribe(samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size, lang, words, 0, self._out, _OUT_CAPACITY)
            if ret < 0:
                raise RuntimeError(f"Whistle transcription failed: {self._error()}")
            raw = self._out.value
        return cast(dict[str, Any], json.loads(raw))


def to_float_mono(samples: np.ndarray, sample_rate: int, channels: int = 1, is_packed: bool = False) -> np.ndarray:
    """
    Convert PCM samples to what Whistle transcribes: float32 in -1..1, 16 kHz mono.

    Channels are averaged. A different rate is resampled with linear interpolation, after a
    moving average over the decimation ratio when the rate goes down.

    Args:
        samples (np.ndarray): PCM samples in any ``BaseMicrophone`` format, interleaved if multichannel.
        sample_rate (int): Sample rate of the samples, in Hz.
        channels (int): Number of interleaved channels. Default: 1.
        is_packed (bool): True for 24-bit samples packed in int32. Default: False.

    Returns:
        np.ndarray: float32 samples at 16 kHz mono.
    """
    x = samples.astype(np.float32)
    kind = samples.dtype.kind
    if kind == "i":
        bits = 24 if is_packed else 8 * samples.dtype.itemsize
        x /= float(2 ** (bits - 1))
    elif kind == "u":
        half = float(2 ** (8 * samples.dtype.itemsize - 1))
        x = (x - half) / half
    if channels > 1:
        x = x[: x.size - x.size % channels].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if sample_rate == SAMPLE_RATE or x.size == 0:
        return x
    ratio = sample_rate / SAMPLE_RATE
    if ratio > 1:
        width = round(ratio)
        if width > 1:
            x = np.convolve(x, np.full(width, 1.0 / width, dtype=np.float32), mode="same")
    n_out = max(1, round(x.size / ratio))
    return np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x).astype(np.float32)


class UtteranceCollector:
    """
    Cuts a stream of PCM chunks into the utterances to transcribe, at the pauses of the speaker.

    An utterance ends at a pause of ``pause_s``, or once it is ``MAX_UTTERANCE_SECONDS`` long.
    With ``gate`` set (a live microphone), audio outside utterances is dropped except the
    ``preroll_s`` before the first speech, and so is a burst with less than ``min_speech_s``
    of speech followed by a pause (a click, a cough). Without ``gate`` (a finite recording)
    every sample belongs to an utterance.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        is_packed: bool = False,
        pause_s: float = 0.7,
        gate: bool = True,
        preroll_s: float = 0.3,
        min_speech_s: float = 0.3,
    ) -> None:
        """
        Args:
            sample_rate (int): Sample rate of the chunks, in Hz.
            channels (int): Number of interleaved channels. Default: 1.
            is_packed (bool): True for 24-bit samples packed in int32. Default: False.
            pause_s (float): Pause that ends an utterance, in seconds. Default: 0.7.
            gate (bool): Drop the audio outside utterances. Default: True.
            preroll_s (float): Audio kept before the first speech of an utterance, in seconds. Default: 0.3.
            min_speech_s (float): Speech an utterance needs, in seconds. Default: 0.3.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._is_packed = is_packed
        self._gate = gate
        self._preroll_s = preroll_s
        self._min_speech_s = min_speech_s
        self._segmenter = SpeechSegmenter(pause_s=pause_s, min_speech_s=min_speech_s)
        self._chunks: list[np.ndarray] = []
        self._buffered_s = 0.0
        self._preroll: deque[tuple[np.ndarray, float]] = deque()
        self._preroll_buffered_s = 0.0
        self._cuts = 0

    def push(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Account one PCM chunk.

        Args:
            chunk (np.ndarray): PCM samples, interleaved if multichannel.

        Returns:
            np.ndarray | None: The utterance this chunk ends, converted by :func:`to_float_mono`,
                or None.
        """
        duration_s = chunk.size / (self._sample_rate * self._channels)
        ended = self._segmenter.update_level(chunk_level(chunk, self._is_packed), duration_s)
        if self._gate and not self._segmenter.in_segment:
            self._keep_preroll(chunk, duration_s)
            return None
        if self._preroll:
            self._chunks.extend(c for c, _ in self._preroll)
            self._buffered_s += self._preroll_buffered_s
            self._preroll.clear()
            self._preroll_buffered_s = 0.0
        self._chunks.append(chunk)
        self._buffered_s += duration_s
        if self._gate and not ended and self._segmenter.detector.paused and self._segmenter.speech_s < self._min_speech_s:
            self._drop()
            return None
        if ended or self._buffered_s >= MAX_UTTERANCE_SECONDS:
            return self._cut()
        return None

    def finish(self) -> np.ndarray | None:
        """
        End the stream.

        Returns:
            np.ndarray | None: The utterance in progress, or None if there is none. With
                ``gate``, one without enough speech is dropped; without it, so is the audio
                after the last utterance when no speech started there.
        """
        if self._gate:
            no_utterance = self._segmenter.speech_s < self._min_speech_s
        else:
            no_utterance = self._cuts > 0 and not self._segmenter.in_segment
        if not self._chunks or no_utterance:
            self._drop()
            return None
        return self._cut()

    def _keep_preroll(self, chunk: np.ndarray, duration_s: float) -> None:
        self._preroll.append((chunk, duration_s))
        self._preroll_buffered_s += duration_s
        while self._preroll_buffered_s - self._preroll[0][1] >= self._preroll_s:
            self._preroll_buffered_s -= self._preroll.popleft()[1]

    def _cut(self) -> np.ndarray:
        pcm = to_float_mono(np.concatenate(self._chunks), self._sample_rate, self._channels, self._is_packed)
        self._drop()
        self._cuts += 1
        return pcm

    def _drop(self) -> None:
        self._chunks = []
        self._buffered_s = 0.0
        self._segmenter.reset()
