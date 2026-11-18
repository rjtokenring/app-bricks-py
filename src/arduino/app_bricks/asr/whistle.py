# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Whistle speech-to-text, served by the whistle-runner container of the arduino:whistle service.

The runtime library and the model ship in that image; nothing is downloaded and nothing
leaves the board. The brick cuts the audio into utterances at the pauses of the speaker
and sends them one at a time.
"""

import base64
from collections import deque
from typing import Any, cast

import numpy as np

from arduino.app_peripherals.microphone import SpeechSegmenter, chunk_level
from arduino.app_utils import http

WHISTLE_MODEL = "whistle"
"""Catalog id of the Whistle model."""

LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
"""ISO 639-1 codes of the languages Whistle transcribes."""

SAMPLE_RATE = 16000
"""Sample rate of the audio Whistle transcribes, in Hz."""

MAX_UTTERANCE_SECONDS = 29.0
"""Longest utterance transcribed at once: one runtime call takes at most 30 s of audio."""


class WhistleClient:
    """The Whistle server of the arduino:whistle service, which transcribes one utterance per request."""

    _HEALTH_RETRIES = 6
    _TRANSCRIBE_TIMEOUT_S = 60.0

    def __init__(self, base_url: str) -> None:
        """
        Args:
            base_url (str): URL of the server, e.g. ``http://whistle-runner:8087``.
        """
        self._base_url = base_url

    def wait_ready(self) -> None:
        """
        Wait until the server answers, for a few seconds: it loads the model before it listens.

        Raises:
            http.ConnectionFailed: If the server does not answer.
            http.HTTPStatusError: If it answers with an error.
        """
        http.get(f"{self._base_url}/health", timeout=2, retries=self._HEALTH_RETRIES, backoff_factor=0.25).raise_for_status()

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
            http.ConnectionFailed: If the server does not answer.
            RuntimeError: If the server fails the transcription.
        """
        samples = np.ascontiguousarray(pcm[: 30 * SAMPLE_RATE], dtype="<f4")
        body = {"pcm": base64.b64encode(samples.tobytes()).decode(), "language": language, "keywords": keywords}
        response = http.post(f"{self._base_url}/transcribe", json=body, timeout=self._TRANSCRIBE_TIMEOUT_S)
        if response.status_code != 200:
            try:
                error = str(response.json().get("error"))
            except ValueError:
                error = response.text[:200]
            raise RuntimeError(f"Whistle transcription failed (HTTP {response.status_code}): {error}")
        return cast(dict[str, Any], response.json())


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
