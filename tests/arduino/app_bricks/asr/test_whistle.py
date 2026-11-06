# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The whistle model runs in the app process: utterances are cut locally at the pauses of the
speaker and transcribed one at a time by the Cactus runtime, here replaced by a fake engine."""

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from arduino.app_bricks.asr import ASRError, ASRUnavailableError, AutomaticSpeechRecognition, WAVAutomaticSpeechRecognition
from arduino.app_bricks.asr import whistle
from arduino.app_bricks.asr.whistle import MAX_UTTERANCE_SECONDS, UtteranceCollector, WhistleEngine, to_float_mono

from conftest import _FakeMic, _wav_bytes

RATE = 16000
CHUNK = 1024  # Microphone() default buffer size: 64 ms at 16 kHz
CHUNK_S = CHUNK / RATE
SPEECH = 0.1  # -20 dBFS
DIP = 0.025  # 12 dB below, one chunk every four, like syllables
NOISE = 0.0001  # -80 dBFS


def _chunks(pattern: list[tuple[float, float]]) -> list[np.ndarray]:
    """int16 chunks for (level, seconds) runs; a SPEECH run is modulated like real speech."""
    out = []
    for level, seconds in pattern:
        for i in range(round(seconds / CHUNK_S)):
            lv = DIP if level == SPEECH and i % 4 == 3 else level
            out.append(np.full(CHUNK, round(lv * 32767), dtype=np.int16))
    return out


def _collect(collector: UtteranceCollector, chunks: list[np.ndarray]) -> list[np.ndarray]:
    utterances = [u for c in chunks if (u := collector.push(c)) is not None]
    tail = collector.finish()
    return utterances + ([tail] if tail is not None else [])


def _seconds(pcm: np.ndarray) -> float:
    return pcm.size / RATE


@pytest.fixture
def whistle_model(monkeypatch: pytest.MonkeyPatch):
    """Configure the brick with the whistle model; the service must never be looked up."""

    def no_service(host):
        raise AssertionError("the whistle model needs no service address")

    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.get_brick_configured_model", lambda _id, brick_config=None: "whistle")
    monkeypatch.setattr("arduino.app_bricks.asr.local_asr.resolve_address", no_service)


class _FakeEngine:
    """Stands in for WhistleEngine: records the calls, answers with the given texts in turn."""

    def __init__(self, texts: list[str], language: str = "it"):
        self.texts = list(texts)
        self.language = language
        self.calls: list[tuple[np.ndarray, str | None, list[str] | None]] = []

    def transcribe(self, pcm, language=None, keywords=None):
        self.calls.append((pcm, language, keywords))
        return {"text": self.texts.pop(0) if self.texts else "", "language": language or self.language}


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch, whistle_model):
    engine = _FakeEngine(["Accendi la luce.", "Grazie!"])
    monkeypatch.setattr(WhistleEngine, "shared", classmethod(lambda cls: engine))
    return engine


@pytest.fixture
def worker_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(2)


class TestToFloatMono:
    def test_int16_full_scale_is_one(self):
        out = to_float_mono(np.full(8, 32767, dtype=np.int16), RATE)
        assert out.dtype == np.float32
        assert out == pytest.approx(np.ones(8), abs=1e-4)

    def test_uint8_midpoint_is_silence(self):
        assert to_float_mono(np.full(8, 128, dtype=np.uint8), RATE) == pytest.approx(np.zeros(8))

    def test_packed_24_bit_uses_24_bit_scale(self):
        assert to_float_mono(np.full(8, 2**23 - 1, dtype=np.int32), RATE, is_packed=True) == pytest.approx(np.ones(8), abs=1e-4)

    def test_channels_are_averaged(self):
        stereo = np.array([16384, 0, 16384, 0], dtype=np.int16)  # left half scale, right silent
        assert to_float_mono(stereo, RATE, channels=2) == pytest.approx([0.25, 0.25], abs=1e-4)

    def test_48k_is_resampled_to_16k_keeping_a_voice_tone(self):
        t = np.arange(48000) / 48000
        tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        out = to_float_mono(tone, 48000)
        assert out.size == 16000
        assert np.sqrt(np.mean(out**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)

    def test_16k_mono_float_is_unchanged(self):
        x = np.linspace(-1, 1, 10, dtype=np.float32)
        assert to_float_mono(x, RATE) == pytest.approx(x)


class TestUtteranceCollectorLive:
    """``gate=True``: a live microphone, silence between utterances is dropped."""

    def test_an_utterance_ends_at_the_pause_with_its_preroll(self):
        utterances = _collect(UtteranceCollector(RATE), _chunks([(NOISE, 1.0), (SPEECH, 2.0), (NOISE, 2.0)]))
        assert len(utterances) == 1
        # ~0.3 s pre-roll + 2 s speech + the 0.7 s pause that ended it
        assert _seconds(utterances[0]) == pytest.approx(3.0, abs=2 * CHUNK_S)

    def test_two_sentences_are_two_utterances(self):
        pattern = [(NOISE, 1.0), (SPEECH, 1.5), (NOISE, 1.0), (SPEECH, 1.5), (NOISE, 1.0)]
        assert len(_collect(UtteranceCollector(RATE), _chunks(pattern))) == 2

    def test_a_short_breath_does_not_cut(self):
        pattern = [(NOISE, 1.0), (SPEECH, 1.5), (NOISE, 0.3), (SPEECH, 1.5), (NOISE, 1.0)]
        assert len(_collect(UtteranceCollector(RATE), _chunks(pattern))) == 1

    def test_a_click_is_not_an_utterance(self):
        assert _collect(UtteranceCollector(RATE), _chunks([(NOISE, 1.0), (SPEECH, 0.1), (NOISE, 3.0)])) == []

    def test_a_click_and_the_silence_after_it_are_not_sent_with_the_next_sentence(self):
        utterances = _collect(UtteranceCollector(RATE), _chunks([(NOISE, 1.0), (SPEECH, 0.1), (NOISE, 3.0), (SPEECH, 2.0), (NOISE, 2.0)]))
        assert len(utterances) == 1
        assert _seconds(utterances[0]) == pytest.approx(3.0, abs=2 * CHUNK_S)

    def test_silence_is_never_buffered(self):
        collector = UtteranceCollector(RATE)
        _collect(collector, _chunks([(NOISE, 20.0)]))
        assert collector._chunks == []

    def test_continuous_speech_is_cut_within_one_runtime_call(self):
        utterances = _collect(UtteranceCollector(RATE), _chunks([(NOISE, 1.0), (SPEECH, 35.0)]))
        assert _seconds(utterances[0]) <= 30.0
        assert _seconds(utterances[0]) >= MAX_UTTERANCE_SECONDS

    def test_speech_cut_by_the_end_of_the_stream_is_returned(self):
        utterances = _collect(UtteranceCollector(RATE), _chunks([(NOISE, 1.0), (SPEECH, 2.0)]))
        assert len(utterances) == 1

    def test_pause_follows_the_configured_vad(self):
        pattern = [(NOISE, 1.0), (SPEECH, 1.5), (NOISE, 1.0), (SPEECH, 1.5), (NOISE, 2.0)]
        assert len(_collect(UtteranceCollector(RATE, pause_s=1.5), _chunks(pattern))) == 1


class TestUtteranceCollectorRecording:
    """``gate=False``: a finite recording, every sample is transcribed."""

    def test_the_whole_recording_is_kept(self):
        chunks = _chunks([(NOISE, 1.0), (SPEECH, 2.0), (NOISE, 0.3)])
        utterances = _collect(UtteranceCollector(RATE, gate=False), chunks)
        assert sum(u.size for u in utterances) == sum(c.size for c in chunks)

    def test_silence_after_the_last_utterance_is_dropped(self):
        chunks = _chunks([(NOISE, 1.0), (SPEECH, 2.0), (NOISE, 3.0)])
        assert len(_collect(UtteranceCollector(RATE, gate=False), chunks)) == 1

    def test_a_recording_with_no_detected_speech_is_cut_every_29_s(self):
        utterances = _collect(UtteranceCollector(RATE, gate=False), [np.zeros(CHUNK, dtype=np.int16)] * round(35 / CHUNK_S))
        assert _seconds(utterances[0]) == pytest.approx(MAX_UTTERANCE_SECONDS, abs=CHUNK_S)


class TestOptions:
    def test_no_service_is_needed(self, whistle_model):
        asr = AutomaticSpeechRecognition(mic=_FakeMic())
        assert asr.model == "whistle"

    def test_translate_raises_with_a_hint(self, whistle_model):
        with pytest.raises(ASRError, match="does not translate") as exc_info:
            AutomaticSpeechRecognition(mic=_FakeMic(), translate=True)
        assert "whisper-small-quantized" in (exc_info.value.hint or "")

    def test_an_unsupported_language_raises_with_the_supported_ones(self, whistle_model):
        with pytest.raises(ASRError, match="'ja'") as exc_info:
            WAVAutomaticSpeechRecognition(np.zeros(10, dtype=np.int16), language="ja")
        assert "it" in (exc_info.value.hint or "")

    def test_supported_language_and_keywords_are_accepted(self, whistle_model):
        asr = AutomaticSpeechRecognition(mic=_FakeMic(), language="it", keywords=["UNO Q"])
        assert asr.keywords == ["UNO Q"]

    def test_keywords_with_the_service_model_raise(self):
        with pytest.raises(ASRError, match="keywords"):
            AutomaticSpeechRecognition(mic=_FakeMic(), keywords=["UNO Q"])

    def test_translate_set_later_raises_when_the_session_starts(self, fake_engine, worker_loop):
        asr = WAVAutomaticSpeechRecognition(_wav_bytes(np.zeros(RATE, dtype=np.int16)))
        asr._worker_loop.set_result(worker_loop)
        asr.translate = True
        with pytest.raises(ASRError, match="does not translate"):
            asr.transcribe()


class TestEngineLoading:
    def test_missing_files_raise_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            WhistleEngine(tmp_path / "libcactus.so", tmp_path / "whistle.cact")

    def test_start_raises_unavailable_with_a_hint_on_an_image_without_the_model(self, monkeypatch, whistle_model, tmp_path):
        monkeypatch.setattr(whistle, "_LIB_PATH", tmp_path / "libcactus.so")
        monkeypatch.setattr(WhistleEngine, "_shared", None)
        asr = WAVAutomaticSpeechRecognition(np.zeros(10, dtype=np.int16))
        with pytest.raises(ASRUnavailableError, match="not installed") as exc_info:
            asr.start()
        assert exc_info.value.hint

    def test_start_loads_the_model(self, fake_engine):
        asr = WAVAutomaticSpeechRecognition(np.zeros(10, dtype=np.int16))
        asr.start()
        assert asr._engine is fake_engine


class TestTranscription:
    @staticmethod
    def _two_sentences() -> bytes:
        pcm = np.concatenate(_chunks([(NOISE, 1.0), (SPEECH, 1.5), (NOISE, 1.0), (SPEECH, 1.5), (NOISE, 0.3)]))
        return _wav_bytes(pcm)

    def test_each_sentence_is_one_full_text_with_its_language(self, fake_engine, worker_loop):
        asr = WAVAutomaticSpeechRecognition(self._two_sentences())
        asr._worker_loop.set_result(worker_loop)
        with asr.transcribe_stream() as stream:
            events = list(stream)
        assert [(e.type, e.data, e.language) for e in events] == [
            ("full_text", "Accendi la luce.", "it"),
            ("full_text", " Grazie!", "it"),
        ]

    def test_transcribe_joins_the_sentences(self, fake_engine, worker_loop):
        asr = WAVAutomaticSpeechRecognition(self._two_sentences())
        asr._worker_loop.set_result(worker_loop)
        assert asr.transcribe() == "Accendi la luce. Grazie!"

    def test_the_engine_gets_16k_float_audio_language_and_keywords(self, fake_engine, worker_loop):
        pcm = np.concatenate(_chunks([(NOISE, 1.0), (SPEECH, 1.5), (NOISE, 0.3)]))
        stereo_48k = np.repeat(np.repeat(pcm, 3), 2)  # same audio, 48 kHz stereo
        asr = WAVAutomaticSpeechRecognition(_wav_bytes(stereo_48k, sample_rate=48000, channels=2), language="it", keywords=["UNO Q"])
        asr._worker_loop.set_result(worker_loop)
        asr.transcribe()
        audio, language, keywords = fake_engine.calls[0]
        assert audio.dtype == np.float32
        assert _seconds(audio) == pytest.approx(pcm.size / RATE, abs=0.01)
        assert (language, keywords) == ("it", ["UNO Q"])

    def test_silence_gives_an_empty_transcription(self, fake_engine, worker_loop):
        fake_engine.texts = []
        asr = WAVAutomaticSpeechRecognition(np.zeros(RATE, dtype=np.int16))
        asr._worker_loop.set_result(worker_loop)
        assert asr.transcribe() == ""

    def test_engine_failure_raises_asr_error(self, fake_engine, worker_loop):
        def fail(pcm, language=None, keywords=None):
            raise RuntimeError("Whistle transcription failed: boom")

        fake_engine.transcribe = fail
        asr = WAVAutomaticSpeechRecognition(self._two_sentences())
        asr._worker_loop.set_result(worker_loop)
        with pytest.raises(ASRError, match="boom"):
            asr.transcribe()
