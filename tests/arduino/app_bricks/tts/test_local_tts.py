# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import base64
import contextlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from arduino.app_bricks.tts import TextToSpeech, TTSBusyError
from arduino.app_bricks.tts.local_tts import LLAMACPP_LEAD_SILENCE_S, TTS_MAX_CHARS
from arduino.app_peripherals.speaker import BaseSpeaker, FormatPlain, FormatPacked
from arduino.app_utils import App


class FakeResponse:
    """A runner response: read whole (status_code, json()) or streamed (status_code, iter_content(), json())."""

    def __init__(self, status_code=200, json_data=None, content=b"", chunks=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self._chunks = chunks
        self.headers = headers or {}
        self.close_called = False

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size):
        if self._chunks is not None:
            yield from self._chunks
            return

        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def close(self):
        self.close_called = True


class BlockingSpeaker(BaseSpeaker):
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        format: FormatPlain | FormatPacked = np.int16,
        buffer_size: int = 4,
        auto_reconnect: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, channels=channels, format=format, buffer_size=buffer_size, auto_reconnect=auto_reconnect)
        self.chunks_written = []
        self.first_chunk_written = threading.Event()
        self.release_first_chunk = threading.Event()
        self.close_called = False

    def _open_speaker(self):
        pass

    def _close_speaker(self):
        self.close_called = True

    def _write_audio(self, audio_chunk: np.ndarray):
        self.chunks_written.append(audio_chunk.copy())
        if len(self.chunks_written) == 1:
            self.first_chunk_written.set()
            self.release_first_chunk.wait(timeout=2)


def make_tts(monkeypatch, speaker, post_response, cancel_response=None, **tts_kwargs):
    models = [
        {
            # The runner reports the name from the model's config.json (underscore
            # style), while the brick is configured with the catalog id (piper-tts-en).
            "name": "pipertts_en",
            "voices": [
                {
                    "language": "en",
                    "name": "default",
                    "sample_rate": 44100,
                }
            ],
        }
    ]

    def post(url, json=None, **kwargs):
        assert url.endswith("/tts/cancel"), f"{url}: the synthesis must be streamed"
        if cancel_response is not None:
            return cancel_response(url, json, **kwargs)
        return FakeResponse(content=b"cancelled")

    @contextlib.contextmanager
    def stream(method, url, json=None, **kwargs):
        assert (method, url.rsplit("/", 2)[-2:]) == ("POST", ["tts", "synthesize"])
        response = post_response(url, json, **kwargs)
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr("arduino.app_utils.http.get", lambda url, **kwargs: FakeResponse(json_data=models))
    monkeypatch.setattr("arduino.app_utils.http.post", post)
    monkeypatch.setattr("arduino.app_utils.http.stream", stream)

    tts = TextToSpeech(speaker=speaker, **tts_kwargs)
    App.unregister(tts)
    speaker.start()
    return tts


def test_resolve_voice_matches_runner_name_with_different_separators(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=b""))

    # Configured catalog id `piper-tts-en` resolves to the runner-reported
    # `pipertts_en`, which is what synthesis requests must send.
    assert tts._voice["model"] == "pipertts_en"
    assert tts._voice["language"] == "en"


def test_cancel_without_active_speech_keeps_speaker_running(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    tts.cancel()

    assert speaker.is_started() is True
    assert speaker.close_called is False
    assert speaker.chunks_written == []


def test_chunk_text_splits_on_sentence_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 1000}. {'b' * 1000}"

    chunks = tts._chunk_text(text)

    assert chunks == [f"{'a' * 1000}.", "b" * 1000]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_newline_when_no_sentence_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 900}\n{'b' * 300}"

    chunks = tts._chunk_text(text)

    assert chunks == ["a" * 900, "b" * 300]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_space_when_no_sentence_or_newline_boundary(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))
    text = f"{'a' * 900} {'b' * 300}"

    chunks = tts._chunk_text(text)

    assert chunks == ["a" * 900, "b" * 300]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_chunk_text_splits_on_character_limit(monkeypatch):
    speaker = BlockingSpeaker()
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=np.arange(4, dtype=np.int16).tobytes()))

    chunks = tts._chunk_text("é" * 1200)

    assert chunks == ["é" * TTS_MAX_CHARS, "é" * 176]
    assert all(len(chunk) <= TTS_MAX_CHARS for chunk in chunks)


def test_speak_synthesizes_text_chunks(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []
    text = f"{'a' * 1000}. {'b' * 1000}"

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        assert json["keep_alive"] is True
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)
    expected_chunks = tts._chunk_text(text)

    tts.speak(text)

    assert post_calls == expected_chunks
    assert len(speaker.chunks_written) == len(expected_chunks)


def test_synthesize_pcm_stream_yields_response_chunks(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    audio_chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]

    def post_response(url, json, **kwargs):
        assert json["keep_alive"] is False
        return FakeResponse(chunks=audio_chunks)

    tts = make_tts(monkeypatch, speaker, post_response)

    assert list(tts.synthesize_pcm_stream("hello")) == audio_chunks
    assert tts.synthesize_pcm("hello") == b"".join(audio_chunks)


def test_speak_plays_first_stream_chunk_before_response_completes(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    first_chunk_sent = threading.Event()
    release_second_chunk = threading.Event()
    audio_chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]

    def stream_chunks():
        first_chunk_sent.set()
        yield audio_chunks[0]
        release_second_chunk.wait(timeout=2)
        yield audio_chunks[1]

    def post_response(url, json, **kwargs):
        return FakeResponse(chunks=stream_chunks())

    tts = make_tts(monkeypatch, speaker, post_response)
    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)

    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)
    assert first_chunk_sent.is_set()
    assert speak_thread.is_alive() is True
    np.testing.assert_array_equal(speaker.chunks_written[0], np.arange(4, dtype=np.int16))

    speaker.release_first_chunk.set()
    release_second_chunk.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert len(speaker.chunks_written) == 2


def test_cancel_stops_playback_without_stopping_speaker(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    cancel_calls = []
    tts = make_tts(
        monkeypatch,
        speaker,
        lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()),
        cancel_response=lambda url, json=None, **kwargs: cancel_calls.append(url) or FakeResponse(content=b"cancelled"),
    )

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)
    tts.cancel()
    speaker.release_first_chunk.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert len(speaker.chunks_written) == 1
    np.testing.assert_array_equal(speaker.chunks_written[0], pcm_audio[:4])
    assert speaker.is_started() is True
    assert speaker.close_called is False
    assert cancel_calls == [tts.api_base_url + "/tts/cancel"]


def test_cancel_during_synthesis_skips_playback(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    synthesis_started = threading.Event()
    release_synthesis = threading.Event()
    pcm_audio = np.arange(12, dtype=np.int16)

    def post_response(url, json, **kwargs):
        synthesis_started.set()
        release_synthesis.wait(timeout=2)
        return FakeResponse(content=pcm_audio.tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert synthesis_started.wait(timeout=2)
    tts.cancel()
    release_synthesis.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert speaker.chunks_written == []
    assert speaker.is_started() is True
    assert speaker.close_called is False


def test_synthesize_pcm_raises_when_busy(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    first_synthesis_started = threading.Event()
    release_first_synthesis = threading.Event()
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        if json["text"] == "first":
            first_synthesis_started.set()
            release_first_synthesis.wait(timeout=2)
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    first_thread = threading.Thread(target=tts.synthesize_pcm, args=("first",), daemon=True)
    first_thread.start()
    assert first_synthesis_started.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.synthesize_pcm("second")

    assert post_calls == ["first"]

    release_first_synthesis.set()
    first_thread.join(timeout=2)

    assert first_thread.is_alive() is False
    assert post_calls == ["first"]


def test_speak_raises_when_busy(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.speak("second")

    speaker.release_first_chunk.set()
    tts.cancel()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_speak_non_blocking_returns_before_playback_completes(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)

    assert speaker.first_chunk_written.wait(timeout=2)
    assert tts.is_speaking() is True

    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert len(speaker.chunks_written) == 3
    np.testing.assert_array_equal(np.concatenate(speaker.chunks_written), pcm_audio)


def test_speak_non_blocking_queues_sequential_playback(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    tts.speak("first", block=False)
    worker_thread = tts._speak_thread
    tts.speak("second", block=False)

    assert tts._speak_thread is worker_thread

    assert wait_until(lambda: not tts.is_speaking())
    assert post_calls == ["first", "second"]
    assert len(speaker.chunks_written) == 2


def test_speak_non_blocking_raises_when_queue_full(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()), max_queue_size=1)

    tts.speak("first", block=False)
    # The worker dequeues "first" and blocks on the first written chunk, leaving the queue empty.
    assert speaker.first_chunk_written.wait(timeout=2)
    tts.speak("second", block=False)

    with pytest.raises(TTSBusyError):
        tts.speak("third", block=False)

    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert len(speaker.chunks_written) == 6  # "first" and "second" played, "third" rejected


def test_max_queue_size_must_be_positive(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)

    with pytest.raises(ValueError):
        make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=b""), max_queue_size=0)


def test_speak_blocking_raises_when_background_playback_active(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)
    assert speaker.first_chunk_written.wait(timeout=2)

    with pytest.raises(TTSBusyError):
        tts.speak("second")

    tts.cancel()
    speaker.release_first_chunk.set()
    assert wait_until(lambda: not tts.is_speaking())


def test_speak_non_blocking_cancel_stops_playback_and_drops_queued_speech(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=pcm_audio.tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    tts.speak("first", block=False)
    assert speaker.first_chunk_written.wait(timeout=2)
    tts.speak("second", block=False)

    tts.cancel()
    speaker.release_first_chunk.set()

    assert wait_until(lambda: not tts.is_speaking())
    assert post_calls == ["first"]
    assert len(speaker.chunks_written) == 1
    assert speaker.is_started() is True


def test_speak_non_blocking_logs_synthesis_errors(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    error_response = FakeResponse(status_code=500, json_data={"error": {"message": "synthesis boom"}})
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: error_response)

    logged_errors = []
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.logger.error", lambda msg, *args, **kwargs: logged_errors.append(msg))

    tts.speak("hello", block=False)

    assert wait_until(lambda: not tts.is_speaking())
    assert speaker.chunks_written == []
    assert any("synthesis boom" in msg for msg in logged_errors)


def test_stop_terminates_background_worker(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)
    worker_thread = tts._speak_thread

    assert speaker.first_chunk_written.wait(timeout=2)
    speaker.release_first_chunk.set()
    tts.stop()

    worker_thread.join(timeout=2)
    assert worker_thread.is_alive() is False
    assert tts._speak_thread is None
    assert tts.is_speaking() is False
    assert speaker.close_called is True


def test_synthesis_streams_from_a_real_http_server(monkeypatch):
    """No mock of the HTTP client: each HTTP chunk is yielded as it arrives, and a JSON error body becomes the error."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from arduino.app_utils import http

    real_stream = http.stream
    chunks = [np.arange(4, dtype=np.int16).tobytes(), np.arange(4, 8, dtype=np.int16).tobytes()]
    send_second_chunk = threading.Event()

    class Runner(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if request["text"] == "fail":
                body = json.dumps({"error": {"message": "synthesis boom"}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for index, chunk in enumerate(chunks):
                if index == 1:
                    send_second_chunk.wait(5)
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Runner)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        speaker = BlockingSpeaker(buffer_size=4)  # Reads of 4 int16 samples: one chunk at a time
        tts = make_tts(monkeypatch, speaker, post_response=None)
        monkeypatch.setattr(http, "stream", real_stream)
        tts.api_base_url = f"http://127.0.0.1:{server.server_address[1]}/audio-analytics/v1/api"

        audio = iter(tts.synthesize_pcm_stream("hello"))
        assert next(audio) == chunks[0]  # Before the runner sends the rest
        send_second_chunk.set()
        assert list(audio) == chunks[1:]

        with pytest.raises(RuntimeError, match="synthesis boom"):
            list(tts.synthesize_pcm_stream("fail"))
    finally:
        server.shutdown()
        server.server_close()


def test_speak_stream_speaks_each_sentence_while_the_text_still_arrives(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []
    first_synthesized = threading.Event()

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        first_synthesized.set()
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)

    def answer():
        yield "Hello wor"
        yield "ld. How"
        # The first sentence is complete: it is being spoken before the rest is generated.
        assert first_synthesized.wait(timeout=2)
        yield " are you?\nFine"

    spoken = tts.speak_stream(answer())

    assert spoken == "Hello world. How are you? Fine"
    assert post_calls == ["Hello world.", "How are you?", "Fine"]
    assert not tts.is_speaking()


def test_speak_stream_without_text_says_nothing(monkeypatch):
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=b"")

    tts = make_tts(monkeypatch, BlockingSpeaker(), post_response)

    assert tts.speak_stream(iter(["", "  ", "\n"])) == ""
    assert post_calls == []


def test_speak_stream_stops_reading_the_text_after_cancel(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    post_calls = []

    def post_response(url, json, **kwargs):
        post_calls.append(json["text"])
        return FakeResponse(content=np.arange(4, dtype=np.int16).tobytes())

    tts = make_tts(monkeypatch, speaker, post_response)
    read = []

    def answer():
        yield "One. "
        tts.cancel()
        read.append("after cancel")
        yield "Two. "
        read.append("never read")
        yield "Three."

    spoken = tts.speak_stream(answer())

    assert spoken == "One."
    assert read == ["after cancel"]
    assert "Two." not in post_calls and "Three." not in post_calls


def test_wait_until_done_waits_for_queued_speech(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    pcm_audio = np.arange(12, dtype=np.int16)
    tts = make_tts(monkeypatch, speaker, lambda url, json, **kwargs: FakeResponse(content=pcm_audio.tobytes()))

    tts.speak("hello", block=False)
    assert speaker.first_chunk_written.wait(timeout=2)

    assert tts.wait_until_done(timeout=0.05) is False
    speaker.release_first_chunk.set()
    assert tts.wait_until_done(timeout=2) is True
    assert not tts.is_speaking()


# --------------------------------------------------------------------------- #
# llama.cpp speech service (llamacpp:* models, e.g. Pocket TTS)
# --------------------------------------------------------------------------- #

LLAMACPP_PCM = "audio/pcm;rate=24000;encoding=float;bits=32"
POCKET_VOICES = ["alba", "giovanni", "vera"]


def make_llamacpp_tts(
    monkeypatch, speaker, post_response, served=("pocket-tts-italian",), voices=POCKET_VOICES, clones=False, default_voice=None, **tts_kwargs
):
    """A TextToSpeech configured with llamacpp:pocket-tts-italian, whose runner serves *served*, each with the
    predefined *voices*, and with a voice encoder when it *clones*. The models list declares *default_voice* for it."""
    from arduino.app_internal.core.module import ModelEntry

    requests = []

    @contextlib.contextmanager
    def stream(method, url, json=None, **kwargs):
        requests.append((method, url, json))
        response = post_response(url, json, **kwargs)
        try:
            yield response
        finally:
            response.close()

    def get(url, params=None, **kwargs):
        assert url == "http://llamacpp-tts-runner:9998/props"
        if params["model"] not in served:  # The router's answer for a model it does not serve
            return FakeResponse(status_code=400, json_data={"error": {"code": 400, "message": f"model '{params['model']}' not found"}})
        props = {"modalities": {"vision": False, "video": False, "audio": clones}}
        return FakeResponse(json_data=props | ({"voices": voices} if voices else {}))

    def post(url, json=None, **kwargs):
        raise AssertionError(f"{url}: llama.cpp is never sent a POST outside of the synthesis stream")

    metadata = {"default_voice": default_voice} if default_voice else {}
    catalog = {"llamacpp:pocket-tts-italian": ModelEntry(model_id="llamacpp:pocket-tts-italian", metadata=metadata)}
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.get_brick_configured_model", lambda brick_id: "llamacpp:pocket-tts-italian")
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.load_model_list", lambda: catalog)
    monkeypatch.setattr("arduino.app_utils.http.get", get)
    monkeypatch.setattr("arduino.app_utils.http.post", post)
    monkeypatch.setattr("arduino.app_utils.http.stream", stream)

    tts = TextToSpeech(speaker=speaker, **tts_kwargs)
    App.unregister(tts)
    speaker.start()
    return tts, requests


def float_pcm_response(samples, chunk_bytes=None):
    """A llama.cpp /tts stream of float32 *samples*, cut into pieces of *chunk_bytes*."""
    data = np.asarray(samples, dtype="<f4").tobytes()
    size = chunk_bytes or len(data)
    return FakeResponse(chunks=[data[i : i + size] for i in range(0, len(data), size)], headers={"Content-Type": LLAMACPP_PCM})


def test_llamacpp_model_synthesizes_through_the_speech_service(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    samples = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24000)
    tts, requests = make_llamacpp_tts(monkeypatch, speaker, lambda url, json, **kwargs: float_pcm_response(samples), voice="giovanni")

    pcm = np.frombuffer(tts.synthesize_pcm("Ciao a tutti."), dtype="<i2")

    ((method, url, payload),) = requests
    assert (method, url) == ("POST", "http://llamacpp-tts-runner:9998/tts")
    # A predefined voice by name, no recording to clone (speaker_ref)
    assert set(payload) == {"model", "input", "voice", "n_predict", "response_format", "stream"}
    assert (payload["model"], payload["input"], payload["voice"]) == ("pocket-tts-italian", "Ciao a tutti.", "giovanni")
    assert (payload["response_format"], payload["stream"]) == ("pcm", True)
    # 0.1 s at 24 kHz becomes 0.1 s of 16-bit PCM at the speaker rate, the same tone, after the lead silence
    speech = pcm[round(LLAMACPP_LEAD_SILENCE_S * 44100) :]
    assert abs(speech.size - 4410) <= 1
    expected = np.rint(0.5 * np.sin(2 * np.pi * 440 * np.arange(speech.size) / 44100) * 32767)
    assert np.max(np.abs(speech - expected)) < 0.01 * 32767


def test_llamacpp_speak_stream_sends_a_one_word_sentence_with_the_next_one(monkeypatch):
    """The llama.cpp speech service distorts a request of a single word ("Ciao!"): it is spoken together with what follows."""
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    tts, requests = make_llamacpp_tts(monkeypatch, speaker, lambda url, json, **kwargs: float_pcm_response([0.0] * 8))

    spoken = tts.speak_stream(iter(["Ciao! ", "Come stai? Bene", ", grazie.\nPerfetto!\n", "A dopo."]))

    assert [payload["input"] for _method, _url, payload in requests] == ["Ciao! Come stai?", "Bene, grazie.", "Perfetto! A dopo."]
    assert spoken == "Ciao! Come stai? Bene, grazie. Perfetto! A dopo."


def test_llamacpp_speak_stream_still_speaks_a_lone_word(monkeypatch):
    speaker = BlockingSpeaker(buffer_size=4)
    speaker.release_first_chunk.set()
    tts, requests = make_llamacpp_tts(monkeypatch, speaker, lambda url, json, **kwargs: float_pcm_response([0.0] * 8))

    assert tts.speak_stream(iter(["Ciao!"])) == "Ciao!"
    assert [payload["input"] for _method, _url, payload in requests] == ["Ciao!"]


def test_llamacpp_speech_opens_with_a_lead_silence_once_per_request(monkeypatch):
    """Pocket TTS speaks from its first frames: the audio opens with silence, then the speech, unbroken."""
    lead = round(LLAMACPP_LEAD_SILENCE_S * 44100)
    constant = lambda url, json, **kwargs: float_pcm_response(np.full(2400, 0.5), 64)  # noqa: E731
    tts, _ = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), constant)

    chunks = [np.frombuffer(chunk, dtype="<i2") for chunk in tts.synthesize_pcm_stream("hello")]

    assert LLAMACPP_LEAD_SILENCE_S >= 0.2
    assert np.all(chunks[0][:lead] == 0)
    speech = np.concatenate(chunks)[lead:]
    assert np.all(speech == round(0.5 * 32767))


def test_llamacpp_request_allows_frames_for_the_whole_chunk(monkeypatch):
    """llama-server stops at 512 frames (41 s) unless told otherwise: a full chunk needs more."""
    speaker = BlockingSpeaker(buffer_size=4)
    tts, requests = make_llamacpp_tts(monkeypatch, speaker, lambda url, json, **kwargs: float_pcm_response([0.0] * 8))

    tts.synthesize_pcm("a" * TTS_MAX_CHARS)
    tts.synthesize_pcm("ok")

    # Measured: about 0.75 frames per character, 12.5 frames per second
    assert requests[0][2]["n_predict"] >= 1.5 * 0.75 * TTS_MAX_CHARS
    assert requests[1][2]["n_predict"] >= 64


def test_llamacpp_stream_is_seamless_across_unaligned_chunks(monkeypatch):
    """HTTP chunks split float samples anywhere: the output is the one of the whole stream at once."""
    samples = np.random.default_rng(0).uniform(-1, 1, 1000)
    whole, _ = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), lambda url, json, **kwargs: float_pcm_response(samples))
    expected = whole.synthesize_pcm("hello")
    pieces, _ = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), lambda url, json, **kwargs: float_pcm_response(samples, 7))

    assert len(expected) > 0
    assert pieces.synthesize_pcm("hello") == expected


def test_llamacpp_stream_without_a_sample_rate_is_an_error(monkeypatch):
    response = FakeResponse(chunks=[b"\0" * 16], headers={"Content-Type": "audio/wav"})
    tts, _ = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), lambda url, json, **kwargs: response)

    with pytest.raises(RuntimeError, match="audio/wav"):
        tts.synthesize_pcm("hello")


def test_llamacpp_model_speaks_with_the_default_voice_of_the_models_list(monkeypatch):
    """The voice is named in every request, also when the app does not pick one."""
    silence = lambda url, json, **kwargs: float_pcm_response([0.0] * 8)  # noqa: E731
    tts, requests = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), silence, default_voice="giovanni")

    tts.synthesize_pcm("ciao")

    assert requests[0][2]["voice"] == "giovanni"


def test_llamacpp_voice_of_the_app_wins_over_the_default_of_the_models_list(monkeypatch):
    silence = lambda url, json, **kwargs: float_pcm_response([0.0] * 8)  # noqa: E731
    tts, requests = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), silence, default_voice="giovanni", voice="vera")

    tts.synthesize_pcm("ciao")

    assert requests[0][2]["voice"] == "vera"


def test_llamacpp_model_speaks_with_its_first_voice_without_a_default_in_the_models_list(monkeypatch):
    """A hand-installed model has no entry in the models list."""
    tts, requests = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), lambda url, json, **kwargs: float_pcm_response([0.0] * 8))

    tts.synthesize_pcm("hello")

    assert requests[0][2]["voice"] == "alba"


def test_llamacpp_default_voice_the_model_lacks_falls_back_to_its_first_voice(monkeypatch):
    warnings = []
    monkeypatch.setattr("arduino.app_bricks.tts.local_tts.logger.warning", lambda msg, *args, **kwargs: warnings.append(msg))
    silence = lambda url, json, **kwargs: float_pcm_response([0.0] * 8)  # noqa: E731
    tts, requests = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), silence, default_voice="cosette")

    tts.synthesize_pcm("ciao")

    assert requests[0][2]["voice"] == "alba"
    assert len(warnings) == 1 and "cosette" in warnings[0]


def test_llamacpp_voice_must_be_one_of_the_model(monkeypatch):
    from arduino.app_bricks.tts import TTSError
    from arduino.app_utils import AppError

    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(TTSError, match="giovani") as unknown:
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, voice="giovani")
    assert isinstance(unknown.value, AppError)
    assert "alba, giovanni, vera" in (unknown.value.hint or "")


def test_llamacpp_model_without_predefined_voices_needs_a_voice_reference(monkeypatch):
    """Without a voice prompt Pocket TTS says a fraction of a second of noise (measured): refuse at init."""
    from arduino.app_bricks.tts import TTSError

    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(TTSError, match="no predefined voices") as missing:
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, voices=[], clones=True)
    assert "voice_reference=" in (missing.value.hint or "")


def test_llamacpp_voice_reference_is_cloned_by_a_model_with_a_voice_encoder(monkeypatch, tmp_path):
    recording = tmp_path / "voice.wav"
    recording.write_bytes(b"RIFF fake voice")
    silence = lambda url, json, **kwargs: float_pcm_response([0.0] * 8)  # noqa: E731
    tts, requests = make_llamacpp_tts(monkeypatch, BlockingSpeaker(buffer_size=4), silence, voices=[], clones=True, voice_reference=recording)

    tts.synthesize_pcm("hello")

    payload = requests[0][2]
    assert payload["speaker_ref_b64"] == base64.b64encode(b"RIFF fake voice").decode()
    assert "voice" not in payload
    assert tts._voice["name"] == "voice.wav"


def test_llamacpp_voice_reference_is_refused_by_a_model_without_a_voice_encoder(monkeypatch, tmp_path):
    """The catalog's Pocket TTS models have predefined voices and no voice encoder: they cannot clone."""
    from arduino.app_bricks.tts import TTSError

    recording = tmp_path / "voice.wav"
    recording.write_bytes(b"RIFF fake voice")
    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(TTSError, match="cannot clone") as refused:
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, voice_reference=recording)
    assert "giovanni" in (refused.value.hint or "")


def test_llamacpp_unreadable_voice_reference_is_an_error(monkeypatch, tmp_path):
    from arduino.app_bricks.tts import TTSError

    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(TTSError, match="nowhere.wav"):
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, voices=[], clones=True, voice_reference=tmp_path / "nowhere.wav")


def test_voice_and_voice_reference_are_exclusive(monkeypatch, tmp_path):
    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(ValueError, match="voice_reference"):
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, clones=True, voice="giovanni", voice_reference=tmp_path / "voice.wav")


def test_llamacpp_model_must_be_served_by_the_runner(monkeypatch):
    no_audio = lambda url, json, **kwargs: float_pcm_response([0.0])  # noqa: E731
    with pytest.raises(RuntimeError, match="llamacpp:pocket-tts-italian"):
        make_llamacpp_tts(monkeypatch, BlockingSpeaker(), no_audio, served=("pocket-tts-english",))


def test_llamacpp_cancel_closes_the_stream_without_a_cancel_request(monkeypatch):
    """llama-server has no cancel endpoint: it stops when the client goes away."""
    speaker = BlockingSpeaker(buffer_size=4)
    responses = []

    def post_response(url, json, **kwargs):
        responses.append(float_pcm_response(np.zeros(4800), 64))
        return responses[-1]

    tts, _ = make_llamacpp_tts(monkeypatch, speaker, post_response)
    speak_thread = threading.Thread(target=tts.speak, args=("hello",), daemon=True)
    speak_thread.start()

    assert speaker.first_chunk_written.wait(timeout=2)
    tts.cancel()  # make_llamacpp_tts fails the test on any POST
    speaker.release_first_chunk.set()
    speak_thread.join(timeout=2)

    assert speak_thread.is_alive() is False
    assert len(speaker.chunks_written) == 1
    assert responses[0].close_called is True


def test_a_voice_is_ignored_by_models_with_a_voice_of_their_own(monkeypatch):
    tts = make_tts(monkeypatch, BlockingSpeaker(buffer_size=4), lambda url, json, **kwargs: FakeResponse(content=b""), voice="giovanni")

    assert tts.api_base_url.startswith("http://audio-analytics-runner:8085/")
    assert tts._voice["model"] == "pipertts_en"


REPO_ROOT = Path(__file__).resolve().parents[4]


def _matching_services(model: str) -> list[str]:
    """The services arduino-app-cli starts for the TTS brick with *model*: "*" is the only wildcard."""
    import fnmatch

    import yaml

    config = yaml.safe_load((REPO_ROOT / "src" / "arduino" / "app_bricks" / "tts" / "brick_config.yaml").read_text())
    return [entry["id"] for entry in config["requires_services"] if fnmatch.fnmatchcase(model, entry["when"]["model"])]


@pytest.mark.parametrize(
    ("model", "service"),
    [("llamacpp:pocket-tts-italian", "arduino:llamacpp_tts"), ("pipertts_en", "arduino:genie_audio"), ("melo-tts-en", "arduino:genie_audio")],
)
def test_the_model_selects_the_service(model, service):
    assert _matching_services(model) == [service]


def test_catalog_llamacpp_tts_ids_are_the_names_the_runner_serves():
    """The runner serves a curated GGUF under its file stem, which is what the brick asks for once the prefix is stripped."""
    import yaml

    catalog = yaml.safe_load((REPO_ROOT / "models" / "models-list.yaml").read_text(encoding="utf-8"))
    ids = []
    for entry in catalog["models"]:
        ((model_id, model),) = entry.items()
        if not model_id.startswith("llamacpp:") or "arduino:tts" not in [brick["id"] for brick in model.get("bricks") or []]:
            continue
        ids.append(model_id)
        for platform in model["deployment"]["platforms"]:
            ((_board, config),) = platform.items()
            variables = config["variables"]
            assert variables["models_repository"] == "llamacpp"
            assert variables["model_url"].rsplit("/", 1)[-1] == model_id.removeprefix("llamacpp:") + ".gguf"
            assert variables["model_mmproj_url"]
    languages = ("english", "french", "german", "italian", "spanish")
    assert {f"llamacpp:pocket-tts-{language}" for language in languages} <= set(ids)


def test_catalog_pocket_tts_models_declare_the_default_voice_of_their_language():
    """Kyutai's defaults (DEFAULT_VOICE_FOR_LANGUAGE in pocket_tts/default_parameters.py), alba for English."""
    import yaml

    catalog = yaml.safe_load((REPO_ROOT / "models" / "models-list.yaml").read_text(encoding="utf-8"))
    defaults = {model_id: (model.get("metadata") or {}).get("default_voice") for entry in catalog["models"] for model_id, model in entry.items()}

    expected = {"english": "alba", "french": "estelle", "german": "juergen", "italian": "giovanni", "spanish": "lola"}
    assert {language: defaults[f"llamacpp:pocket-tts-{language}"] for language in expected} == expected
