# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from arduino.app_bricks.image_segmentation import ImageSegmentation, Segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_META_PERSON = {"person_detected": True, "person_ratio": 0.3, "confidence": 0.9, "bounding_box_xyxy": [10, 20, 110, 220]}
_META_SMALL = {"person_detected": True, "person_ratio": 0.01, "confidence": 0.8, "bounding_box_xyxy": [0, 0, 5, 5]}
_META_NONE = {"person_detected": False, "person_ratio": 0.0, "confidence": 0.0, "bounding_box_xyxy": None}

WAIT_TIMEOUT = 2.0  # seconds – maximum time to wait for a callback to complete


def _wait(event: threading.Event, msg: str = ""):
    assert event.wait(timeout=WAIT_TIMEOUT), f"Timed out waiting for: {msg}"


def _make(monkeypatch: pytest.MonkeyPatch, **kwargs) -> ImageSegmentation:
    monkeypatch.setattr(
        "arduino.app_bricks.image_segmentation.image_segmentation.load_brick_compose_file",
        lambda cls: {"services": {"image_segmentation": {}}},
    )
    monkeypatch.setattr(
        "arduino.app_bricks.image_segmentation.image_segmentation.resolve_address",
        lambda host: "127.0.0.1",
    )
    instance = ImageSegmentation(camera=MagicMock(), **kwargs)
    # Provide a real executor so callbacks actually run in threads
    instance._executor = ThreadPoolExecutor(max_workers=4)
    instance._is_running = True
    return instance


@pytest.fixture()
def seg(monkeypatch: pytest.MonkeyPatch):
    instance = _make(monkeypatch)
    yield instance
    instance._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Enter / exit
# ---------------------------------------------------------------------------


class TestEnterExit:
    def test_enter_and_exit_fire_once_per_edge(self, seg: ImageSegmentation):
        events: list[str] = []
        done = threading.Event()
        seg.on_enter(lambda: events.append("enter"))
        seg.on_exit(lambda: (events.append("exit"), done.set()))

        for metadata in (_META_PERSON, _META_PERSON, _META_NONE, _META_NONE):
            seg._process_detection(metadata)
            time.sleep(0.05)  # let each callback finish so none is dropped as busy

        _wait(done, "exit")
        assert events == ["enter", "exit"]
        assert seg.person_present is False

    def test_exit_waits_for_the_debounce(self, monkeypatch: pytest.MonkeyPatch):
        seg = _make(monkeypatch, exit_debounce_sec=0.2)
        exited = threading.Event()
        seg.on_exit(exited.set)

        seg._process_detection(_META_PERSON)
        seg._process_detection(_META_NONE)
        assert seg.person_present is True  # a single empty frame is flicker
        seg._process_detection(_META_PERSON)
        seg._process_detection(_META_NONE)
        time.sleep(0.25)
        seg._process_detection(_META_NONE)

        _wait(exited, "debounced exit")
        assert seg.person_present is False
        seg._executor.shutdown(wait=True)

    def test_regions_below_min_person_ratio_are_nobody(self, monkeypatch: pytest.MonkeyPatch):
        seg = _make(monkeypatch, min_person_ratio=0.05)
        entered = threading.Event()
        seg.on_enter(entered.set)
        seg.on_segmentation(lambda s: entered.set())

        seg._process_detection(_META_SMALL)
        time.sleep(0.1)

        assert not entered.is_set()
        assert seg.person_present is False
        seg._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Segmentation payload
# ---------------------------------------------------------------------------


class TestSegmentationCallback:
    def test_receives_the_parsed_segmentation(self, seg: ImageSegmentation):
        received: list[Segmentation] = []
        done = threading.Event()
        seg.on_segmentation(lambda s: (received.append(s), done.set()))

        seg._process_detection(_META_PERSON)

        _wait(done, "segmentation")
        assert received == [Segmentation(person_ratio=0.3, confidence=0.9, bounding_box_xyxy=(10, 20, 110, 220))]

    def test_not_called_without_people(self, seg: ImageSegmentation):
        called = threading.Event()
        seg.on_segmentation(lambda s: called.set())

        seg._process_detection(_META_NONE)
        seg._process_detection({})
        time.sleep(0.1)

        assert not called.is_set()

    def test_busy_callback_drops_events(self, seg: ImageSegmentation):
        release = threading.Event()
        calls: list[Segmentation] = []

        def slow(s: Segmentation):
            calls.append(s)
            release.wait(WAIT_TIMEOUT)

        seg.on_segmentation(slow)
        for _ in range(5):
            seg._process_detection(_META_PERSON)
        release.set()
        time.sleep(0.1)

        assert len(calls) == 1

    def test_malformed_metadata_reports_an_error(self, seg: ImageSegmentation):
        errors: list[Exception] = []
        done = threading.Event()
        seg.on_error(lambda e: (errors.append(e), done.set()))

        seg._process_detection({"person_detected": True, "bounding_box_xyxy": [1, 2]})

        _wait(done, "error")
        assert isinstance(errors[0], IndexError)

    def test_callback_exception_goes_to_on_error(self, seg: ImageSegmentation):
        done = threading.Event()
        errors: list[Exception] = []
        seg.on_error(lambda e: (errors.append(e), done.set()))

        def boom(s: Segmentation):
            raise RuntimeError("boom")

        seg.on_segmentation(boom)
        seg._process_detection(_META_PERSON)

        _wait(done, "error from callback")
        assert str(errors[0]) == "boom"

    def test_unregistered_callback_is_not_called(self, seg: ImageSegmentation):
        called = threading.Event()
        seg.on_segmentation(lambda s: called.set())
        seg.on_segmentation(None)

        seg._process_detection(_META_PERSON)
        time.sleep(0.1)

        assert not called.is_set()


# ---------------------------------------------------------------------------
# Settings forwarded to the model runner
# ---------------------------------------------------------------------------


class TestRunnerConfig:
    def test_defaults(self, seg: ImageSegmentation):
        assert seg._runner_config() == {"mask_threshold": 0.5, "background_color": [68, 132, 255], "background_opacity": 1.0}

    def test_setters_change_the_config(self, seg: ImageSegmentation):
        seg.set_confidence(0.7)
        seg.set_background_color((0, 255, 0))
        seg.set_background_opacity(0.4)

        assert seg._runner_config() == {"mask_threshold": 0.7, "background_color": [0, 255, 0], "background_opacity": 0.4}

    @pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.5", None])
    def test_unit_values_out_of_range_are_rejected(self, seg: ImageSegmentation, value):
        with pytest.raises(ValueError):
            seg.set_confidence(value)
        with pytest.raises(ValueError):
            seg.set_background_opacity(value)

    @pytest.mark.parametrize("color", [(0, 0), (0, 0, 256), (0, -1, 0), (0.5, 0, 0), "red", (True, 0, 0)])
    def test_bad_colors_are_rejected(self, seg: ImageSegmentation, color):
        with pytest.raises(ValueError):
            seg.set_background_color(color)

    def test_constructor_validates(self, monkeypatch: pytest.MonkeyPatch):
        with pytest.raises(ValueError):
            _make(monkeypatch, confidence=2.0)
        with pytest.raises(ValueError):
            _make(monkeypatch, exit_debounce_sec=-1)
        with pytest.raises(ValueError):
            _make(monkeypatch, background_color=(1, 2))
