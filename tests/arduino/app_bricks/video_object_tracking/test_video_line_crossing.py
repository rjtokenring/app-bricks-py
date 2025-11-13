# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from arduino.app_bricks.video_object_tracking import VideoObjectTracking


class FakeWS:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def send(self, *args, **kwargs):
        pass

    def recv(self, *args, **kwargs):
        return '{"type": "hello"}'  # or any test message


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Mock external dependencies in __init__.

    This is needed to avoid network calls and other side effects.
    """
    fake_compose = {"services": {"models-runner": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"]}}}
    monkeypatch.setattr("arduino.app_internal.core.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda x: [(None, None), (None, "8100")])
    monkeypatch.setattr("websockets.sync.client.connect", lambda *args, **kwargs: FakeWS())


@pytest.fixture
def detector() -> VideoObjectTracking:
    """Fixture to create an instance of VideoObjectTracking.

    Returns:
        VideoObjectTracking: An instance of the VideoObjectTracking class.
    """
    return VideoObjectTracking()


def test_horizontal_line_crossing(detector: VideoObjectTracking):
    """Test the _record_line_crossing method with horizontal line crossing.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """

    detector.set_horizontal_crossing_line(y=200)

    detector._record_object(detected_object_label="person", object_id=1, x=100, y=150)
    detections = detector.get_line_crossing_counts()
    assert "person" not in detections

    # up
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=250)
    detections = detector.get_line_crossing_counts()
    assert "person" in detections
    assert detections["person"] == 1

    # down
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=100)
    detections = detector.get_line_crossing_counts()
    assert "person" in detections
    assert detections["person"] == 2


def test_vertical_line_crossing(detector: VideoObjectTracking):
    """Test the _record_line_crossing method with vertical line crossing.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """

    detector.set_vertical_crossing_line(x=200)

    detector._record_object(detected_object_label="car", object_id=1, x=150, y=100)
    detections = detector.get_line_crossing_counts()
    assert "car" not in detections

    # right
    detector._record_object(detected_object_label="car", object_id=1, x=250, y=100)
    detections = detector.get_line_crossing_counts()
    assert "car" in detections
    assert detections["car"] == 1

    # left
    detector._record_object(detected_object_label="car", object_id=1, x=100, y=100)
    detections = detector.get_line_crossing_counts()
    assert "car" in detections
    assert detections["car"] == 2


def test_diagonal_line_crossing(detector: VideoObjectTracking):
    """Test the _record_line_crossing method with diagonal line crossing.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """

    detector.set_crossing_line_coordinates(x1=0, y1=0, x2=200, y2=200)

    detector._record_object(detected_object_label="dog", object_id=1, x=50, y=30)
    detections = detector.get_line_crossing_counts()
    assert "dog" not in detections

    # diagonal up
    detector._record_object(detected_object_label="dog", object_id=1, x=200, y=220)
    detections = detector.get_line_crossing_counts()
    assert "dog" in detections
    assert detections["dog"] == 1

    # diagonal down
    detector._record_object(detected_object_label="dog", object_id=1, x=50, y=30)
    detections = detector.get_line_crossing_counts()
    assert "dog" in detections
    assert detections["dog"] == 2


def test_no_line_set(detector: VideoObjectTracking):
    """Test the _record_line_crossing method when no line is set.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """

    detector._record_object(detected_object_label="bicycle", object_id=1, x=100, y=50)
    detector._record_object(detected_object_label="bicycle", object_id=1, x=150, y=200)
    detections = detector.get_line_crossing_counts()
    # since no line is set means the x1,y1 and x2,y2 are all zero, so no crossing can be detected
    assert "bicycle" not in detections


def test_record_object(detector: VideoObjectTracking):
    """Test the _record_object method.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """

    detector._record_object(detected_object_label="cat", object_id=1, x=100, y=100)
    unique_count = detector.get_unique_objects_count()
    assert unique_count == {"cat": 1}

    # same object_id should not increase count
    detector._record_object(detected_object_label="cat", object_id=1, x=100, y=100)
    unique_count = detector.get_unique_objects_count()
    assert unique_count == {"cat": 1}

    # same label, different object_id should increase count
    detector._record_object(detected_object_label="cat", object_id=2, x=150, y=150)
    unique_count = detector.get_unique_objects_count()
    assert unique_count == {"cat": 2}

    detector._record_object(detected_object_label="dog", object_id=3, x=200, y=200)
    unique_count = detector.get_unique_objects_count()
    assert unique_count == {"cat": 2, "dog": 1}


def test_reset_counters(detector: VideoObjectTracking):
    """Test the reset_counters method.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """
    detector._record_object(detected_object_label="cat", object_id=1, x=0, y=-1)
    detector._record_object(detected_object_label="cat", object_id=1, x=150, y=200)

    unique_count = detector.get_unique_objects_count()
    assert unique_count == {"cat": 1}

    line_crossings = detector.get_line_crossing_counts()
    assert "cat" in line_crossings
    assert line_crossings["cat"] == 1

    detector.reset_counters()

    unique_count = detector.get_unique_objects_count()
    assert unique_count == {}

    line_crossings = detector.get_line_crossing_counts()
    assert line_crossings == {}


def test_get_objects_directions(detector: VideoObjectTracking):
    """Test the get_objects_directions method.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
    """
    detector._record_object(detected_object_label="person", object_id=1, x=50, y=50)
    detector._record_object(detected_object_label="person", object_id=1, x=50, y=150)  # down
    detector._record_object(detected_object_label="person", object_id=1, x=50, y=50)  # up
    detector._record_object(detected_object_label="person", object_id=1, x=70, y=50)  # left
    detector._record_object(detected_object_label="person", object_id=1, x=50, y=50)  # right
    detector._record_object(detected_object_label="person", object_id=1, x=30, y=70)  # down-right
    detector._record_object(detected_object_label="person", object_id=1, x=10, y=50)  # up-right
    detector._record_object(detected_object_label="person", object_id=1, x=30, y=70)  # down-left
    detector._record_object(detected_object_label="person", object_id=1, x=80, y=20)  # up-left

    directions = detector.get_objects_directions()
    assert directions[1] == ["down", "up", "left", "right", "down-right", "up-right", "down-left", "up-left"]
