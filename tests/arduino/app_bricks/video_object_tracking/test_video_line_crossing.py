# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from arduino.app_bricks.video_object_tracking import VideoObjectTracking


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Mock external dependencies in __init__.

    This is needed to avoid network calls and other side effects.
    """
    fake_compose = {"services": {"models-runner": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:${BIND_PORT:-8100}:8100"]}}}
    monkeypatch.setattr("arduino.app_internal.core.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda x: [(None, None), (None, "8100")])


@pytest.fixture
def detector() -> VideoObjectTracking:
    """Fixture to create an instance of VideoObjectTracking.

    Returns:
        VideoObjectTracking: An instance of the VideoObjectTracking class.
    """
    return VideoObjectTracking()


def test_horizontal_line_crossing(detector: VideoObjectTracking):
    """Test the detect method with valid inputs.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    detector.set_horizontal_crossing_line(y=200)

    detector._record_line_crossing(detected_object_label="person", object_id=1, x=100, y=150)
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=150)

    detections = detector.get_line_crossing_counts()
    assert "person" not in detections

    # crossing down

    detector._record_line_crossing(detected_object_label="person", object_id=1, x=100, y=250)
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=250)

    detections = detector.get_line_crossing_counts()
    assert "person" in detections
    assert detections["person"] == 1

    # no crossing down

    detector._record_line_crossing(detected_object_label="person", object_id=1, x=100, y=300)
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=300)

    detections = detector.get_line_crossing_counts()
    assert "person" in detections
    assert detections["person"] == 1

    ### crossing up!

    detector._record_line_crossing(detected_object_label="person", object_id=1, x=100, y=150)
    detector._record_object(detected_object_label="person", object_id=1, x=100, y=150)

    detections = detector.get_line_crossing_counts()
    assert "person" in detections
    assert detections["person"] == 2


def test_vertical_line_crossing(detector: VideoObjectTracking):
    """Test the detect method with valid inputs.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    detector.set_vertical_crossing_line(x=200)

    detector._record_line_crossing(detected_object_label="car", object_id=1, x=150, y=100)
    detector._record_object(detected_object_label="car", object_id=1, x=150, y=100)

    detections = detector.get_line_crossing_counts()
    assert "car" not in detections

    # crossing right

    detector._record_line_crossing(detected_object_label="car", object_id=1, x=250, y=100)
    detector._record_object(detected_object_label="car", object_id=1, x=250, y=100)

    detections = detector.get_line_crossing_counts()
    assert "car" in detections
    assert detections["car"] == 1

    # no crossing right

    detector._record_line_crossing(detected_object_label="car", object_id=1, x=300, y=100)
    detector._record_object(detected_object_label="car", object_id=1, x=300, y=100)

    detections = detector.get_line_crossing_counts()
    assert "car" in detections
    assert detections["car"] == 1

    ### crossing left!

    detector._record_line_crossing(detected_object_label="car", object_id=1, x=150, y=100)
    detector._record_object(detected_object_label="car", object_id=1, x=150, y=100)

    detections = detector.get_line_crossing_counts()
    assert "car" in detections
    assert detections["car"] == 2


def test_diagonal_line_crossing(detector: VideoObjectTracking):
    """Test the detect method with valid inputs.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    detector.set_crossing_line_coordinates(x1=0, y1=0, x2=200, y2=200)  # letter-box image

    detector._record_line_crossing(detected_object_label="dog", object_id=1, x=50, y=30)  # below line
    detector._record_object(detected_object_label="dog", object_id=1, x=50, y=30)

    detections = detector.get_line_crossing_counts()
    assert "dog" not in detections

    # crossing diagonal

    detector._record_line_crossing(detected_object_label="dog", object_id=1, x=200, y=220)  # above line
    detector._record_object(detected_object_label="dog", object_id=1, x=200, y=220)

    detections = detector.get_line_crossing_counts()
    assert "dog" in detections
    assert detections["dog"] == 1


def test_no_line_set(detector: VideoObjectTracking):
    """Test the detect method with valid inputs.

    Args:
        detector (VideoObjectTracking): An instance of the VideoObjectTracking class.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture to mock external dependencies.
    """

    detector._record_line_crossing(detected_object_label="bicycle", object_id=1, x=150, y=100)
    detector._record_object(detected_object_label="bicycle", object_id=1, x=150, y=100)

    detector._record_line_crossing(detected_object_label="bicycle", object_id=1, x=150, y=200)
    detector._record_object(detected_object_label="bicycle", object_id=1, x=150, y=100)

    detections = detector.get_line_crossing_counts()
    assert "bicycle" not in detections
