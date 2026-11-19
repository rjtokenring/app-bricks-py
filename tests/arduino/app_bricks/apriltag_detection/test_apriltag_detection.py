# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from PIL.Image import Image

from arduino.app_bricks.apriltag_detection import AprilTag, AprilTagDetection, draw_tag

# Tag 0 of each family as published by AprilRobotics (apriltag-imgs), black border included: 0 is black
OFFICIAL_TAG_0 = {
    "tag36h11": ["00000000", "01101010", "00111010", "00110000", "01010000", "00101100", "00001000", "00000000"],
    "tag25h9": ["0000000", "0101010", "0010110", "0100100", "0111110", "0100010", "0000000"],
    "tag16h5": ["000000", "000100", "000110", "000010", "010110", "000000"],
}

MODULE = 12  # pixels per tag module
ORIGIN = 100  # top-left corner of the black border of an unrotated tag


def tag_frame(family: str = "tag36h11", quarter_turns: int = 0, bgr: bool = True) -> np.ndarray:
    """A white 400x400 frame with the official tag 0 of the family, rotated clockwise by quarter turns."""
    bits = np.array([[int(b) for b in row] for row in OFFICIAL_TAG_0[family]], dtype=np.uint8) * 255
    tag = np.kron(np.rot90(bits, -quarter_turns), np.ones((MODULE, MODULE), dtype=np.uint8))
    frame = np.full((400, 400), 255, dtype=np.uint8)
    frame[ORIGIN : ORIGIN + tag.shape[0], ORIGIN : ORIGIN + tag.shape[1]] = tag
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if bgr else frame


def square(family: str) -> np.ndarray:
    """Corners of the black border of an unrotated tag: top-left, top-right, bottom-right, bottom-left.

    They lie on the edge between the border and the white around it, half a pixel out of the centers of
    the outermost black pixels.
    """
    near = ORIGIN - 0.5
    far = near + MODULE * len(OFFICIAL_TAG_0[family])
    return np.array([[near, near], [far, near], [far, far], [near, far]], dtype=float)


def camera_showing(*frames: np.ndarray | None) -> MagicMock:
    camera = MagicMock()
    camera.capture.side_effect = list(frames)
    return camera


@pytest.mark.parametrize("family", ["tag36h11", "tag25h9", "tag16h5"])
def test_official_tag_is_found_with_its_id_and_corners_from_the_top_left(family: str):
    detector = AprilTagDetection(MagicMock(), families=family)

    tags = detector.detect(tag_frame(family))

    assert [(tag.id, tag.family) for tag in tags] == [(0, family)]
    assert tags[0].corners.shape == (4, 2)
    np.testing.assert_allclose(tags[0].corners, square(family), atol=0.3)
    np.testing.assert_allclose(tags[0].center, square(family).mean(axis=0), atol=0.3)


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_corners_follow_the_rotation_of_the_tag(quarter_turns: int):
    detector = AprilTagDetection(MagicMock())

    tags = detector.detect(tag_frame(quarter_turns=quarter_turns))

    # A clockwise quarter turn moves the top-left corner of the tag to the top-right of the image, and so on
    np.testing.assert_allclose(tags[0].corners, np.roll(square("tag36h11"), -quarter_turns, axis=0), atol=0.3)


def test_center_is_where_the_diagonals_cross_on_a_tag_seen_at_an_angle():
    frame = tag_frame()
    src = square("tag36h11").astype(np.float32)
    dst = np.array([[120, 110], [300, 90], [330, 320], [90, 260]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, homography, (400, 400), borderValue=(255, 255, 255))
    true_center = cv2.perspectiveTransform(src.mean(axis=0).reshape(1, 1, 2), homography).reshape(2)

    tags = AprilTagDetection(MagicMock()).detect(warped)

    assert [tag.id for tag in tags] == [0]
    np.testing.assert_allclose(tags[0].corners, dst, atol=2)
    np.testing.assert_allclose(tags[0].center, true_center, atol=2)
    assert np.linalg.norm(dst.mean(axis=0) - true_center) > 5  # The mean of the corners would be off


def test_greyscale_and_read_only_frames_are_accepted():
    frame = tag_frame(bgr=False)
    frame.setflags(write=False)

    assert [tag.id for tag in AprilTagDetection(MagicMock()).detect(frame)] == [0]


def test_frame_without_tags_gives_no_detection():
    assert AprilTagDetection(MagicMock()).detect(np.full((400, 400, 3), 255, dtype=np.uint8)) == []


def test_tag_of_another_family_is_not_reported():
    assert AprilTagDetection(MagicMock(), families="tag25h9").detect(tag_frame("tag36h11")) == []


def test_tags_of_several_families_are_found_in_the_same_frame_each_with_its_family():
    mixed = np.hstack([tag_frame("tag36h11"), tag_frame("tag25h9"), tag_frame("tag16h5", quarter_turns=1)])
    detector = AprilTagDetection(MagicMock(), families=["tag25h9", "tag36h11", "tag16h5"])

    tags = sorted(detector.detect(mixed), key=lambda tag: tag.center[0])

    assert [(tag.id, tag.family) for tag in tags] == [(0, "tag36h11"), (0, "tag25h9"), (0, "tag16h5")]
    np.testing.assert_allclose(tags[1].corners, square("tag25h9") + [400, 0], atol=0.3)
    np.testing.assert_allclose(tags[2].corners, np.roll(square("tag16h5"), -1, axis=0) + [800, 0], atol=0.3)


def test_only_the_listed_families_are_reported():
    mixed = np.hstack([tag_frame("tag36h11"), tag_frame("tag25h9"), tag_frame("tag16h5")])

    tags = AprilTagDetection(MagicMock(), families=["tag36h11", "tag16h5"]).detect(mixed)

    assert sorted(tag.family for tag in tags) == ["tag16h5", "tag36h11"]


@pytest.mark.parametrize("families", ["tagStandard41h12", ["tag36h11", "tagStandard41h12"], []])
def test_unsupported_or_missing_families_are_rejected(families):
    with pytest.raises(ValueError, match="tag36h11, tag25h9, tag16h5"):
        AprilTagDetection(MagicMock(), families=families)


def test_default_camera_is_created_when_none_is_given(monkeypatch: pytest.MonkeyPatch):
    default_camera = MagicMock()
    monkeypatch.setattr("arduino.app_bricks.apriltag_detection.detection.Camera", default_camera)

    AprilTagDetection(camera=None).start()

    default_camera.return_value.start.assert_called_once()


def test_tag_in_view_is_reported_at_every_frame():
    received: list[AprilTag] = []
    detector = AprilTagDetection(camera_showing(tag_frame(), tag_frame()))
    detector.on_detect(lambda frame, tag: received.append(tag))

    detector.loop()
    detector.loop()

    assert [tag.id for tag in received] == [0, 0]


def test_list_callback_receives_all_tags_at_once():
    received: list[list[AprilTag]] = []
    frames: list[Image] = []

    def on_tags(frame: Image, tags: list[AprilTag]) -> None:
        frames.append(frame)
        received.append(tags)

    two_tags = np.hstack([tag_frame(), tag_frame(quarter_turns=1)])
    detector = AprilTagDetection(camera_showing(two_tags))
    detector.on_detect(on_tags)
    detector.loop()

    assert [sorted(tag.center[0] // 400 for tag in tags) for tags in received] == [[0, 1]]
    assert frames[0].size == (800, 400)


def test_on_detect_is_not_called_without_tags_but_on_frame_is():
    on_detect, on_frame = MagicMock(), MagicMock()
    detector = AprilTagDetection(camera_showing(np.full((400, 400, 3), 255, dtype=np.uint8), None))
    detector.on_detect(on_detect)
    detector.on_frame(on_frame)

    detector.loop()
    detector.loop()  # The camera has no frame

    on_detect.assert_not_called()
    assert on_frame.call_count == 1


def test_removed_callback_is_no_longer_called():
    on_detect = MagicMock()
    detector = AprilTagDetection(camera_showing(tag_frame()))
    detector.on_detect(on_detect)
    detector.on_detect(None)

    detector.loop()

    on_detect.assert_not_called()


def test_camera_and_callback_errors_reach_on_error():
    errors: list[Exception] = []
    camera = MagicMock()
    camera.capture.side_effect = [RuntimeError("camera unplugged"), tag_frame()]
    detector = AprilTagDetection(camera)
    detector.on_error(errors.append)
    detector.on_detect(MagicMock(side_effect=ValueError("bad callback")))

    detector.loop()
    detector.loop()

    assert [str(e) for e in errors] == ["camera unplugged", "bad callback"]


def test_draw_tag_outlines_the_tag_and_marks_its_first_corner():
    frame = PILImage.new("RGB", (400, 400))
    corners = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.float32)

    annotated = draw_tag(frame, AprilTag(7, "tag36h11", corners, (200.0, 200.0)))

    assert annotated is frame
    assert annotated.getpixel((200, 100)) == (0, 255, 0)
    assert annotated.getpixel((100, 200)) == (0, 255, 0)
    assert annotated.getpixel((100, 100)) == (255, 0, 0)
    assert annotated.getpixel((300, 300)) == (0, 255, 0)
    assert annotated.crop((170, 170, 230, 230)).getbbox() is not None  # The id is written at the center
    assert annotated.getpixel((150, 250)) == (0, 0, 0)
