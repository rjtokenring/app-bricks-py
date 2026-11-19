# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass
import inspect
import threading
from collections.abc import Callable, Sequence
from typing import Literal, TypeIs

import cv2
import numpy as np
from PIL.Image import Image

from arduino.app_peripherals.camera import Camera, BaseCamera
from arduino.app_utils.image import numpy_to_pil
from arduino.app_utils import brick, Logger

logger = Logger("AprilTagDetection")

type Family = Literal["tag36h11", "tag25h9", "tag16h5"]

_DICTIONARIES: dict[str, int] = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


@dataclass
class AprilTag:
    """An AprilTag found in a video frame.

    Attributes:
        id (int): The number encoded by the tag, unique within its family.
        family (str): The family the tag belongs to, e.g. "tag36h11".
        corners (np.ndarray): A NumPy array of shape (4, 2) with the (x, y) pixel coordinates of the corners of the
            black border of the tag. The order follows the tag, not the image: top-left, top-right, bottom-right and
            bottom-left corner of the tag as it is printed upright, so the first corner tells how the tag is rotated.
        center (tuple[float, float]): The (x, y) pixel coordinates of the center of the tag.
    """

    id: int
    family: str
    corners: np.ndarray
    center: tuple[float, float]


def _expects_list(
    callback: Callable[[Image, list[AprilTag]], None] | Callable[[Image, AprilTag], None],
) -> TypeIs[Callable[[Image, list[AprilTag]], None]]:
    """Tell an on_detect callback taking all the tags at once by the annotation of its second parameter."""
    params = list(inspect.signature(callback).parameters.values())
    return len(params) >= 2 and params[1].annotation == list[AprilTag]


def _expects_single(
    callback: Callable[[Image, list[AprilTag]], None] | Callable[[Image, AprilTag], None],
) -> TypeIs[Callable[[Image, AprilTag], None]]:
    """Tell an on_detect callback taking one tag per call, the default for any other annotation."""
    return not _expects_list(callback)


def _center(corners: np.ndarray) -> tuple[float, float]:
    """The point where the diagonals of the tag cross, which is its center seen from any angle."""
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = corners.tolist()
    ux, uy = cx - ax, cy - ay  # Diagonal from the first corner
    vx, vy = dx - bx, dy - by  # Diagonal from the second corner
    t = ((bx - ax) * vy - (by - ay) * vx) / (ux * vy - uy * vx)
    return ax + t * ux, ay + t * uy


@brick
class AprilTagDetection:
    """Finds AprilTag fiducial markers in a camera video feed.

    Every frame is scanned and all the tags in it are reported, frame after frame, each with its id and the
    position of its corners.

    Args:
        camera (BaseCamera): The camera instance to use for capturing video. If None, a default camera will be initialized.
        families (str | Sequence[str]): The tag family to look for, or a list of them when tags of different
            families share the scene: "tag36h11", "tag25h9" and "tag16h5". Defaults to "tag36h11".

    Raises:
        ValueError: If no family is given or one of them is not supported.
    """

    def __init__(self, camera: BaseCamera | None = None, families: Family | Sequence[Family] = "tag36h11") -> None:
        """Initialize the AprilTagDetection brick."""
        names = tuple(dict.fromkeys([families] if isinstance(families, str) else families))
        unsupported = [name for name in names if name not in _DICTIONARIES]
        if not names or unsupported:
            raise ValueError(f"Unsupported tag families {unsupported or list(names)}, use one or more of: {', '.join(_DICTIONARIES)}.")

        self._camera = camera if camera else Camera()
        self._families = names
        parameters = cv2.aruco.DetectorParameters()
        # Puts the corners on the edge of the black border instead of on its outermost pixels
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionaries = [cv2.aruco.getPredefinedDictionary(_DICTIONARIES[name]) for name in names]
        self._detector = cv2.aruco.ArucoDetector(dictionaries, parameters)

        # These callbacks don't require locking as long as we're running on CPython
        self._on_frame_cb: Callable[[Image], None] | None = None
        self._on_error_cb: Callable[[Exception], None] | None = None

        self._on_detect_list_cb: Callable[[Image, list[AprilTag]], None] | None = None
        self._on_detect_single_cb: Callable[[Image, AprilTag], None] | None = None
        self._on_detect_cb_lock = threading.Lock()  # Synchronizes access to both callbacks

    def start(self) -> None:
        """Start the detector and begin scanning for tags."""
        self._camera.start()

    def stop(self) -> None:
        """Stop the detector and release resources."""
        self._camera.stop()

    def on_detect(self, callback: Callable[[Image, list[AprilTag]], None] | Callable[[Image, AprilTag], None] | None) -> None:
        """Registers or removes a callback to be triggered when tags are found in a frame.

        The callback is invoked for every frame with at least one tag in it, so a tag that stays in view is
        reported again at each frame. The callback function should accept the Image frame and a list[AprilTag]
        or an AprilTag. If the former is used, it will receive all the tags of the frame at once. If the latter
        is used, it will be called once for each tag. If None is provided, the callback will be removed.

        Args:
            callback (Callable[[Image, list[AprilTag]], None]): A callback that will be called with all the tags
                                                                of a frame.
            callback (Callable[[Image, AprilTag], None]): A callback that will be called with a single tag.
            callback (None): To unregister the current callback, if any.

        Example:
            def on_tag(frame: Image, tag: AprilTag):
                print(f"Tag {tag.id} at {tag.center}")

            detector.on_detect(on_tag)
        """
        with self._on_detect_cb_lock:
            self._on_detect_list_cb = callback if callback is not None and _expects_list(callback) else None
            self._on_detect_single_cb = callback if callback is not None and _expects_single(callback) else None

    def on_frame(self, callback: Callable[[Image], None] | None) -> None:
        """Registers a callback function to be called when a new camera frame is captured.

        The callback function should accept the Image frame.
        If None is provided, the callback is removed.

        Args:
            callback (Callable[[Image], None]): A callback that will be called with each captured frame.
            callback (None): Signals to remove the current callback, if any.
        """
        self._on_frame_cb = callback

    def on_error(self, callback: Callable[[Exception], None] | None) -> None:
        """Registers a callback function to be called when an error occurs in the detector.

        The callback function should accept the exception as an argument.
        If None is provided, the callback is removed.

        Args:
            callback (Callable): A callback that will be called with the exception raised in the detector.
            callback (None): Signals to remove the current callback, if any.
        """
        self._on_error_cb = callback

    def detect(self, frame: np.ndarray) -> list[AprilTag]:
        """Find the tags of the configured families in a single image.

        Args:
            frame (np.ndarray): The image to scan, greyscale or BGR, as returned by the camera.

        Returns:
            list[AprilTag]: The tags found, empty if there is none.
        """
        quads, ids, _, families = self._detector.detectMarkersMultiDict(frame)
        tags: list[AprilTag] = []
        if len(quads) == 0:
            return tags
        for quad, tag_id, family in zip(quads, np.asarray(ids).flatten(), np.asarray(families).flatten()):
            # OpenCV keeps the AprilTag codes rotated by half a turn with respect to the official tag images
            corners = np.roll(quad.reshape(4, 2), 2, axis=0)
            tags.append(AprilTag(int(tag_id), self._families[int(family)], corners, _center(corners)))
        return tags

    def loop(self) -> None:
        """Main loop to capture frames and detect tags."""
        try:
            frame = self._camera.capture()
            if frame is None:
                return
        except Exception as e:
            self._on_error(e)
            return

        pil_frame = numpy_to_pil(frame)
        self._on_frame(pil_frame)

        try:
            tags = self.detect(frame)
        except Exception as e:
            logger.error(f"Failed to detect tags: {e}")
            self._on_error(e)
            return
        self._on_detect(pil_frame, tags)

    def _on_frame(self, frame: Image) -> None:
        if self._on_frame_cb:
            try:
                self._on_frame_cb(frame)
            except Exception as e:
                logger.error(f"Failed to run on_frame callback: {e}")
                self._on_error(e)

    def _on_detect(self, frame: Image, tags: list[AprilTag]) -> None:
        with self._on_detect_cb_lock:
            if len(tags) > 0:
                try:
                    if self._on_detect_list_cb is not None:
                        self._on_detect_list_cb(frame, tags)
                    elif self._on_detect_single_cb is not None:
                        for tag in tags:
                            self._on_detect_single_cb(frame, tag)
                except Exception as e:
                    logger.error(f"Failed to run on_detect callback: {e}")
                    self._on_error(e)

    def _on_error(self, exception: Exception) -> None:
        if self._on_error_cb:
            try:
                self._on_error_cb(exception)
            except Exception as e:
                logger.exception(f"Failed to run on_error callback: {e}")
