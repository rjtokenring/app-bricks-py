# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from typing import Callable
from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import json
from collections import Counter, OrderedDict
import threading

logger = Logger("VideoObjectTracking")


class LRUDict(OrderedDict):
    """A dictionary-like object with a fixed size
    that evicts the least recently used items.
    """

    def __init__(self, maxsize=128, *args, **kwargs):
        self.maxsize = maxsize
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)

        super().__setitem__(key, value)

        if len(self) > self.maxsize:
            # Evict the least recently used item (the first item)
            self.popitem(last=False)


@brick
class VideoObjectTracking(VideoObjectDetection):
    """Module for object tracking on a **live video stream** using a specified machine learning model.

    This brick:
      - Connects to a model runner over WebSocket.
      - Parses incoming classification messages with bounding boxes.
      - Filters detections by a configurable confidence threshold.
      - Debounces repeated triggers of the same label.
      - Invokes per-label callbacks and/or a catch-all callback.
    """

    def __init__(self, confidence: float = 0.3, debounce_sec: float = 0.0, labels_to_track: list[str] = None):
        """Initialize the VideoObjectDetection class.

        Args:
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            labels_to_track (list[str], optional): List of labels to track. If None, all labels are tracked.

        Raises:
            RuntimeError: If the host address could not be resolved.
        """
        super().__init__(confidence=confidence, debounce_sec=debounce_sec)
        self._labels_to_track = labels_to_track

        # Counter for tracked objects
        self._counter_lock = threading.Lock()
        self._object_counters = Counter()
        self._recent_objects = LRUDict(maxsize=100)  # To track recent object IDs and their labels

    def _record_object(self, detected_object: str, object_id: float):
        """Record that an object with a specific label and ID has been seen."""

        with self._counter_lock:
            if object_id in self._recent_objects:
                return  # Already recorded recently

            self._recent_objects[object_id] = detected_object
            self._object_counters[detected_object] += 1

    def get_tracked_objects(self) -> dict:
        """Get the current counts of tracked objects by label.

        Returns:
            dict: A dictionary with labels as keys and their respective counts as values.
        """
        with self._counter_lock:
            return dict(self._object_counters)

    def reset_counters(self):
        """Reset the counts of tracked objects."""
        with self._counter_lock:
            self._object_counters.clear()
            self._recent_objects.clear()

    def on_detect(self, object: str, callback: Callable[[], None]):
        """Register a callback invoked when a **specific label** is detected.

        Args:
            object (str): The label of the object to check for in the classification results.
            callback (Callable[[], None]): A function with **no parameters**.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` accepts any parameters.
        """
        super().on_detect(object, callback)

    def on_detect_all(self, callback: Callable[[dict], None]):
        """Register a callback invoked for **every detection event**.

        This is useful to receive a consolidated dictionary of detections for each frame.

        Args:
            callback (Callable[[dict], None]): A function that accepts **one dict argument** with
                the shape `{label: confidence, ...}`.

        Raises:
            TypeError: If `callback` is not a function.
            ValueError: If `callback` does not accept exactly one argument.
        """
        super().on_detect_all(callback)

    def start(self):
        """Start the video object detection process."""
        super().start()

    def stop(self):
        """Stop the video object detection process."""
        super().stop()

    def execute(self):
        """Connect to the model runner and process messages until `stop` is called.

        Behavior:
            - Establishes a WebSocket connection to the runner.
            - Parses ``"hello"`` messages to capture model metadata and optionally
              performs a threshold override to align the runner with the local setting.
            - Parses ``"classification"`` messages, filters detections by confidence,
              applies debounce, then invokes registered callbacks.
            - Retries on transient WebSocket errors while running.

        Exceptions:
            ConnectionClosedOK:
                Propagated to exit cleanly when the server closes the connection.
            ConnectionClosedError, TimeoutError, ConnectionRefusedError:
                Logged and retried with a short backoff while running.
        """
        while self._is_running.is_set():
            try:
                with connect(self._uri) as ws:
                    while self._is_running.is_set():
                        try:
                            message = ws.recv()
                            if not message:
                                continue
                            self._process_message(ws, message)
                        except ConnectionClosedOK:
                            raise
                        except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                            logger.warning(f"Connection lost. Retrying...")
                            raise
                        except Exception as e:
                            logger.exception(f"Failed to process detection: {e}")
            except ConnectionClosedOK:
                logger.debug(f"Disconnected cleanly, exiting WebSocket read loop.")
                return
            except (TimeoutError, ConnectionRefusedError, ConnectionClosedError):
                logger.debug(f"Waiting for model runner. Retrying...")
                import time

                time.sleep(2)
                continue
            except Exception as e:
                logger.exception(f"Failed to establish WebSocket connection to {self._host}: {e}")

    def _process_message(self, ws: ClientConnection, message: str):
        jmsg = json.loads(message)
        if jmsg.get("type") == "hello":
            # Parse hello message to extract model info if needed
            logger.debug(f"Connected to model runner: {jmsg}")
            try:
                self._model_info = EdgeImpulseRunnerFacade.parse_model_info_message(jmsg)
                if self._model_info and self._model_info.thresholds is not None:
                    self._override_threshold(ws, self._confidence)

            except Exception as e:
                logger.error(f"Error parsing WS hello message: {e}")
            return

        elif jmsg.get("type") == "handling-message-success":
            # Ignore handling-message-success messages
            return

        elif jmsg.get("type") == "classification":
            result = jmsg.get("result", {})
            if not isinstance(result, dict):
                return

            bounding_boxes = result.get("object_tracking", [])
            if bounding_boxes:
                if len(bounding_boxes) == 0:
                    return

                # Process each bounding box
                detections = {}
                for box in bounding_boxes:
                    detected_object = box.get("label")
                    if detected_object is None:
                        continue

                    object_id = box.get("object_id", 0.0)
                    if object_id <= 0.0:
                        continue

                    # Extract bounding box coordinates if needed
                    xyxy_bbox = (
                        box.get("x", 0),
                        box.get("y", 0),
                        box.get("x", 0) + box.get("width", 0),
                        box.get("y", 0) + box.get("height", 0),
                    )

                    detection_details = {"object_id": object_id, "bounding_box_xyxy": xyxy_bbox}
                    detections[detected_object] = detection_details

                    self._record_object(detected_object=detected_object, object_id=object_id)

                    # Check if the class_id matches any registered handlers
                    super()._execute_handler(detection=detected_object, detection_details=detection_details)

                if len(detections) > 0:
                    # If there are detections, invoke the all-detection handler
                    super()._execute_global_handler(detections=detections)

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def override_threshold(self, value: float):
        """Override the threshold for object detection model.

        Args:
            value (float): The new value for the threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        super().override_threshold(value)
