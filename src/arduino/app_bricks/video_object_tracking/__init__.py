# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import brick, Logger, LRUDict
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
from arduino.app_internal.core import EdgeImpulseRunnerFacade
from typing import Callable
from websockets.sync.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
import json
from collections import Counter
import threading

logger = Logger("VideoObjectTracking")


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

    def __init__(
        self,
        confidence: float = 0.5,
        keep_grace: int = 3,
        max_observations: int = 3,
        iou_threshold: float = 0.1,
        debounce_sec: float = 0.0,
        labels_to_track: list[str] = None,
    ):
        """Initialize the VideoObjectDetection class.

        Args:
            confidence (float): Confidence level for detection. Default is 0.3 (30%).
            debounce_sec (float): Minimum seconds between repeated detections of the same object. Default is 0 seconds.
            keep_grace (int): Number of frames to keep an object if it disappears. Default is 3.
            max_observations (int): Maximum number of observations to consider. Default is 3
            iou_threshold (float): Intersection over Union threshold for tracking. Default is 0.1.
            labels_to_track (list[str], optional): List of labels to track. If None, all labels are tracked.

        Raises:
            RuntimeError: If the host address could not be resolved.
        """
        super().__init__(confidence=confidence, debounce_sec=debounce_sec)
        self._labels_to_track = labels_to_track
        self._max_observations = max_observations
        self._keep_grace = keep_grace
        self._iou_threshold = iou_threshold

        # Counter for tracked objects
        self._counter_lock = threading.Lock()
        self._object_counters = Counter()
        # Map of recent object IDs to their last seen positions (x, y)
        self._recent_objects = LRUDict(maxsize=150)  # To track recent object IDs and their labels

        # Crossing line coordinates
        self._line_coordinates = (0, 0, 0, 0)  # x1, y1, x2, y2
        self._crossing_line_object_counters = Counter()

    def _is_label_enabled(self, label: str) -> bool:
        """Check if a label is enabled for tracking.

        Args:
            label (str): The label to check.
        Returns:
            bool: True if the label is enabled for tracking, False otherwise.
        """
        if self._labels_to_track is None:
            return True
        return label in self._labels_to_track

    def _record_object(self, detected_object_label: str, object_id: float, x: int, y: int):
        """Record that an object with a specific label and ID has been seen.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (float): The unique ID of the detected object.
            x (int): The x-coordinate of the detected object.
            y (int): The y-coordinate of the detected object.
        """

        logger.debug(f"Recording object: label={detected_object_label}, id={object_id}, x={x}, y={y}")
        if not self._is_label_enabled(detected_object_label):
            # Discard if the label is not enabled for tracking
            logger.debug(f"Label {detected_object_label} is not enabled for tracking. Discarding.")
            return

        with self._counter_lock:
            logger.debug(f"Current object counters before recording: {self._object_counters}")
            if object_id not in self._recent_objects:
                logger.debug(f"New object ID {object_id} detected for label {detected_object_label}. Incrementing counter.")
                self._object_counters[detected_object_label] += 1

        self._record_line_crossing(detected_object_label, object_id, x, y)

    def _record_line_crossing(self, detected_object_label: str, object_id: float, x: int, y: int):
        """Record that an object with a specific label and ID has crossed the line.

        Args:
            detected_object_label (str): The label of the detected object.
            object_id (float): The unique ID of the detected object.
            x (int): The x-coordinate of the detected object.
            y (int): The y-coordinate of the detected object.
        """
        logger.debug(f"Checking line crossing for object: label={detected_object_label}, id={object_id}, x={x}, y={y}")

        if not self._is_label_enabled(detected_object_label):
            # Discard if the label is not enabled for tracking
            logger.debug(f"Label {detected_object_label} is not enabled for tracking. Discarding line crossing check.")
            return

        with self._counter_lock:
            if object_id in self._recent_objects:
                logger.debug(f"Object ID {object_id} has been seen before. Checking for line crossing.")
                last_x, last_y = self._recent_objects[object_id]
                x1, y1, x2, y2 = self._line_coordinates

                # Update the last seen position
                self._recent_objects[object_id] = (x, y)

                # Simple line crossing detection (horizontal line)
                if (last_y < y1 <= y) or (last_y > y1 >= y):
                    logger.debug(
                        f"Object ID {object_id} crossed the horizontal line. Incrementing crossing counter for label {detected_object_label}."
                    )
                    self._crossing_line_object_counters[detected_object_label] += 1
                # Simple line crossing detection (vertical line)
                elif (last_x < x1 <= x) or (last_x > x1 >= x):
                    logger.debug(f"Object ID {object_id} crossed the vertical line. Incrementing crossing counter for label {detected_object_label}.")
                    self._crossing_line_object_counters[detected_object_label] += 1
                else:
                    if (x2 - x1) == 0:
                        return
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    line_y_at_last_x = slope * last_x + intercept
                    line_y_at_current_x = slope * x + intercept
                    if (last_y < line_y_at_last_x and y >= line_y_at_current_x) or (last_y > line_y_at_last_x and y <= line_y_at_current_x):
                        logger.debug(
                            f"Object ID {object_id} crossed the diagonal line. Incrementing crossing counter for label {detected_object_label}."
                        )
                        self._crossing_line_object_counters[detected_object_label] += 1
            else:
                # First time seeing this object ID, just record its position
                logger.debug(f"First time seeing object ID {object_id}. Recording position without line crossing check.")
                self._recent_objects[object_id] = (x, y)

    def get_unique_objects_count(self) -> dict:
        """Get all identified object types and their counts since the last reset.
            This includes all distinguished objects sees, based on their unique IDs.

        Returns:
            dict: A dictionary with labels as keys and their respective counts as values.
        """
        with self._counter_lock:
            return dict(self._object_counters)

    def get_line_crossing_counts(self) -> dict:
        """Get the count of a specific identified object type since the last reset.
            This includes all distinguished objects sees, based on their unique IDs.

        Returns:
            dict: A dictionary with labels as keys and their respective counts as values.
        """
        with self._counter_lock:
            return dict(self._crossing_line_object_counters)

    def set_crossing_line_coordinates(self, x1: int, y1: int, x2: int, y2: int):
        """Set the coordinates of the line for counting objects crossing it."""
        with self._counter_lock:
            self._line_coordinates = (x1, y1, x2, y2)

        self.reset_counters()

    def set_horizontal_crossing_line(self, y: int):
        """Set a horizontal line for counting objects crossing it.

        Args:
            y (int): The y-coordinate of the horizontal line.
        """
        self.set_crossing_line_coordinates(0, y, 480, y)

    def set_vertical_crossing_line(self, x: int):
        """Set a vertical line for counting objects crossing it.

        Args:
            x (int): The x-coordinate of the vertical line.
        """
        self.set_crossing_line_coordinates(x, 0, x, 480)

    def reset_counters(self):
        """Reset the counts of tracked objects."""
        with self._counter_lock:
            self._object_counters.clear()
            self._recent_objects.clear()
            self._crossing_line_object_counters.clear()

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
                    self._override_thresholds(ws, self._confidence, self._max_observations, self._keep_grace, self._iou_threshold)

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

                    self._record_object(detected_object_label=detected_object, object_id=object_id, x=box.get("x", 0), y=box.get("y", 0))

                    # Check if the class_id matches any registered handlers
                    super()._execute_handler(detection=detected_object, detection_details=detection_details)

                if len(detections) > 0:
                    # If there are detections, invoke the all-detection handler
                    super()._execute_global_handler(detections=detections)

        else:
            # Leave logging for unknown message types for debugging purposes
            logger.warning(f"Unknown message type: {jmsg.get('type')}")

    def override_confidence(self, confidence: float):
        """Override the threshold for object detection model.

        Args:
            confidence (float): The new value for the confidence threshold in the range [0.0, 1.0].

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, confidence, self._max_observations, self._keep_grace, self._iou_threshold)

    def override_keep_grace(self, keep_grace: int):
        """Override keep grace for object detection model.
            Keep Grace: how many frames an object is kept if it disappears.

        Args:
            keep_grace (int): The new value for the keep grace.

        Raises:
            TypeError: If the value is not a number.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, self._confidence, self._max_observations, keep_grace, self._iou_threshold)

    def override_max_observations(self, max_observations: int):
        """Override max observations for object detection model.
            Max Observations: how many frames an object is kept if it disappears.

        Args:
            max_observations (int): The new value for the max observations.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, self._confidence, max_observations, self._keep_grace, self._iou_threshold)

    def override_iou_threshold(self, iou_threshold: int):
        """Override IoU threshold for object detection model.
            IOU Threshold: Intersection over Union threshold for tracking.

        Args:
            iou_threshold (float): The new value for the IoU threshold.

        Raises:
            TypeError: If the value is not a number.
            RuntimeError: If the model information is not available or does not support threshold override.
        """
        with connect(self._uri) as ws:
            self._override_threshold(ws, self._confidence, self._max_observations, self._keep_grace, iou_threshold)

    def _override_thresholds(self, ws: ClientConnection, confidence: float, max_observations: int, keep_grace: int, iou_threshold: float):
        """Override the threshold for object detection model.

        Args:
            ws (ClientConnection): The WebSocket connection to send the message through.
            confidence (float): The new value for the threshold.
            max_observations (int): Maximum number of observations to consider.
            keep_grace (int): Grace period to keep observations.
            iou_threshold (float): Intersection over Union threshold for tracking.

        Raises:
            TypeError: If the value is not a number.
        """
        if self._model_info is None or self._model_info.thresholds is None or len(self._model_info.thresholds) == 0:
            raise RuntimeError("Model information is not available or does not support threshold override.")

        for th in self._model_info.thresholds:
            if th.get("type") == "object_detection":
                id = th["id"]
                message = {"type": "threshold-override", "id": id, "key": "min_score", "value": confidence}

                logger.info(f"Overriding detection threshold. New confidence: {confidence}")
                ws.send(json.dumps(message))
                # Update local confidence value
                self._confidence = confidence

            if th.get("type") == "object_tracking":
                id = th["id"]
                message = {"type": "threshold-override", "id": id, "key": "max_observations", "value": max_observations}

                ws.send(json.dumps(message))
                # Update local max observations value
                self._max_observations = max_observations

                message = {"type": "threshold-override", "id": id, "key": "keep_grace", "value": keep_grace}

                ws.send(json.dumps(message))
                # Update local keep grace value
                self._keep_grace = keep_grace

                # Update Intersection over Union threshold
                message = {"type": "threshold-override", "id": id, "key": "threshold", "value": iou_threshold}

                ws.send(json.dumps(message))
                # Update local keep grace value
                self._iou_threshold = iou_threshold

                logger.info(f"Overriding thresholds - max_observations: {max_observations}, keep_grace: {keep_grace}, iou_threshold: {iou_threshold}")
