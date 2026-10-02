# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from collections.abc import Callable

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

from .detections import Segmentation, parse_segmentation

logger = Logger("ImageSegmentation")


@brick
class ImageSegmentation:
    def __init__(
        self,
        camera: BaseCamera | None = None,
        confidence: float = 0.5,
        min_person_ratio: float = 0.0,
        exit_debounce_sec: float = 0.0,
        background_color: tuple[int, int, int] = (68, 132, 255),
        background_opacity: float = 1.0,
    ) -> None:
        """Initialize the ImageSegmentation brick.

        Args:
            camera (BaseCamera): The camera instance to use for capturing video. If None, a default
                camera will be initialized. Pass the same instance shared with other bricks to reuse
                a single camera.
            confidence (float): Per-pixel confidence, in [0.0, 1.0], above which a pixel belongs to
                a person. Lower values grow the person outline, higher values shrink it. Applied by
                the model runner, so it shapes both the reported data and the overlay. Default is
                0.5. Changeable at runtime with `set_confidence()`.
            min_person_ratio (float): Minimum fraction of the frame, in [0.0, 1.0], people must
                cover to count as present, e.g. 0.05 to ignore people far from the camera.
                Default is 0 (any person).
            exit_debounce_sec (float): Minimum seconds the scene must stay empty before `on_exit`
                reports it, so that a dropped detection frame cannot fake a person leaving. People
                appearing are always reported at once. Default is 0 (no debounce).
            background_color (tuple[int, int, int]): (R, G, B) color, each channel in [0, 255],
                painted over the background on the video overlay served by the model runner.
                Changeable at runtime with `set_background_color()`.
            background_opacity (float): Opacity of the background color in [0.0, 1.0]: 1 replaces
                the background, 0 leaves the video untouched. Default is 1. Changeable at runtime
                with `set_background_opacity()`.

        Raises:
            ValueError: If a numeric argument is out of its range or background_color is not an
                (R, G, B) tuple.
            RuntimeError: If the model runner host address could not be resolved.
        """
        self._camera = camera if camera else Camera(fps=30)
        self._confidence = self._validate_unit("confidence", confidence)
        self._min_person_ratio = self._validate_unit("min_person_ratio", min_person_ratio)
        self._exit_debounce_sec = self._validate_non_negative("exit_debounce_sec", exit_debounce_sec)
        self._background_color = self._validate_color(background_color)
        self._background_opacity = self._validate_unit("background_opacity", background_opacity)

        # Callbacks
        self._callbacks: dict[str, Callable[..., Any]] = {}
        self._callbacks_lock = threading.Lock()

        # State tracking
        self._person_present = False
        self._absent_since: float | None = None
        self._is_running = False

        self._camera_frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)

        # Callback executor and per-callback in-progress locks
        self._executor: ThreadPoolExecutor | None = None
        self._callback_locks: dict[str, threading.Lock] = {}

        # WebSocket endpoints
        infra: dict[str, Any] | None = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        services: dict[str, Any] = infra["services"]
        service = next(iter(services))  # Only one service is expected

        self._host = resolve_address(service)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._ws_send_url = f"ws://{self._host}:5000"
        self._ws_recv_url = f"ws://{self._host}:5001"

    def start(self) -> None:
        """Start the capture thread and asyncio event loop."""
        self._executor = ThreadPoolExecutor()
        self._camera.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop all tracking and close connections."""
        self._is_running = False
        self._camera.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        # Reset the presence state so a restart begins from a clean slate
        self._person_present = False
        self._absent_since = None

    def on_segmentation(self, callback: Callable[[Segmentation], None] | None) -> None:
        """Register a callback invoked for every processed frame in which people are present.

        Args:
            callback (Callable[[Segmentation], None]): Function to call with the frame's
                `Segmentation` (area covered by people, confidence and bounding box).
                None to unregister.
        """
        self._register_callback("segmentation", callback)

    def on_enter(self, callback: Callable[[], None] | None) -> None:
        """Register a callback for when people enter the scene.

        Args:
            callback (Callable[[], None]): Function to call when people are detected after
                nobody was in view. None to unregister.
        """
        self._register_callback("enter", callback)

    def on_exit(self, callback: Callable[[], None] | None) -> None:
        """Register a callback for when the last person leaves the scene.

        Args:
            callback (Callable[[], None]): Function to call when no people are detected
                anymore. None to unregister.
        """
        self._register_callback("exit", callback)

    def on_frame(self, callback: Callable[[np.ndarray], None] | None) -> None:
        """Register a callback that receives each raw camera frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with camera frame data.
                None to unregister.
        """
        self._register_callback("frame", callback)

    def on_error(self, callback: Callable[[Exception], None] | None) -> None:
        """Register a callback invoked when an error occurs while processing detections.

        Args:
            callback (Callable[[Exception], None]): Function to call with the raised exception.
                None to unregister.
        """
        self._register_callback("error", callback)

    @property
    def person_present(self) -> bool:
        """Whether people are in view right now, the state `on_enter`/`on_exit` last reported."""
        return self._person_present

    def set_confidence(self, confidence: float) -> None:
        """Change the per-pixel confidence above which a pixel belongs to a person, effective immediately.

        Args:
            confidence (float): New threshold in [0.0, 1.0], forwarded to the model runner.

        Raises:
            ValueError: If confidence is not a number in [0.0, 1.0].
        """
        self._confidence = self._validate_unit("confidence", confidence)
        logger.debug(f"segmentation confidence set to {self._confidence}")

    def set_background_color(self, color: tuple[int, int, int]) -> None:
        """Change the color painted over the background on the overlay, effective immediately.

        Args:
            color (tuple[int, int, int]): (R, G, B) color, each channel in [0, 255].

        Raises:
            ValueError: If color is not an (R, G, B) tuple of integers in [0, 255].
        """
        self._background_color = self._validate_color(color)
        logger.debug(f"background color set to {self._background_color}")

    def set_background_opacity(self, opacity: float) -> None:
        """Change the opacity of the background color on the overlay, effective immediately.

        Args:
            opacity (float): Opacity in [0.0, 1.0]: 1 replaces the background, 0 leaves the
                video untouched.

        Raises:
            ValueError: If opacity is not a number in [0.0, 1.0].
        """
        self._background_opacity = self._validate_unit("background_opacity", opacity)
        logger.debug(f"background opacity set to {self._background_opacity}")

    @staticmethod
    def _validate_non_negative(name: str, value: object) -> float:
        """Check that value is a non-negative number and return it as a float."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number, got {value!r}")
        return float(value)

    @staticmethod
    def _validate_unit(name: str, value: object) -> float:
        """Check that value is a number in [0.0, 1.0] and return it as a float."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a number in [0.0, 1.0], got {value!r}")
        return float(value)

    @staticmethod
    def _validate_color(color: object) -> tuple[int, int, int]:
        """Check that color is an (R, G, B) tuple of integers in [0, 255]."""
        if not isinstance(color, (tuple, list)):
            raise ValueError(f"color must be an (R, G, B) tuple, got {color!r}")
        channels: list[int] = []
        for channel in cast("tuple[object, ...] | list[object]", color):
            if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255:
                raise ValueError(f"color channels must be integers in [0, 255], got {channel!r}")
            channels.append(channel)
        if len(channels) != 3:
            raise ValueError(f"color must be an (R, G, B) tuple, got {color!r}")
        return (channels[0], channels[1], channels[2])

    def _register_callback(self, key: str, callback: Callable[..., Any] | None) -> None:
        with self._callbacks_lock:
            if callback is None:
                self._callbacks.pop(key, None)
                self._callback_locks.pop(key, None)
            else:
                self._callbacks[key] = callback
                if key not in self._callback_locks:
                    self._callback_locks[key] = threading.Lock()

    def _get_callback(self, key: str) -> Callable[..., Any] | None:
        with self._callbacks_lock:
            return self._callbacks.get(key)

    @brick.loop
    def _capture_loop(self) -> None:
        """Continuously capture frames from camera (runs in dedicated thread)."""
        try:
            frame = self._camera.capture()
            if frame is None:
                time.sleep(0.01)
                return

            frame_cb = self._get_callback("frame")
            if frame_cb:
                try:
                    frame_cb(frame)
                except Exception as e:
                    logger.error(f"Error in frame callback: {e}")

            jpeg_frame = compress_to_jpeg(frame)
            if jpeg_frame is None:
                time.sleep(0.01)
                return

            try:
                self._camera_frame_queue.put(jpeg_frame, block=False)
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self._camera_frame_queue.get_nowait()
                    self._camera_frame_queue.put(jpeg_frame, block=False)
                except (queue.Empty, queue.Full):
                    pass

        except Exception as e:
            if self._is_running:
                logger.error(f"Error capturing frame: {e}")

    @brick.execute
    def _send_receive_loop(self) -> None:
        """Run the asyncio event loop in a dedicated thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            tasks = asyncio.gather(self._send_frames_task(), self._receive_detections_task(), return_exceptions=True)
            loop.run_until_complete(tasks)

        except Exception as e:
            logger.error(f"Error in asyncio loop: {e}")
        finally:
            loop.close()

    def _runner_config(self) -> dict[str, Any]:
        """The settings the model runner applies, as its config message carries them."""
        return {
            "mask_threshold": self._confidence,
            "background_color": list(self._background_color),
            "background_opacity": self._background_opacity,
        }

    async def _send_frames_task(self) -> None:
        """Send frames to the processing container via WebSocket."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_send_url) as ws:
                    sent_config: dict[str, Any] | None = None
                    while self._is_running:
                        config = self._runner_config()
                        if config != sent_config:
                            await ws.send(json.dumps({"config": config}))
                            sent_config = config
                        try:
                            frame = await asyncio.get_event_loop().run_in_executor(None, self._camera_frame_queue.get, True, 0.1)
                        except queue.Empty:
                            continue

                        b64_frame = base64.b64encode(frame.tobytes()).decode("utf-8")
                        payload = {"frame": b64_frame}

                        await ws.send(json.dumps(payload))

            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in send frames task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    async def _receive_detections_task(self) -> None:
        """Receive detection results and dispatch events."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_recv_url) as ws:
                    while self._is_running:
                        data = await ws.recv()
                        detection = json.loads(data)

                        self._process_detection(detection.get("metadata", {}))

            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON data: {e}")
            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in receive detections task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    def _process_detection(self, metadata: dict[str, Any]) -> None:
        """Process detection data and dispatch appropriate events."""
        try:
            segmentation = parse_segmentation(metadata)
        except Exception as e:
            logger.error(f"Error parsing detection metadata: {e}")
            self._submit_callback("error", e)
            return

        if segmentation is not None and segmentation.person_ratio < self._min_person_ratio:
            segmentation = None

        # Dispatch enter/exit events, the exit debounced to filter out detection flicker
        present = segmentation is not None
        if present == self._person_present:
            self._absent_since = None
        elif present:
            self._person_present = True
            self._submit_callback("enter")
        else:
            now = time.monotonic()
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self._exit_debounce_sec:
                self._person_present = False
                self._absent_since = None
                self._submit_callback("exit")

        if segmentation is not None:
            self._submit_callback("segmentation", segmentation)

    def _submit_callback(self, key: str, *args: Any) -> None:
        """Acquire the per-callback lock and submit the callback to the executor.

        If the lock is already held (callback still running), the event is discarded.
        """
        callback = self._get_callback(key)
        if callback is None or self._executor is None:
            return
        with self._callbacks_lock:
            lock = self._callback_locks.get(key)
        if lock is None or not lock.acquire(blocking=False):
            return
        try:
            self._executor.submit(self._run_callback, lock, callback, *args)
        except RuntimeError:
            # Executor was shut down before the task could be submitted
            lock.release()

    def _run_callback(self, lock: threading.Lock, callback: Callable[..., Any], *args: Any) -> None:
        """Run a callback and release its lock when done."""
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            error_cb = self._get_callback("error")
            if error_cb and callback is not error_cb:
                try:
                    error_cb(e)
                except Exception as nested:
                    logger.error(f"Error in error callback: {nested}")
        finally:
            lock.release()
