# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Literal

import numpy as np
import websockets

from arduino.app_peripherals.camera import BaseCamera, Camera
from arduino.app_utils import brick, Logger
from arduino.app_utils.image.adjustments import compress_to_jpeg
from arduino.app_internal.core.module import load_brick_compose_file, resolve_address

logger = Logger("ImageSegmentation")


@brick
class ImageSegmentation:
    def __init__(self, camera: BaseCamera | None = None):
        if camera is None:
            camera = Camera(fps=30)
        self._camera = camera

        # Callbacks
        self._gesture_callbacks = {}  # {(gesture, hand): callback}
        self._enter_callback = None
        self._exit_callback = None
        self._frame_callback = None
        self._callbacks_lock = threading.Lock()

        # State tracking
        self._had_hands = False
        self._is_running = False

        self._camera_frame_queue = queue.Queue(maxsize=2)

        # Callback executor and per-callback in-progress locks
        self._executor: ThreadPoolExecutor | None = None
        self._callback_locks: dict[str | tuple, threading.Lock] = {}  # keyed by "enter", "exit", or (gesture, hand)

        # WebSocket endpoints
        infra = load_brick_compose_file(self.__class__)
        if infra is None or "services" not in infra:
            raise RuntimeError("Infrastructure configuration could not be loaded.")
        for k, _ in infra["services"].items():
            self._host = k
            break  # Only one service is expected

        self._host = resolve_address(self._host)
        if not self._host:
            raise RuntimeError("Host address could not be resolved. Please check your configuration.")

        self._ws_send_url = f"ws://{self._host}:5000"
        self._ws_recv_url = f"ws://{self._host}:5001"

    def start(self):
        """Start the capture thread and asyncio event loop."""
        self._executor = ThreadPoolExecutor()
        self._camera.start()
        self._is_running = True

    def stop(self):
        """Stop all tracking and close connections."""
        self._is_running = False
        self._camera.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def on_enter(self, callback: Callable[[], None]):
        """
        Register a callback for when hands become visible.

        Args:
            callback (Callable[[], None]): Function to call when at least one hand is detected
        """
        with self._callbacks_lock:
            self._enter_callback = callback
            if callback is not None:
                self._callback_locks["enter"] = threading.Lock()
            else:
                self._callback_locks.pop("enter", None)

    def on_exit(self, callback: Callable[[], None]):
        """
        Register a callback for when hands are no longer visible.

        Args:
            callback (Callable[[], None]): Function to call when no hands are detected anymore
        """
        with self._callbacks_lock:
            self._exit_callback = callback
            if callback is not None:
                self._callback_locks["exit"] = threading.Lock()
            else:
                self._callback_locks.pop("exit", None)

    def on_frame(self, callback: Callable[[np.ndarray], None]):
        """
        Register a callback that receives each camera frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with camera frame data. None to unregister.
        """
        with self._callbacks_lock:
            self._frame_callback = callback

    @brick.loop
    def _capture_loop(self):
        """Continuously capture frames from camera (runs in dedicated thread)."""
        try:
            frame = self._camera.capture()
            if frame is None:
                time.sleep(0.01)
                return

            with self._callbacks_lock:
                frame_cb = self._frame_callback
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
                except:
                    pass

        except Exception as e:
            if self._is_running:
                logger.error(f"Error capturing frame: {e}")

    @brick.execute
    def _send_receive_loop(self):
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

    async def _send_frames_task(self):
        """Send frames to the processing container via WebSocket."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_send_url) as ws:
                    while self._is_running:
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

    async def _receive_detections_task(self):
        """Receive detection results and dispatch events."""
        while self._is_running:
            try:
                async with websockets.connect(self._ws_recv_url) as ws:
                    while self._is_running:
                        data = await ws.recv()
                        detection = json.loads(data)

                        # TODO: implement metadata extraction and callbacks

            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON data: {e}")
                pass
            except Exception as e:
                if self._is_running:
                    logger.error(f"Error in receive detections task: {e}. Reconnecting...")
                    await asyncio.sleep(3)

    def _submit_callback(self, key: str | tuple, callback: Callable, *args):
        """Acquire the per-callback lock and submit callback to the executor.

        If the lock is already held (callback still running), the event is discarded.
        """
        if self._executor is None:
            return
        lock = self._callback_locks.get(key)
        if lock is None or not lock.acquire(blocking=False):
            return
        self._executor.submit(self._run_callback, lock, callback, *args)

    def _run_callback(self, lock: threading.Lock, callback: Callable, *args):
        """Run a callback and release its lock when done."""
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
        finally:
            lock.release()
