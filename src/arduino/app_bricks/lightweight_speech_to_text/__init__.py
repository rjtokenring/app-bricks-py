from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils import Logger, brick
from arduino.app_peripherals.microphone import Microphone
from typing import Iterator
import threading
import websocket
import json

logger = Logger(__name__)


@brick
class LightweightSpeechToText:
    """
    LightweightSpeechToText module for speech-to-text conversion using VOSK model.
    """

    def __init__(self, mic: Microphone = None, stream_partial: bool = True, custom_model: str = None):
        if mic:
            logger.info(f"[{self.__class__.__name__}] Using provided microphone: {mic}")
            self._mic = mic
            self._mic_provided = True
        else:
            self._mic = Microphone()  # Default microphone if none provided
            self._mic_provided = False

        self.custom_model = custom_model
        self._stream_partial = stream_partial
        self._mic_lock = threading.Lock()

        infra = load_brick_compose_file(self.__class__)
        for k, v in infra["services"].items():
            self._host = k
            self.infra = v
            break  # Only one service is expected

        self._host = resolve_address(self._host)

        self._port = 7777  # Default VOSK CONTAINER HTTP port
        self._url = f"ws://{self._host}:{self._port}/ws"
        logger.info(f"[{self.__class__.__name__}] Host: {self._host} - Ports: {self._port} - URL: {self._url}")

    def start(self):
        """Start the LightweightSpeechToText module."""
        if self._mic_provided:
            with self._mic_lock:
                self._mic.start()
                logger.info(f"[{self.__class__.__name__}] Microphone started.")

    def stop(self):
        """Stop the LightweightSpeechToText module."""
        if self._mic_provided:
            with self._mic_lock:
                self._mic.stop()
                logger.info(f"[{self.__class__.__name__}] Microphone stopped.")

    def _init_ws_connection(self):
        """Initialize WebSocket connection to the ASR server."""
        ws = websocket.WebSocket()
        ws.connect(self._url, ping_interval=10, ping_timeout=10)
        # Send configuration message
        config_msg = {
            "config": {
                "stream_partial": self._stream_partial,
            }
        }
        if self.custom_model is not None and self.custom_model != "":
            config_msg["config"]["custom_model"] = self.custom_model
        ws.send(json.dumps(config_msg))
        return ws

    def transcribe(self) -> Iterator[dict]:
        """Perform speech-to-text recognition.

        Returns:
            Iterator[dict]: An iterator yielding recognition results as dictionaries. "event" key indicates the type of result ("partial_text", "text", "error").
                            "data" key contains the recognized text.

        """

        ws = self._init_ws_connection()
        try:
            for chunk in self._mic.stream():
                if chunk is None:
                    continue

                ws.send_binary(chunk.tobytes())
                rec = ws.recv()
                if "noop" in rec:
                    # No operation, continue to next chunk
                    continue

                msg = json.loads(rec)
                yield msg

        except KeyboardInterrupt:
            logger.info("Recognition interrupted by user. Exiting...")
        except Exception as e:
            logger.error(f"Error during recognition: {e}")
        finally:
            # Close WebSocket connection
            ws.close()
