from arduino.app_internal.core import load_brick_compose_file, resolve_address
from arduino.app_utils import Logger, brick
from arduino.app_peripherals.microphone import Microphone
import threading
import websocket

logger = Logger(__name__)

@brick
class LightweightSpeechToText:
    """
    LightweightSpeechToText module for speech-to-text conversion using VOSK model.
    """
    
    def __init__(self, mic: Microphone = None):
        # TODO: evaluate if it is needed to reduce to: periodsize=512
        self._mic = mic if mic else Microphone() # Default microphone if none provided
        self._mic_lock = threading.Lock()

        infra = load_brick_compose_file(self.__class__)
        for k, v in infra["services"].items():
            self._host = k
            self.infra = v
            break  # Only one service is expected

        self._host = resolve_address(self._host)

        self._port = 7777  # Default VOSK HTTP port
        self._url = f"ws://{self._host}:{self._port}/ws"
        logger.warning(f"[{self.__class__.__name__}] Host: {self._host} - Ports: {self._port} - URL: {self._url}")

    def start(self):
        """Start the LightweightSpeechToText module."""
        with self._mic_lock:
            self._mic.start()
            logger.info(f"[{self.__class__.__name__}] Microphone started.")

    def stop(self):
        """Stop the LightweightSpeechToText module."""
        with self._mic_lock:
            self._mic.stop()
            logger.info(f"[{self.__class__.__name__}] Microphone stopped.")


    def recognize(self, stream_partial: bool = True):
        """Perform speech-to-text recognition.

        Args:
            stream_partial (bool, optional): Whether to stream partial results. Defaults to True.

        Returns:
            str: Recognized text from the audio input.
        """
        ws = websocket.WebSocket()
        ws.connect(self._url, ping_interval=10, ping_timeout=10)
        try:
            for chunk in self._mic.stream():
                if chunk is None:
                    continue

                ws.send_binary(chunk.tobytes())
                rec=ws.recv()
                if "noop" in rec:
                    continue
                print(rec)        

        # TODO: handle exceptions properly
        except KeyboardInterrupt:
            print("\nStopping loop...")
            ws.close()
        except Exception as e:
            print("\nStopping loop...")
            ws.close()