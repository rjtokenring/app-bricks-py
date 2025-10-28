# Install: pip install fastapi uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from vosk import Model, KaldiRecognizer
import threading
from log import load_logger
import json

# ---------------- Logging ----------------
logger = load_logger()

# ---------------- Configuration ----------------
DEFAULT_BITRATE = 16000
SERVER_BIND_ADDRESS = "0.0.0.0"
SERVER_BIND_PORT = 7777
MODEL_PATH = "/home/arduino/models/vosk-model-small-en-us-0.15"
CUSTOM_MODEL_PATH = "/home/arduino/custom-models"

# ---------------- Load Model ----------------
logger.info("Loading default Vosk model...")
model = Model(MODEL_PATH)
logger.info("Model loaded.")

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logger.info(f"Client connected: {websocket.client}")
    rec_lock = threading.Lock()
    with rec_lock:
        # Load default model
        try:
            rec = KaldiRecognizer(model, DEFAULT_BITRATE)
        except Exception as e:
            logger.error(f"Failed to load custom model from {CUSTOM_MODEL_PATH}: {e}")
            return

    try:
        last_partial = ""
        while True:
            # Wait for a message from the client
            data = await websocket.receive()

            if "type" in data and data["type"] == "websocket.disconnect":
                logger.info("Client disconnected")
                return

            # Handle text messages (e.g., config)
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if "config" in msg:
                        custom_model = msg["config"].get("custom_model", None)
                        if custom_model is not None and custom_model != "":
                            try:
                                logger.info(f"Loading custom model from: {CUSTOM_MODEL_PATH}/{custom_model}")
                                custom_model_path = f"{CUSTOM_MODEL_PATH}/{custom_model}"
                                custom_vosk_model = Model(custom_model_path)
                                with rec_lock:
                                    rec = KaldiRecognizer(custom_vosk_model, DEFAULT_BITRATE)
                                    logger.info(f"Custom model {custom_model} loaded successfully.")
                            except Exception as e:
                                logger.error(f"Failed to load custom model: {e}")
                                await websocket.send_text(json.dumps({"event": "error", "message": f"Failed to load custom model: {e}"}))
                                continue
                        else:
                            # Revert to default model
                            logger.info("Reverting to default model.")
                            rec = KaldiRecognizer(model, DEFAULT_BITRATE)

                        stream_partial = bool(msg["config"].get("stream_partial", True))
                        await websocket.send_text(json.dumps({"event": "info", "message": "configuration updated successfully"}))
                except Exception as e:
                    logger.warning(f"Invalid text frame: {e}")
                continue

            # Handle binary audio frames
            elif "bytes" in data:
                audio_bytes = data["bytes"]
                if not audio_bytes:
                    continue

                if rec.AcceptWaveform(audio_bytes):
                    res = rec.Result()
                    await websocket.send_text(json.dumps({"event": "text", "data": json.loads(res)}))
                    last_partial = ""
                else:
                    if stream_partial:
                        partial_res = rec.PartialResult()
                        if partial_res:
                            json_data = json.loads(partial_res)
                            partial_text = json_data.get("partial", "").strip()
                            if partial_text and partial_text != last_partial:
                                last_partial = partial_text
                                await websocket.send_text(json.dumps({"event": "partial_text", "data": json_data}))
                            else:
                                await websocket.send_text(json.dumps({"event": "noop"}))
            else:
                logger.warning("Unknown message type from client.")

    except (WebSocketDisconnect, KeyboardInterrupt):
        await websocket.close()
        logger.info(f"Client cleanup complete.")
    except Exception as e:
        logger.exception(f"Error in processing loop: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_BIND_ADDRESS, port=SERVER_BIND_PORT)
