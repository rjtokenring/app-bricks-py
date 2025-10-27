# Install: pip install fastapi uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from vosk import Model, KaldiRecognizer
import sys
import logging
import json

# ---------------- Logging ----------------
logger = logging.getLogger("vosk-runner")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------- Configuration ----------------
SERVER_BIND_ADDRESS = "0.0.0.0"
SERVER_BIND_PORT = 7777
MODEL_PATH = "models/vosk-model-small-en-us-0.15"

# ---------------- Load Model ----------------
logger.info("Loading Vosk model...")
model = Model(MODEL_PATH)
logger.info("Model loaded.")

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    rec = KaldiRecognizer(model, 16000)
    try:
        last_partial = ""
        while True:
            # Wait for a message from the client
            data = await websocket.receive()

            # Handle text messages (e.g., config)
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if "config" in msg:
                        stream_partial = bool(msg["config"].get("stream_partial", True))
                        await websocket.send_text(json.dumps({"event":  "info", "message": "config_updated"}))
                    else:
                        await websocket.send_text(json.dumps({"event": "echo", "message": f"{data}"}))
                except Exception as e:
                    logger.warning(f"Invalid text frame: {e}")
                continue

            # Handle binary audio frames
            if "bytes" in data:
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
        print("Client disconnected")
        await websocket.close()
        logger.info(f"Client cleanup complete.")
    except Exception as e:
        logger.exception(f"Error in processing loop: {e}")
        
if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_BIND_ADDRESS, port=SERVER_BIND_PORT)
