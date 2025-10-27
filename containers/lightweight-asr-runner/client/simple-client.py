import websocket
import json
from arduino.app_peripherals.microphone import Microphone

ws = websocket.WebSocket()
ws.connect("ws://localhost:7777/ws", ping_interval=10, ping_timeout=10)

#websocket.enableTrace(True)

# Send config first
ws.send(json.dumps({"config": {"stream_partial": True}}))

mic = Microphone(periodsize=512)
mic.start()

# Process chunks
try:
    for chunk in mic.stream():
        if chunk is None:
            continue

        ws.send_binary(chunk.tobytes())
        rec=ws.recv()
        if "noop" in rec:
            continue
        print(rec)        

except KeyboardInterrupt:
    print("\nStopping loop...")
    mic.stop()

mic.stop()
ws.close()

