from piper import PiperVoice
from piper.config import SynthesisConfig
from arduino.app_peripherals.speaker import Speaker

voice = PiperVoice.load("en_US-lessac-low.onnx")

long_story = """
Once upon a time, in a quiet village surrounded by green hills, there lived a boy named Alberto and his best friend, a golden dog named Milo.
Alberto and Milo were inseparable. Every morning, before school, Alberto would toss a stick across the meadow, and Milo would dash after it, tail wagging like a flag in the wind. When Alberto came home in the afternoon, Milo was always waiting by the gate, ears perked, eyes bright with joy.
One summer day, a storm rolled in unexpectedly. The wind howled, and the rain fell in sheets. Alberto realized that his little kite, the one he had built with his father, was still out by the riverbank. Without thinking, he ran outside to fetch it. Milo barked and followed.
When they reached the river, the kite was tangled in a bush dangerously close to the water. As Alberto tried to pull it free, his foot slipped on the muddy ground—and he tumbled in. The current was strong and cold. He cried out, but the wind swallowed his voice.
"""

out = Speaker(sample_rate=16000, format="FLOAT_LE")
out.start()

cfg = SynthesisConfig(
    length_scale=1.4,
)

print("---> Start streaming")
for chunk in voice.synthesize(text=long_story, syn_config=cfg):
    print(f"[C]--->{chunk}")
    out.play(chunk.audio_float_array, block_on_queue=True)

import time

time.sleep(120)
