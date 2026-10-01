# Image Segmentation Brick

This image segmentation brick analyzes a camera video stream and separates the people in view from the background, pixel by pixel. The output is a video stream with the background covered by a color overlay, with the added capability to trigger actions based on people presence, the area they cover and their bounding box.

## Overview

The Image Segmentation Brick allows you to:

- Separate people from the background in real time from a camera stream.
- React to people entering or leaving the camera view.
- Read, for every frame, how much of the picture people cover and the box that encloses them.
- Change the overlay color and opacity, and the segmentation sensitivity, while the app runs.
- Access the raw camera frames via a frame callback.

## Prerequisites

To use this Brick you need to have a camera connected to your board.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.image_segmentation import ImageSegmentation

segmentation = ImageSegmentation()
segmentation.on_enter(lambda: print("Person detected!"))
segmentation.on_exit(lambda: print("No person detected"))
segmentation.on_segmentation(lambda s: print(f"People cover {s.person_ratio:.0%} of the frame, box {s.bounding_box_xyxy}"))

App.run()
```

You can change the overlay at runtime, e.g. a green screen that turns translucent when people come close:

```python
from arduino.app_utils import App
from arduino.app_bricks.image_segmentation import ImageSegmentation, Segmentation

segmentation = ImageSegmentation(background_color=(0, 255, 0))


def on_segmentation(s: Segmentation):
    segmentation.set_background_opacity(0.5 if s.person_ratio > 0.4 else 1.0)


segmentation.on_segmentation(on_segmentation)

App.run()
```

## Configuration

`ImageSegmentation(camera=None, confidence=0.5, min_person_ratio=0.0, exit_debounce_sec=0.0, background_color=(68, 132, 255), background_opacity=1.0)`:

- `camera` (`BaseCamera`, optional): the camera instance to use. If not provided, a default `Camera(fps=30)` is created.
- `confidence` (`float`): per-pixel confidence (0.0 to 1.0) above which a pixel belongs to a person. Lower values grow the person outline, higher values shrink it.
- `min_person_ratio` (`float`): minimum fraction of the frame (0.0 to 1.0) people must cover to count as present, e.g. 0.05 to ignore people far from the camera. Default is 0 (any person).
- `exit_debounce_sec` (`float`): minimum seconds the scene must stay empty before `on_exit` reports it, so that a dropped detection frame cannot fake a person leaving. Default is 0 (no debounce).
- `background_color` (`tuple[int, int, int]`): (R, G, B) color painted over the background on the video overlay, each channel in [0, 255].
- `background_opacity` (`float`): opacity of the background color (0.0 to 1.0): 1 replaces the background, 0 leaves the video untouched.

## Methods

- **`on_segmentation(callback)`**: registers a callback that receives a `Segmentation` for every processed frame in which people are present: `person_ratio` (fraction of the frame covered by people), `confidence` (mean model confidence over the person pixels) and `bounding_box_xyxy` (`(x1, y1, x2, y2)` in frame pixels).
- **`on_enter(callback)`**: registers a zero-argument callback invoked when people become visible after nobody was in view.
- **`on_exit(callback)`**: registers a zero-argument callback invoked when no people are visible anymore.
- **`on_frame(callback)`**: registers a callback that receives each raw camera frame as a NumPy array.
- **`on_error(callback)`**: registers a callback that receives the exceptions raised while processing detections or inside other callbacks.
- **`set_confidence(value)`**: changes the per-pixel confidence threshold at runtime.
- **`set_background_color(color)`**: changes the overlay color at runtime.
- **`set_background_opacity(value)`**: changes the overlay opacity at runtime.

Pass `callback=None` to any `on_*` method to unregister. While a callback is still running, further events of the same kind are discarded instead of queueing up.

## Properties

- **`person_present`**: whether people are in view right now (bool), the state `on_enter`/`on_exit` last reported.

## Technical Details

**Model**: MediaPipe Selfie Segmentation (Qualcomm AI Hub, quantized), running on the NPU with a 256x256 input. Each frame is stretched to the model input, so area ratios and boxes map back to the frame with a plain per-axis scale.

**One segmentation for everybody**: the model labels pixels as person or background and does not tell people apart. With several people in view, `person_ratio` and `bounding_box_xyxy` cover all of them. Person regions smaller than 0.2% of the frame are treated as noise and dropped.

**Runner**: the annotated video is served as an MJPEG stream on port 5002.
