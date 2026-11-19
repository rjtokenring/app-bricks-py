# AprilTag Detection Brick

This Brick finds AprilTag fiducial markers in a camera video stream and reports, for every frame, the id and the position of each tag in view.

## Overview

AprilTags are square black and white markers that encode a number. They are printed and attached to objects, walls or floors so that a camera can recognize them and tell where they are and how they are rotated. Typical uses are robot localization, object tracking and camera-guided positioning.

The AprilTag Detection Brick allows you to:

- Capture frames from a Camera (see Camera peripheral for supported cameras).
- Configure Camera settings (resolution and frame rate).
- Choose the tag family to look for, or several of them for scenes that mix tags of different families.
- Get the id, the four corners and the center of every tag found in each frame.
- Process detections with customizable callbacks.

## Features

- Supported tag families: `tag36h11` (default and recommended), `tag25h9` and `tag16h5`
- Detects several tags in the same frame, also of different families
- Reports the tags of every frame, so a tag can be followed while it moves
- Corners are ordered after the tag, not the image, so they also tell how the tag is rotated
- Runs on the board, no AI model and no additional container required

## Prerequisites

To use this Brick you can choose to plug a camera to your board or use a network-connected camera.

**Tip**: Use a USB-C® Hub with USB-A connectors to support commercial web cameras.

The tags must belong to one of the families the Brick is configured for. Ready-to-print images of every family are published by the AprilTag authors at https://github.com/AprilRobotics/apriltag-imgs: scale them up without smoothing and keep the white margin around the black border, a tag without it is not detected.

## Code example and usage

```python
from PIL.Image import Image
from arduino.app_utils import App
from arduino.app_bricks.apriltag_detection import AprilTagDetection, AprilTag

def handle_tag(frame: Image, tag: AprilTag):
    print(f"Tag {tag.id} at {tag.center}")

detector = AprilTagDetection()
detector.on_detect(handle_tag)

App.run()
```

You can also select a specific camera and the tag families, receive all the tags of a frame at once and draw them on the frame:

```python
from PIL.Image import Image
from arduino.app_utils import App
from arduino.app_peripherals.camera import Camera
from arduino.app_bricks.apriltag_detection import AprilTagDetection, AprilTag, draw_tag

def handle_tags(frame: Image, tags: list[AprilTag]):
    for tag in tags:
        draw_tag(frame, tag)
    print(f"Tags in view: {[(tag.family, tag.id) for tag in tags]}")

# Select the camera you want to use, its resolution and the max fps
camera = Camera(resolution=(1280, 720), fps=15)
detector = AprilTagDetection(camera, families=["tag36h11", "tag25h9"])
detector.on_detect(handle_tags)

App.run()
```

Notes:

- The constructor is `AprilTagDetection(camera=None, families="tag36h11")`. `families` is optional and takes one family name or a list of them, e.g. `families=["tag36h11", "tag25h9"]`: every tag reports the family it belongs to and ids are unique only within a family.
- Prefer `tag36h11`: the more bits a family encodes, the less likely a pattern in the background is taken for a tag. `tag16h5` encodes the fewest and can occasionally report a tag that is not there, so list it only when tags of that family are in use.
- Each `AprilTag` carries the `id` (int), the `family` (str), the `corners` and the `center` of the tag. `corners` is a NumPy array of shape (4, 2) with the pixel coordinates of the top-left, top-right, bottom-right and bottom-left corner of the black border of the tag as it is printed upright: when the tag is rotated, the first corner moves with it. `center` is an `(x, y)` tuple.
- The `on_detect` callback is invoked for every frame that contains tags, so a tag that stays in view is reported again at each frame. By default it is invoked once per tag. To receive all the tags of a frame in a single call, annotate the second parameter as `list[AprilTag]`.
- Use `on_frame(callback)` to receive every captured frame, with or without tags, and `on_error(callback)` to be notified of errors raised while capturing or scanning.
- `detect(frame)` scans a single image, a NumPy array in greyscale or BGR format, and returns the list of the tags found.
- `draw_tag(frame, tag)` draws the outline of a tag, a dot on its top-left corner and its id on the frame.
- A tag is found more reliably the larger it appears in the frame: move the camera closer, print a larger tag or raise the camera resolution when small or distant tags are missed.
