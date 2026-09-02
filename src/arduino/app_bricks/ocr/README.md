# OCR Brick

This OCR brick extracts the text visible in an image using the EasyOCR model
(CRAFT text detector plus CRNN recognizer) accelerated on the board's NPU. Given
an image it returns the recognized text in reading order, along with the position
and confidence of every detected text region.

The API is a single blocking call:

```python
from arduino.app_bricks.ocr import OCR

ocr = OCR()
result = ocr.extract_text("/path/to/photo.jpg")
print(result.text)
```

`extract_text` accepts a numpy array in BGR channel order (as returned by
`Camera.capture()`), the raw bytes of an encoded image file (e.g. JPEG or PNG),
or a path to an image file. It returns an `OcrResult`:

- `result.text` holds every recognized string joined by newlines, in reading
  order (top to bottom, left to right); it is empty when no text was found.
  `str(result)` yields the same text.
- `result.detections` lists one `TextDetection` per piece of text found, in
  reading order, each carrying the recognized `text`, the recognition
  `confidence`, the axis-aligned `bounding_box_xyxy` box and the `polygon` of
  the detected region — its 4 (x, y) vertices ordered top-left, top-right,
  bottom-right, bottom-left. Polygon and bounding box coincide for horizontal
  text; when the text is slanted, the polygon is the exact (rotated) region
  while the bounding box is the straight rectangle enclosing it.

Reading the text seen by a camera:

```python
from arduino.app_bricks.ocr import OCR
from arduino.app_peripherals.camera import Camera

ocr = OCR()
camera = Camera()
camera.start()

frame = camera.capture()
if frame is not None:
    result = ocr.extract_text(frame)
    for detection in result.detections:
        print(f"{detection.text} ({detection.confidence:.2f}) at {detection.bounding_box_xyxy}")
```

Tuning:

- `confidence` (constructor, overridable per call) drops detections whose
  recognition confidence is below the threshold and rebuilds `result.text` from
  the kept ones. Default is 0.3; pass 0.0 to report everything the model finds.
- `allowlist` (constructor, overridable per call) restricts recognition to the
  given characters, e.g. `"0123456789"` to read only digits from a meter or a
  serial number. It is applied by the model runner while decoding — the excluded
  characters cannot be emitted at all — so it improves accuracy on constrained
  text rather than just filtering the output. Pass `""` in a call to lift the
  restriction for that image only.

```python
from arduino.app_bricks.ocr import OCR

ocr = OCR(confidence=0.5)
reading = ocr.extract_text("/path/to/meter.jpg", allowlist="0123456789.")
print(reading.text)
```

Runner note: the model runner produces text metadata only — there is no annotated
video feed and no MJPEG stream. Calls are serialized and block until the runner
answers; `OCR(timeout=...)` bounds how long a call may wait (connection retries
while the container starts up included, 30 seconds by default). If the runner
cannot be reached in time, `extract_text` raises `OcrError`.
