
# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
import sys

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import inference_callback
from utils.constants import MASK_THRESHOLD
from utils.draw import draw_segmentation

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
LIGHT_GREEN = (144, 238, 144)


def test_inference_shapes_and_metadata():
    """Test that inference_callback returns correct shapes and JSON-serializable metadata."""
    for fname in ("person1.jpg", "person2.jpg"):
        path = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(path)
        assert bgr is not None, f"Could not load {path}"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        annotated, metadata = inference_callback(rgb)

        # Annotated frame keeps original dimensions
        assert annotated.shape == (h, w, 3), f"annotated shape mismatch for {fname}"
        assert annotated.dtype == np.uint8

        # Metadata must be JSON-serializable
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)

        # Metadata fields
        assert isinstance(metadata["person_detected"], bool)
        assert isinstance(metadata["person_ratio"], float)
        assert 0.0 <= metadata["person_ratio"] <= 1.0
        assert isinstance(metadata["mean_confidence"], float)
        assert 0.0 <= metadata["mean_confidence"] <= 1.0

        if metadata["person_detected"]:
            bb = metadata["bounding_box_xyxy"]
            assert len(bb) == 4
            assert all(isinstance(v, int) for v in bb)
        else:
            assert metadata["bounding_box_xyxy"] is None

        print(f"[PASS] {fname}: {json.dumps(metadata, indent=2)}")


def test_segmentation_detects_person():
    """Test that the model detects a person in the selfie images."""
    for fname in ("person1.jpg", "person2.jpg"):
        path = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        _, metadata = inference_callback(rgb)

        assert metadata["person_detected"], f"{fname}: no person detected"
        assert metadata["person_ratio"] > 0.05, f"{fname}: person_ratio={metadata['person_ratio']:.4f}, expected > 0.05"
        print(f"[PASS] {fname}: person_ratio={metadata['person_ratio']:.2%}")


def test_produce_output_images():
    """Produce output images with light green background overlay."""
    from utils.image_processing import resize_to_input, resize_mask_to_original
    from utils.model_io_processing import dequantize
    from utils.constants import INPUT_HEIGHT, INPUT_WIDTH
    from inference import segmentation_model, model_input, model_output

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in ("person1.jpg", "person2.jpg"):
        path = os.path.join(IMAGES_DIR, fname)
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # Run model to get confidence map
        input_image = resize_to_input(rgb, (INPUT_HEIGHT, INPUT_WIDTH))
        input_val = np.expand_dims(input_image, axis=0)
        segmentation_model.set_tensor(model_input[0]["index"], input_val)
        segmentation_model.invoke()
        raw_output = segmentation_model.get_tensor(model_output[0]["index"])
        mask_256 = dequantize(
            raw_output,
            zero_points=model_output[0]["quantization_parameters"]["zero_points"],
            scales=model_output[0]["quantization_parameters"]["scales"],
        ).squeeze()
        confidence_map = resize_mask_to_original(mask_256, orig_h, orig_w)

        # Draw with light green background
        result_rgb = draw_segmentation(rgb, confidence_map, MASK_THRESHOLD, LIGHT_GREEN)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        out_name = f"segmented_{os.path.splitext(fname)[0]}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, result_bgr)
        print(f"[OUTPUT] {out_path}")


if __name__ == "__main__":
    test_inference_shapes_and_metadata()
    test_segmentation_detects_person()
    test_produce_output_images()
    print("\nAll tests passed.")
