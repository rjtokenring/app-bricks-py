# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from utils.tf import load_qnn_delegate

from utils.constants import (
    INPUT_WIDTH,
    INPUT_HEIGHT,
    MASK_THRESHOLD,
    OVERLAY_COLOR,
)
from utils.image_processing import resize_to_input, resize_mask_to_original
from utils.model_io_processing import dequantize
from utils.draw import draw_segmentation

# Load model
segmentation_model = Interpreter(
    "models/mediapipe_selfie-mediapipe-selfie-segmentation-w8a8.tflite",
    experimental_delegates=load_qnn_delegate(),
)
segmentation_model.allocate_tensors()

model_input = segmentation_model.get_input_details()
model_output = segmentation_model.get_output_details()


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Process a single frame through the selfie segmentation pipeline.

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3), dtype uint8.

    Returns:
        tuple[np.ndarray, dict]: contains (annotated_frame, metadata), where
            annotated_frame: RGB frame with background overlay applied.
            metadata (JSON-serializable) contains:
                - 'person_detected': bool, whether any person pixel was found
                - 'person_ratio': float, fraction of pixels classified as person [0, 1]
                - 'mean_confidence': float, average confidence across all pixels
                - 'bounding_box_xyxy': [x1, y1, x2, y2] bounding box of the person
                  region in pixel coordinates, or None if no person detected
    """
    orig_h, orig_w = rgb_frame.shape[:2]

    # Preprocess: resize to model input size (simple bilinear stretch, no letterboxing)
    # Model expects uint8 input [1, 256, 256, 3] with quantization scale ~1/255, zero_point=0
    # So we feed raw uint8 RGB pixels directly.
    input_image = resize_to_input(rgb_frame, (INPUT_HEIGHT, INPUT_WIDTH))
    input_val = np.expand_dims(input_image, axis=0)  # [1, 256, 256, 3]

    # Run segmentation model
    segmentation_model.set_tensor(model_input[0]["index"], input_val)
    segmentation_model.invoke()

    # Get and dequantize output: [1, 256, 256, 1] uint8 -> float32 [0, 1]
    raw_output = segmentation_model.get_tensor(model_output[0]["index"])
    mask_256 = dequantize(
        raw_output,
        zero_points=model_output[0]["quantization_parameters"]["zero_points"],
        scales=model_output[0]["quantization_parameters"]["scales"],
    )

    # Squeeze to [256, 256]
    mask_256 = mask_256.squeeze()

    # Resize mask to original frame dimensions
    confidence_map = resize_mask_to_original(mask_256, orig_h, orig_w)

    # Binary mask
    segmentation_mask = (confidence_map > MASK_THRESHOLD).astype(np.uint8)

    # Draw overlay on frame
    annotated_frame = draw_segmentation(rgb_frame, confidence_map, MASK_THRESHOLD, OVERLAY_COLOR)

    # Build JSON-serializable metadata
    person_pixels = int(segmentation_mask.sum())
    total_pixels = orig_h * orig_w
    person_detected = person_pixels > 0

    bounding_box = None
    if person_detected:
        ys, xs = np.where(segmentation_mask == 1)
        bounding_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    metadata = {
        "person_detected": person_detected,
        "person_ratio": person_pixels / total_pixels,
        "mean_confidence": float(confidence_map.mean()),
        "bounding_box_xyxy": bounding_box,
    }

    return annotated_frame, metadata
