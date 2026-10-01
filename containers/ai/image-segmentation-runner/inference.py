# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from utils.constants import BACKGROUND_COLOR, BACKGROUND_OPACITY, INPUT_HEIGHT, INPUT_WIDTH, MASK_THRESHOLD, MIN_REGION_RATIO
from utils.tf import load_qnn_delegate
from utils.image_processing import resize_to_input
from utils.model_io_processing import dequantize, mask_bbox, person_mask, quantize
from utils.draw import draw_background


# Load model
segmenter = Interpreter(
    "models/mediapipe_selfie-mediapipe-selfie-segmentation-w8a8.tflite",
    experimental_delegates=load_qnn_delegate(),
)
segmenter.allocate_tensors()

segmenter_input = segmenter.get_input_details()
segmenter_output = segmenter.get_output_details()

# Runtime-tunable settings, updated by client config messages (wired in the
# base image's main.py).
_config = {
    "mask_threshold": MASK_THRESHOLD,
    "background_color": BACKGROUND_COLOR,
    "background_opacity": BACKGROUND_OPACITY,
}


def apply_config(config: dict) -> None:
    """Apply a client configuration payload; unknown keys and malformed values are ignored."""
    for key in ("mask_threshold", "background_opacity"):
        value = config.get(key)
        if value is not None:
            _config[key] = max(0.0, min(1.0, float(value)))
            print(f"config: {key} set to {_config[key]}", flush=True)
    value = config.get("background_color")
    if value is not None:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            _config["background_color"] = tuple(max(0, min(255, int(channel))) for channel in value)
            print(f"config: background_color set to {_config['background_color']}", flush=True)
        else:
            print(f"config: background_color ignored, expected [r, g, b], got {value!r}", flush=True)


def _set_input(rgb_input: np.ndarray) -> None:
    """Quantize (if needed) and feed the preprocessed RGB input into the model.

    Args:
        rgb_input: Preprocessed RGB image of shape [1, H, W, 3], dtype uint8 in range [0, 255].
    """
    detail = segmenter_input[0]
    normalized = rgb_input.astype(np.float32) / 255.0  # the model input range is [0, 1]
    if np.issubdtype(detail["dtype"], np.integer):
        input_val = quantize(
            normalized,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    else:
        input_val = normalized.astype(detail["dtype"])
    segmenter.set_tensor(detail["index"], input_val)


def _get_confidence_map() -> np.ndarray:
    """Read the model output as a per-pixel person confidence map of shape (H, W) in [0, 1]."""
    detail = segmenter_output[0]
    tensor = segmenter.get_tensor(detail["index"])
    if np.issubdtype(detail["dtype"], np.integer):
        tensor = dequantize(
            tensor,
            zero_points=detail["quantization_parameters"]["zero_points"],
            scales=detail["quantization_parameters"]["scales"],
        )
    # [1, H, W, 1] -> (H, W)
    return np.clip(tensor.reshape(tensor.shape[1], tensor.shape[2]), 0.0, 1.0)


def inference_callback(rgb_frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Process a single frame through the selfie segmentation pipeline.

    Args:
        rgb_frame: Input frame as RGB np.ndarray (H, W, 3), dtype uint8.

    Returns:
        tuple[np.ndarray, dict]: contains (annotated_frame, metadata), where
            annotated_frame is the frame with the background overlay applied and
            metadata contains:
                - 'person_detected': bool, whether any person region was found
                - 'person_ratio': float, fraction of the frame covered by people, in [0, 1]
                - 'confidence': float, mean model confidence over the person pixels
                  (0.0 when no person is detected)
                - 'bounding_box_xyxy': list [x1, y1, x2, y2] enclosing every person
                  pixel, in frame coordinates, or None when no person is detected
    """
    frame_h, frame_w = rgb_frame.shape[:2]

    input_val = resize_to_input(rgb_frame, (INPUT_HEIGHT, INPUT_WIDTH))
    _set_input(np.expand_dims(input_val, axis=0))
    segmenter.invoke()

    # Everything but the overlay works at model resolution: the input is a
    # stretch of the frame, so ratios match and boxes only need a per-axis scale.
    confidence_map = _get_confidence_map()
    mask = person_mask(confidence_map, _config["mask_threshold"], MIN_REGION_RATIO)
    person_pixels = int(mask.sum())

    metadata = {
        "person_detected": person_pixels > 0,
        "person_ratio": person_pixels / mask.size,
        "confidence": float(confidence_map[mask == 1].mean()) if person_pixels else 0.0,
        "bounding_box_xyxy": mask_bbox(mask, frame_h, frame_w),
    }

    annotated_frame = draw_background(rgb_frame, mask, _config["background_color"], _config["background_opacity"])
    return annotated_frame, metadata
