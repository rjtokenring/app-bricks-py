# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import os

import cv2
import numpy as np
import pytest

import inference
from inference import apply_config, inference_callback
from utils.constants import BACKGROUND_COLOR, BACKGROUND_OPACITY, MASK_THRESHOLD
from utils.model_io_processing import mask_bbox, person_mask

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
IMAGES = ("person1.jpg", "person2.jpg")


def _load_rgb(name: str) -> np.ndarray:
    bgr = cv2.imread(os.path.join(IMAGES_DIR, name))
    assert bgr is not None, f"could not load {name}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(autouse=True)
def _default_config():
    """Every test starts, and leaves, the runner with the default settings."""
    defaults = {"mask_threshold": MASK_THRESHOLD, "background_color": BACKGROUND_COLOR, "background_opacity": BACKGROUND_OPACITY}
    inference._config.update(defaults)
    yield
    inference._config.update(defaults)


@pytest.mark.parametrize("name", IMAGES)
def test_inference_returns_frame_and_json_metadata(name: str):
    rgb = _load_rgb(name)
    h, w = rgb.shape[:2]

    annotated, metadata = inference_callback(rgb)

    assert annotated.shape == (h, w, 3)
    assert annotated.dtype == np.uint8
    json.dumps(metadata)
    assert set(metadata) == {"person_detected", "person_ratio", "confidence", "bounding_box_xyxy"}


@pytest.mark.parametrize("name", IMAGES)
def test_person_is_detected_with_a_box_inside_the_frame(name: str):
    rgb = _load_rgb(name)
    h, w = rgb.shape[:2]

    _, metadata = inference_callback(rgb)

    assert metadata["person_detected"] is True
    assert 0.05 < metadata["person_ratio"] < 1.0
    assert MASK_THRESHOLD < metadata["confidence"] <= 1.0
    x1, y1, x2, y2 = metadata["bounding_box_xyxy"]
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
    assert 0 <= x1 < x2 < w and 0 <= y1 < y2 < h


def test_background_is_tinted_and_person_kept(tmp_path):
    # The faces are blurred for privacy: in person1.jpg the blur hides most of the person, person2.jpg shows the body
    rgb = _load_rgb("person2.jpg")
    apply_config({"background_color": [0, 255, 0]})

    annotated, metadata = inference_callback(rgb)
    cv2.imwrite(str(tmp_path / "segmented_person2.jpg"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # The top-left corner is background: replaced by the color
    assert tuple(annotated[0, 0]) == (0, 255, 0)
    # The box center is the person: left as it was
    x1, y1, x2, y2 = metadata["bounding_box_xyxy"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    assert np.array_equal(annotated[cy, cx], rgb[cy, cx])


def test_zero_opacity_leaves_the_frame_untouched():
    rgb = _load_rgb("person1.jpg")
    apply_config({"background_opacity": 0.0})

    annotated, _ = inference_callback(rgb)

    assert np.array_equal(annotated, rgb)


def test_threshold_one_detects_nobody():
    apply_config({"mask_threshold": 1.0})

    _, metadata = inference_callback(_load_rgb("person1.jpg"))

    assert metadata == {"person_detected": False, "person_ratio": 0.0, "confidence": 0.0, "bounding_box_xyxy": None}


def test_apply_config_clamps_and_ignores_malformed_values():
    apply_config({"mask_threshold": 3, "background_opacity": -1, "background_color": [300, -5, 10], "unknown": 1})
    assert inference._config == {"mask_threshold": 1.0, "background_color": (255, 0, 10), "background_opacity": 0.0}

    apply_config({"background_color": "red"})
    assert inference._config["background_color"] == (255, 0, 10)


def test_person_mask_drops_regions_smaller_than_the_minimum():
    confidence = np.zeros((100, 100), dtype=np.float32)
    confidence[10:60, 10:40] = 0.9  # a person: 1500 px
    confidence[90:92, 90:92] = 0.9  # a speck: 4 px

    mask = person_mask(confidence, threshold=0.5, min_region_ratio=0.01)

    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 1500
    assert mask[91, 91] == 0


def test_mask_bbox_scales_mask_pixels_to_frame_pixels():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = mask[2, 2] = 1

    # Each mask pixel covers 10x20 frame pixels: the box spans mask pixels 1..2 inclusive
    assert mask_bbox(mask, frame_h=80, frame_w=40) == [10, 20, 29, 59]
    assert mask_bbox(np.zeros((4, 4), dtype=np.uint8), 80, 40) is None
    # Full mask: the box is the whole frame, never past its edges
    assert mask_bbox(np.ones((3, 3), dtype=np.uint8), frame_h=100, frame_w=100) == [0, 0, 99, 99]
