# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model input quantization and segmentation mask post-processing."""

import cv2
import numpy as np


def dequantize(values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(
        ((np.int32(values) - np.int32(zero_points)) * np.float64(scales)),
        dtype=np.float32,
    )


def quantize(values: np.ndarray, zero_points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    z = np.asarray(zero_points, dtype=np.int32)
    s = np.asarray(scales, dtype=np.float64)

    q_float = np.rint(v / s) + z

    info = np.iinfo(np.uint8)
    q_clipped = np.clip(q_float, info.min, info.max)

    return q_clipped.astype(np.uint8, copy=False)


def person_mask(confidence_map: np.ndarray, threshold: float, min_region_ratio: float) -> np.ndarray:
    """
    Binary person mask: pixels above threshold, without the regions too small to be a person.

    Parameters
    ----------
    confidence_map
        Per-pixel person confidence, shape (H, W), dtype float32, values in [0, 1].
    threshold
        Confidence above which a pixel belongs to a person.
    min_region_ratio
        Connected regions smaller than this fraction of the mask are dropped.

    Returns
    -------
    np.ndarray
        Mask with shape (H, W), dtype uint8, 1 for person pixels and 0 elsewhere.
    """
    mask = (confidence_map > threshold).astype(np.uint8)
    min_area = min_region_ratio * mask.size
    if min_area <= 1 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area  # label 0 is the background
    return keep[labels]


def mask_bbox(mask: np.ndarray, frame_h: int, frame_w: int) -> list[int] | None:
    """
    Bounding box of the person pixels, scaled from mask to frame coordinates.

    Parameters
    ----------
    mask
        Binary mask with shape (H, W), as returned by `person_mask`.
    frame_h, frame_w
        Size of the frame the mask was computed on (the model input is a stretch of it).

    Returns
    -------
    list[int] | None
        [x1, y1, x2, y2] in frame pixels, inclusive, or None when the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    mask_h, mask_w = mask.shape
    sx, sy = frame_w / mask_w, frame_h / mask_h
    # Each mask pixel covers [i * s, (i + 1) * s) frame pixels
    return [
        int(xs.min() * sx),
        int(ys.min() * sy),
        min(frame_w - 1, int(np.ceil((xs.max() + 1) * sx)) - 1),
        min(frame_h - 1, int(np.ceil((ys.max() + 1) * sy)) - 1),
    ]
