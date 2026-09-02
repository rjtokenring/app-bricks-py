# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Numpy/OpenCV ports of the qai_hub_models image helpers used by EasyOCR."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np


def resize_pad(
    image: np.ndarray,
    dst_size: tuple[int, int],
    pad_value: float = 0.0,
    vertical_float: Literal["center", "top", "bottom"] = "center",
    horizontal_float: Literal["center", "left", "right"] = "center",
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize and pad an image to (dst_size[0], dst_size[1]) without warping or cropping.

    Port of `qai_hub_models.utils.image_processing.resize_pad`.

    Parameters
    ----------
    image
        Input image with shape (H, W) or (H, W, C).
    dst_size
        Desired (height, width).
    pad_value
        Constant value written into the padded region.
    vertical_float
        Where the image floats vertically in the resulting canvas.
    horizontal_float
        Where the image floats horizontally in the resulting canvas.

    Returns
    -------
    rescaled_padded_image : np.ndarray
        Image of shape (dst_h, dst_w) or (dst_h, dst_w, C).
    scale : float
        Scale factor applied to the original image (identical for H and W).
    padding : (int, int)
        (pad_left, pad_top) applied to the resized image.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H, W) or 3D (H, W, C)")

    src_h, src_w = image.shape[:2]
    dst_h, dst_w = int(dst_size[0]), int(dst_size[1])

    scale = min(dst_h / src_h, dst_w / src_w)
    new_h = max(1, math.floor(src_h * scale))
    new_w = max(1, math.floor(src_w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top, pad_bottom = _split_padding(vertical_float, dst_h - new_h)
    pad_left, pad_right = _split_padding(horizontal_float, dst_w - new_w)

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=float(pad_value),
    )

    return padded, scale, (pad_left, pad_top)


def _split_padding(float_img_in_frame: str, pad_size: int) -> tuple[int, int]:
    """Split padding into (left, right) or (top, bottom) based on where the image floats."""
    if float_img_in_frame == "center":
        return (int(pad_size // 2), int(pad_size // 2 + pad_size % 2))
    if float_img_in_frame in ("right", "bottom"):
        return (pad_size, 0)
    if float_img_in_frame in ("left", "top"):
        return (0, pad_size)
    raise ValueError(f"Invalid pad type: {float_img_in_frame}")


def denormalize_coordinates(
    coordinates: np.ndarray,
    input_img_size: tuple[int, int],
    scale: float = 1.0,
    pad: tuple[int, int] = (0, 0),
) -> None:
    """
    Map detection coordinates back into the original (pre-resize) image, in place.

    Port of `qai_hub_models.utils.image_processing.denormalize_coordinates`.

    Parameters
    ----------
    coordinates
        Array of shape [..., 2]. Modified in place.
    input_img_size
        Size of the tensor fed to the network, in the same axis order as `coordinates`.
        Pass (1, 1) when `coordinates` already holds absolute pixel values.
    scale
        Scale factor used to resize the original image for network inference.
    pad
        Padding added during that resize, in the same axis order as `coordinates`.
    """
    img_0, img_1 = input_img_size
    pad_0, pad_1 = pad

    coordinates[..., 0] = ((coordinates[..., 0] * img_0 - pad_0) / scale).astype(np.int32)
    coordinates[..., 1] = ((coordinates[..., 1] * img_1 - pad_1) / scale).astype(np.int32)


def four_point_transform(image: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """
    Warp the parallelogram `rect` out of `image` into an axis-aligned crop.

    Port of `easyocr.utils.four_point_transform`.

    Parameters
    ----------
    image
        Source image.
    rect
        [4, 2] float32 corners, ordered top-left, top-right, bottom-right, bottom-left.

    Returns
    -------
    warped : np.ndarray
        The rectified crop.
    """
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def adjust_contrast(image: np.ndarray, contrast_factor: float) -> np.ndarray:
    """
    Blend an image towards its own mean, matching `torchvision.transforms.functional.adjust_contrast`.

    Parameters
    ----------
    image
        Single-channel float image in the range [0, 1].
    contrast_factor
        > 1 increases contrast, < 1 flattens it.

    Returns
    -------
    adjusted : np.ndarray
        Image of the same shape, clipped back to [0, 1].
    """
    mean = float(np.mean(image, dtype=np.float64))
    blended = contrast_factor * image.astype(np.float32) + (1.0 - contrast_factor) * mean
    return np.clip(blended, 0.0, 1.0)
