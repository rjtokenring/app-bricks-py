# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np


def resize_to_input(image: np.ndarray, dst_size: tuple[int, int]) -> np.ndarray:
    """
    Resize image to dst_size with bilinear interpolation, stretching it (no letterboxing).

    The selfie segmentation model is trained on stretched inputs, and stretching
    keeps the mapping back to the frame a plain per-axis scale.

    Parameters
    ----------
    image
        Input image with shape (H, W, 3), dtype uint8, RGB layout.
    dst_size
        Desired (height, width).

    Returns
    -------
    np.ndarray
        Resized image with shape (dst_h, dst_w, 3), dtype uint8.
    """
    dst_h, dst_w = dst_size
    return cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
