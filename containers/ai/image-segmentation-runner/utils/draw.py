# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import cv2
import numpy as np

from utils.constants import EDGE_FEATHER_KERNEL


def draw_background(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """
    Tint the background of the frame, leaving the person pixels unchanged.

    The mask is feathered at model resolution and upscaled bilinearly, so the
    person outline is smooth at any frame size for the cost of a small image.

    Parameters
    ----------
    frame
        Input RGB image, shape (H, W, 3), dtype uint8.
    mask
        Binary person mask at model resolution, shape (h, w), dtype uint8 (1 = person).
    color
        RGB color of the background overlay.
    opacity
        Overlay opacity in [0, 1]: 1 replaces the background with the color, 0 leaves it untouched.

    Returns
    -------
    np.ndarray
        Annotated RGB frame, shape (H, W, 3), dtype uint8.
    """
    if opacity <= 0.0:
        return frame
    frame_h, frame_w = frame.shape[:2]
    person = cv2.GaussianBlur(mask.astype(np.float32), EDGE_FEATHER_KERNEL, 0)
    person = cv2.resize(person, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    # Overlay weight per pixel: full opacity on the background, none on the person
    alpha = ((1.0 - person) * opacity)[..., np.newaxis]
    overlay = np.array(color, dtype=np.float32)
    blended = frame.astype(np.float32) * (1.0 - alpha) + overlay * alpha
    return blended.astype(np.uint8)
