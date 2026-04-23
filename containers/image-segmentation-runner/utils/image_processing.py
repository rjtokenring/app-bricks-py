import cv2
import numpy as np


def resize_to_input(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size using bilinear interpolation (simple stretch, no letterboxing).

    Parameters
    ----------
    image : np.ndarray
        Input image with shape (H, W, 3), dtype uint8, RGB layout.
    target_size : tuple[int, int]
        Target (height, width).

    Returns
    -------
    np.ndarray
        Resized image with shape (target_h, target_w, 3), dtype uint8.
    """
    target_h, target_w = target_size
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_mask_to_original(mask: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """
    Resize a 2D mask back to the original image dimensions.

    Parameters
    ----------
    mask : np.ndarray
        Mask with shape (H, W), dtype float32, values in [0, 1].
    orig_h : int
        Original image height.
    orig_w : int
        Original image width.

    Returns
    -------
    np.ndarray
        Resized mask with shape (orig_h, orig_w), dtype float32.
    """
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
