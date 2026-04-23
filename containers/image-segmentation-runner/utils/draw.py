import cv2
import numpy as np


def draw_segmentation(
    frame: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
    overlay_color: tuple[int, int, int] = (68, 132, 255),
) -> np.ndarray:
    """
    Blend a segmentation mask overlay onto the frame.
    Pixels where mask > threshold are considered foreground (person) and left unchanged.
    Background pixels are tinted with the overlay color.

    Parameters
    ----------
    frame : np.ndarray
        Input RGB image, shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Segmentation mask, shape (H, W), dtype float32, values in [0, 1].
        Values > threshold = person (foreground).
    threshold : float
        Mask threshold for binary segmentation.
    overlay_color : tuple[int, int, int]
        RGB color for background overlay.

    Returns
    -------
    np.ndarray
        Annotated RGB frame with background overlay, shape (H, W, 3), dtype uint8.
    """
    # Smooth the mask at original resolution to reduce jagged edges
    alpha = cv2.GaussianBlur(mask, (7, 7), 0)

    # Expand alpha to 3 channels for blending
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    # Create solid overlay
    overlay = np.full_like(frame, overlay_color, dtype=np.uint8)

    # Blend: foreground = original * alpha + overlay * (1 - alpha)
    blended = frame.astype(np.float32) * alpha_3ch + overlay.astype(np.float32) * (1.0 - alpha_3ch)

    return blended.astype(np.uint8)
