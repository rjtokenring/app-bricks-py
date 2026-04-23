import numpy as np


def dequantize(
    values: np.ndarray,
    zero_points: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """
    Dequantize model output values.

    Parameters
    ----------
    values : np.ndarray
        Raw quantized output tensor.
    zero_points : np.ndarray
        Zero points from quantization_parameters.
    scales : np.ndarray
        Scales from quantization_parameters.

    Returns
    -------
    np.ndarray
        Dequantized float32 tensor.
    """
    if zero_points.size == 0 or scales.size == 0:
        return values.astype(np.float32)

    return ((values - np.int32(zero_points)) * np.float64(scales)).astype(np.float32)
