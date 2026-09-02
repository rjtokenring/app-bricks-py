# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Thin LiteRT (ai-edge-litert) wrapper.

Hides the two things that differ between the exported EasyOCR .tflite variants:
  * tensor layout - the ai-hub-models input specs are NCHW, but the TFLite export
    applies runtime channel reordering and usually ends up NHWC.
  * quantization - the w8a8 assets take int8/uint8 tensors with per-tensor scales.
"""

from __future__ import annotations

import numpy as np
from ai_edge_litert.interpreter import Delegate, Interpreter


def dequantize(tensor: np.ndarray, zero_points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Map an integer LiteRT tensor back to float using its quantization parameters."""
    if scales is None or len(scales) == 0:
        return tensor.astype(np.float32)
    return (tensor.astype(np.float32) - np.asarray(zero_points, dtype=np.float32)) * np.asarray(scales, dtype=np.float32)


def quantize(array: np.ndarray, zero_points: np.ndarray, scales: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Map a float array onto an integer LiteRT tensor using its quantization parameters."""
    info = np.iinfo(dtype)
    quantized = np.round(array / np.asarray(scales, dtype=np.float32)) + np.asarray(zero_points, dtype=np.float32)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _detect_layout(shape: tuple[int, ...]) -> str:
    """
    Guess whether a 4D input tensor is NHWC or NCHW.

    The channel axis is the only one that can hold 1 (grey) or 3 (RGB) for these two
    models, and the spatial axes are always much larger, so the ambiguity resolves.
    """
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D input tensor, got shape {shape}")
    channels_last = shape[3] in (1, 3)
    channels_first = shape[1] in (1, 3)
    if channels_last and not channels_first:
        return "NHWC"
    if channels_first and not channels_last:
        return "NCHW"
    raise ValueError(f"Cannot infer tensor layout from input shape {shape}")


class LiteRTModel:
    """A single-input LiteRT model that consumes NHWC float arrays."""

    def __init__(self, model_path: str, delegates: list[Delegate] | None = None, num_threads: int | None = None) -> None:
        self.model_path = model_path
        self.interpreter = Interpreter(
            model_path,
            experimental_delegates=delegates,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = tuple(int(x) for x in self.input_details[0]["shape"])
        self.layout = _detect_layout(input_shape)
        if self.layout == "NHWC":
            _, self.height, self.width, self.channels = input_shape
        else:
            _, self.channels, self.height, self.width = input_shape

    @property
    def image_shape(self) -> tuple[int, int]:
        """Network input resolution as (height, width)."""
        return (self.height, self.width)

    def __call__(self, nhwc_input: np.ndarray) -> list[np.ndarray]:
        """
        Run the model.

        Parameters
        ----------
        nhwc_input
            [N, H, W, C] float32 array already scaled to the range the network expects.

        Returns
        -------
        outputs : list[np.ndarray]
            One dequantized float32 array per model output, in `get_output_details()` order.
        """
        details = self.input_details[0]
        tensor = nhwc_input if self.layout == "NHWC" else nhwc_input.transpose(0, 3, 1, 2)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        if details["dtype"] != np.float32:
            params = details["quantization_parameters"]
            tensor = quantize(tensor, params["zero_points"], params["scales"], details["dtype"])

        self.interpreter.set_tensor(details["index"], tensor)
        self.interpreter.invoke()

        outputs = []
        for output in self.output_details:
            raw = self.interpreter.get_tensor(output["index"])
            params = output["quantization_parameters"]
            if output["dtype"] == np.float32:
                outputs.append(raw.astype(np.float32))
            else:
                outputs.append(dequantize(raw, params["zero_points"], params["scales"]))
        return outputs
