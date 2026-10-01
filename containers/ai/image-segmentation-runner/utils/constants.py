# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Model and inference constants."""

# Model input resolution (height, width). MediaPipe Selfie Segmentation expects 256x256 RGB.
INPUT_HEIGHT = 256
INPUT_WIDTH = 256

# Default per-pixel confidence above which a pixel belongs to a person
MASK_THRESHOLD = 0.5

# Person regions smaller than this fraction of the mask are noise: they are
# dropped from the mask, so they neither count as a person nor stretch the box
MIN_REGION_RATIO = 0.002

# Default background overlay: RGB color and opacity in [0, 1] (1 replaces the background)
BACKGROUND_COLOR = (68, 132, 255)
BACKGROUND_OPACITY = 1.0

# Gaussian kernel (pixels of the model mask) that feathers the person outline on the overlay
EDGE_FEATHER_KERNEL = (5, 5)
