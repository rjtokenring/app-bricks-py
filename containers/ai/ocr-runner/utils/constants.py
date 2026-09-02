# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Constants for the EasyOCR LiteRT pipeline.

Values mirror `qai_hub_models/models/easyocr/{app,model}.py` (ai-hub-models v0.61.0)
and `easyocr/config.py` (JaidedAI/EasyOCR).
"""

# --- Model files ------------------------------------------------------------

DETECTOR_MODEL_PATH = "models/easyocr-tflite-w8a8/detector.tflite"
RECOGNIZER_MODEL_PATH = "models/easyocr-tflite-w8a8/recognizer.tflite"

# Network input resolutions, as exported by ai-hub-models.
# detector:   [1, 3, 608, 800] RGB   float [0, 1]
# recognizer: [1, 1,  64, 800] GREY  float [0, 1]
DETECTOR_INPUT_HEIGHT = 608
DETECTOR_INPUT_WIDTH = 800
RECOGNIZER_INPUT_HEIGHT = 64
RECOGNIZER_INPUT_WIDTH = 800

# The CRAFT detector emits score maps at half the input resolution.
DETECTOR_OUTPUT_STRIDE = 2

# --- Detector post-processing (qai_hub_models .. app.DETECTOR_ARGS) ---------

DETECTOR_ARGS = {
    "text_threshold": 0.7,
    "link_threshold": 0.4,
    "low_text": 0.4,
    "poly": False,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
    "min_size": 20,
}

# --- Recognizer post-processing (qai_hub_models .. app.RECOGNIZER_ARGS) ----

RECOGNIZER_ARGS = {
    "allowlist": None,
    "blocklist": None,
    "contrast_ths": 0.1,
    "adjust_contrast": 0.5,
}

# --- Character set ----------------------------------------------------------

# english_g2, the "standard" recognizer network EasyOCR selects for lang_list=["en"].
# The CTC converter prepends a [blank] token, so the model has len(CHARACTERS) + 1 classes.
NUMBER = "0123456789"
SYMBOL = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \u20ac"
EN_CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHARACTERS = NUMBER + SYMBOL + EN_CHAR

# Characters that belong to the selected language(s). Anything in CHARACTERS but not
# in LANG_CHAR is zeroed out of the logits before decoding. For English the two sets
# are identical, so nothing is filtered.
LANG_CHAR = CHARACTERS
