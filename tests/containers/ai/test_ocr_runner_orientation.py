# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Rotated-text reading in the ocr-runner (the `rotations` setting): angle parsing,
cutout rotation, which rotations apply to which cutout, and the per-box choice of the
most confident reading.

The module is loaded standalone by file path (it only depends on numpy), like the other
ocr-runner tests, so the runner's `utils` package name cannot clash with other runners'.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "containers" / "ai" / "ocr-runner" / "utils" / "orientation.py"
_spec = importlib.util.spec_from_file_location("ocr_orientation", _MODULE_PATH)
orientation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orientation)

WIDE = np.zeros((20, 120), dtype=np.uint8)  # an ordinary horizontal text line
TALL = np.zeros((120, 20), dtype=np.uint8)  # a vertical text line
SQUARE = np.zeros((40, 40), dtype=np.uint8)


# --- parse_rotations -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ([], []),
        (90, [90]),
        ([90, 180, 270], [90, 180, 270]),
        ([270, 90], [270, 90]),  # order is kept
        ([90, 90, 180], [90, 180]),  # duplicates dropped
        ([0, 360, 90], [90]),  # upright is always read: 0 is a no-op
        ([-90], [270]),  # normalized modulo 360
        (["90", 180.0], [90, 180]),  # JSON may carry strings or floats
    ],
)
def test_parse_rotations_normalizes(value, expected):
    assert orientation.parse_rotations(value) == expected


@pytest.mark.parametrize("value", [[45], [30, 90], ["ninety"], [None], [1.5]])
def test_parse_rotations_rejects_non_quarter_turns(value):
    with pytest.raises(ValueError):
        orientation.parse_rotations(value)


# --- rotate_cutout -------------------------------------------------------------------------


def test_rotate_cutout_turns_a_vertical_strip_horizontal():
    tall = np.arange(12, dtype=np.uint8).reshape(6, 2)
    for angle in (90, 270):
        rotated = orientation.rotate_cutout(tall, angle)
        assert rotated.shape == (2, 6)
        assert rotated.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(orientation.rotate_cutout(tall, 180), tall[::-1, ::-1])
    np.testing.assert_array_equal(orientation.rotate_cutout(tall, 90), np.rot90(tall))
    assert orientation.rotate_cutout(tall, 0) is tall


# --- which rotations apply to which cutout -------------------------------------------------


def test_quarter_turns_only_apply_to_cutouts_taller_than_wide():
    """Rotating a horizontal line by 90 degrees gives a narrow strip the recognizer reads as one
    confident garbage character (measured on the board), so 90/270 are reserved for tall boxes."""
    rotations = [90, 180, 270]
    assert orientation.angles_for_cutout(WIDE.shape, rotations) == [0, 180]
    assert orientation.angles_for_cutout(SQUARE.shape, rotations) == [0, 180]
    assert orientation.angles_for_cutout(TALL.shape, rotations) == [0, 90, 180, 270]
    assert orientation.angles_for_cutout(TALL.shape, []) == [0]


def test_plan_variants_lists_upright_first_per_box():
    plan = orientation.plan_variants([WIDE, TALL, WIDE], [90, 270])
    assert plan == [(0, 0), (1, 0), (1, 90), (1, 270), (2, 0)]
    assert orientation.plan_variants([], [90]) == []


# --- select_best_readings ------------------------------------------------------------------


def test_select_best_readings_picks_the_most_confident_variant_per_box():
    variants = [(0, 0), (1, 0), (1, 90), (1, 270), (2, 0), (2, 180)]
    readings = [("a", 0.4), ("|", 0.3), ("VERTICAL", 0.9), ("7VC1L83A", 0.5), ("c", 0.6), ("ɔ", 0.2)]
    assert orientation.select_best_readings(readings, variants, n_boxes=3) == [(0, "a", 0.4), (90, "VERTICAL", 0.9), (0, "c", 0.6)]


def test_select_best_readings_prefers_upright_on_ties():
    variants = [(0, 0), (0, 180)]
    assert orientation.select_best_readings([("a", 0.5), ("e", 0.5)], variants, n_boxes=1) == [(0, "a", 0.5)]
    assert orientation.select_best_readings([], [], n_boxes=0) == []


def test_rotated_reading_must_beat_upright_by_the_margin():
    """Measured on the board: the 180-degree reading of an upright word can be confident nonsense
    ("Doltor" 0.75 vs "JOHOQ" 0.82). A small edge is not enough to flip the reading."""
    variants = [(0, 0), (0, 180)]
    margin = orientation.ROTATION_MARGIN
    assert orientation.select_best_readings([("Doltor", 0.75), ("JOHOQ", 0.82)], variants, n_boxes=1) == [(0, "Doltor", 0.75)]
    assert orientation.select_best_readings([("Doltor", 0.75), ("JOHOQ", 0.75 + margin + 0.01)], variants, n_boxes=1)[0][1] == "JOHOQ"
    # Genuinely rotated text: nonsense upright at low confidence, clear text rotated.
    assert orientation.select_best_readings([("|!", 0.2), ("ROTATED", 0.9)], variants, n_boxes=1) == [(180, "ROTATED", 0.9)]


def test_select_best_readings_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        orientation.select_best_readings([("a", 0.5)], [(0, 0), (0, 180)], n_boxes=1)
    with pytest.raises(ValueError):
        orientation.select_best_readings([("a", 0.5)], [(0, 0)], n_boxes=2)  # box 1 never planned
