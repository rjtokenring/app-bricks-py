# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Reading rotated text: recognize cutouts at several rotations, keep the best reading.

Port of EasyOCR's `rotation_info` (`easyocr.Reader.recognize` with
`make_rotated_img_list` / `set_result_with_confidence`), with one guard EasyOCR lacks.
Detection runs once, on the upright image: CRAFT finds vertical or upside-down text lines
as boxes just fine, it is the CRNN recognizer that only reads horizontal, left-to-right
text. So a cutout is recognized as cut and again rotated by each requested angle, and per
box the reading with the highest confidence wins.

The guard: a quarter turn (90/270) is only tried on cutouts that are taller than wide.
Rotating an ordinary horizontal text line by 90 degrees yields a narrow vertical strip
that the recognizer reads as a single character with high confidence - measured on the
board, such readings beat the correct upright ones ("ROBERTO GAINI" 0.64 lost to "L"
0.68). A line of vertical text, on the other hand, is a tall box and becomes a normal
horizontal strip once turned. 180 degrees keeps the shape and is always tried.

Second guard, also missing in EasyOCR: a rotated reading replaces the upright one only if
it is more confident by at least ROTATION_MARGIN. Upside-down readings of upright text can
still come out as confident nonsense ("Doltor" 0.75 lost to "JOHOQ" 0.82 on the board),
while genuinely rotated text reads upright with low confidence and rotated with high, so a
margin separates the two cases at little cost.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# The recognizer input is a horizontal strip, so only quarter turns make sense.
VALID_ROTATIONS = (90, 180, 270)
QUARTER_TURNS = (90, 270)

# How much more confident a rotated reading has to be to replace the upright one.
ROTATION_MARGIN = 0.1


def parse_rotations(value: object) -> list[int]:
    """
    Normalize a `rotations` configuration value into a list of angles.

    Parameters
    ----------
    value
        None, an int, or an iterable of ints (degrees). Angles are taken modulo 360;
        0 means the upright reading, which is always performed, and is dropped.

    Returns
    -------
    rotations : list[int]
        Distinct angles among 90, 180, 270, in the order given.

    Raises
    ------
    ValueError
        If an angle is not a multiple of 90, or the value is not a number/iterable.
    """
    if value is None or value == "":
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        value = [value]

    rotations: list[int] = []
    for item in value:
        try:
            angle = int(item) % 360
        except (TypeError, ValueError):
            raise ValueError(f"rotation {item!r} is not an integer number of degrees") from None
        if angle == 0:
            continue
        if angle not in VALID_ROTATIONS:
            raise ValueError(f"rotation {item!r} is not a multiple of 90 degrees (allowed: {', '.join(map(str, VALID_ROTATIONS))})")
        if angle not in rotations:
            rotations.append(angle)
    return rotations


def rotate_cutout(cutout: np.ndarray, angle: int) -> np.ndarray:
    """Rotate a [H, W] cutout counter-clockwise by a multiple of 90 degrees (0 returns it unchanged)."""
    if angle % 360 == 0:
        return cutout
    return np.ascontiguousarray(np.rot90(cutout, k=(angle // 90) % 4))


def angles_for_cutout(shape: tuple[int, ...], rotations: list[int]) -> list[int]:
    """
    The orientations to read one cutout at: upright first, then the applicable rotations.

    Quarter turns are only applied to cutouts taller than wide (see the module docstring);
    180 degrees always.
    """
    height, width = shape[0], shape[1]
    angles = [0]
    for angle in rotations:
        if angle in QUARTER_TURNS and height <= width:
            continue
        angles.append(angle)
    return angles


def plan_variants(cutouts: list[np.ndarray], rotations: list[int]) -> list[tuple[int, int]]:
    """
    Every (box_index, angle) reading to perform, box by box, upright first within a box.

    Parameters
    ----------
    cutouts
        The [h, w] greyscale crops, one per detected box.
    rotations
        Angles from `parse_rotations`.
    """
    return [(index, angle) for index, cutout in enumerate(cutouts) for angle in angles_for_cutout(cutout.shape, rotations)]


def select_best_readings(
    readings: list[tuple[str, float]],
    variants: list[tuple[int, int]],
    n_boxes: int,
) -> list[tuple[int, str, float]]:
    """
    Pick, for every box, the most confident of its readings.

    Parameters
    ----------
    readings
        (text, confidence) per entry of `variants`, in the same order.
    variants
        The (box_index, angle) plan from `plan_variants`.
    n_boxes
        Number of boxes; every box must appear in `variants` at least once.

    Returns
    -------
    best : list[tuple[int, str, float]]
        One (angle, text, confidence) per box. The upright reading is kept unless a
        rotated one is more confident by more than ROTATION_MARGIN; between rotated
        readings the earlier one wins ties.
    """
    if len(readings) != len(variants):
        raise ValueError(f"expected one reading per planned variant ({len(variants)}), got {len(readings)}")

    best: list[tuple[int, str, float] | None] = [None] * n_boxes
    for (box, angle), (text, confidence) in zip(variants, readings):
        current = best[box]
        margin = ROTATION_MARGIN if angle else 0.0
        if current is None or confidence > current[2] + margin:
            best[box] = (angle, text, confidence)
    missing = [index for index, item in enumerate(best) if item is None]
    if missing:
        raise ValueError(f"no reading planned for box(es) {missing}")
    return best  # type: ignore[return-value]
