# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The data the image segmentation brick hands to an app."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Segmentation:
    """The people segmented in a frame.

    The model does not tell people apart: every person in view is part of the same
    segmentation, so with several people the area and the box cover all of them.

    Attributes:
        person_ratio (float): Fraction of the frame covered by people, in [0.0, 1.0].
            A rough measure of how close people are to the camera.
        confidence (float): Mean model confidence over the person pixels, in [0.0, 1.0].
        bounding_box_xyxy (tuple[int, int, int, int]): (x1, y1, x2, y2) box enclosing
            every person pixel, in frame coordinates.
    """

    person_ratio: float
    confidence: float
    bounding_box_xyxy: tuple[int, int, int, int]


def parse_segmentation(metadata: dict[str, Any]) -> Segmentation | None:
    """The segmentation of one runner result, or None when it found no person."""
    bbox = metadata.get("bounding_box_xyxy")
    if not metadata.get("person_detected") or not bbox:
        return None
    return Segmentation(
        person_ratio=float(metadata.get("person_ratio", 0.0)),
        confidence=float(metadata.get("confidence", 0.0)),
        bounding_box_xyxy=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
    )
