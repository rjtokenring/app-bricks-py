# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from PIL import ImageDraw
from PIL.Image import Image

from arduino.app_utils.image import label_font
from .detection import AprilTag

OUTLINE_COLOR = (0, 255, 0)
FIRST_CORNER_COLOR = (255, 0, 0)


def draw_tag(frame: Image, tag: AprilTag) -> Image:
    """Draws the outline of a detected tag and its id on an image.

    The outline follows the four corners of the tag, a dot marks its top-left corner and the id is written at
    its center.

    Args:
        frame (Image): The PIL Image object to draw on. This image will be modified in-place.
        tag (AprilTag): The tag to draw.

    Returns:
        Image: The same image, drawn on.
    """
    draw = ImageDraw.Draw(frame)
    points = [(float(x), float(y)) for x, y in tag.corners]
    side = max(abs(points[0][0] - points[2][0]), abs(points[0][1] - points[2][1]))
    width = max(2, round(side / 40))

    draw.polygon(points, outline=OUTLINE_COLOR, width=width)
    x, y = points[0]
    draw.ellipse((x - 2 * width, y - 2 * width, x + 2 * width, y + 2 * width), fill=FIRST_CORNER_COLOR)
    draw.text(tag.center, str(tag.id), fill=OUTLINE_COLOR, font=label_font(max(12, round(side / 4))), anchor="mm")
    return frame
