# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pytest configuration for the image-segmentation-runner container tests.

Puts the container directory on ``sys.path`` and makes it the working
directory, so ``inference`` and ``utils`` import and the model loads from
``models/`` the same way they do in the image (WORKDIR /app).
"""

import os
import sys

CONTAINER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CONTAINER_DIR not in sys.path:
    sys.path.insert(0, CONTAINER_DIR)
os.chdir(CONTAINER_DIR)
