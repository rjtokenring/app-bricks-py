# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pytest configuration for the kokoro-tts-runner container tests.

Adds the container ``src`` directory to ``sys.path`` so the modules import the same way
they do at runtime, and provides a synthesizer on a scripted model and phonemizer: the
tests need neither the ONNX weights nor espeak-ng.
"""

import os
import sys

import numpy as np
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from synthesizer import Synthesizer  # noqa: E402
from text import load_vocab  # noqa: E402


class FakeSession:
    """Records the inputs of each run and answers 10 ms of audio per token."""

    def __init__(self):
        self.runs = []
        self.fail = False

    def run(self, output_names, input_feed):
        if self.fail:
            raise RuntimeError("model exploded")
        self.runs.append(input_feed)
        return [np.full(input_feed["input_ids"].shape[1] * 240, 0.5, dtype=np.float32)]


def fake_espeak(text, language):
    """Spell the text back in characters the vocabulary knows, tagged with the language."""
    return f"{language[0]}{text.lower()}"


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def synthesizer(session):
    # The style vector at index n is filled with n, so a test can tell which one was picked
    styles = np.arange(510, dtype=np.float32).reshape(510, 1, 1).repeat(256, axis=2)
    voices = {"af_heart": styles, "if_sara": styles, "jf_alpha": styles}
    return Synthesizer(session, voices, load_vocab(), espeak=fake_espeak)
