# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Text front end of the Kokoro runner: sentences, phonemes and model tokens.

Kokoro reads phonemes (IPA) and punctuation, not text. The phonemes come from the
espeak-ng program, run as a child process: it drops the punctuation, which the model
uses for its prosody, so the text is phonemized between punctuation marks and the
marks are put back in place.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

# The longest phoneme sequence the model takes in one run
MAX_TOKENS = 510

# espeak-ng voice of each Kokoro language, keyed by the first letter of a voice name.
# The Japanese (j) and Mandarin (z) voices have no entry: their phonemes come from
# dedicated dictionaries, espeak-ng's do not match what the model was trained on.
ESPEAK_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

_SENTENCE_END = re.compile("(?<=[.!?…])\\s+|\\n+")
# The marks Kokoro has a token for, and the inverted ones that only open a clause
_PUNCTUATION = re.compile('([;:,.!?¡¿—…"()“”]+)')
# espeak-ng flags a word it reads in another language: "(en)pizza(it)"
_LANGUAGE_SWITCH = re.compile(r"\([a-z-]+\)")

Espeak = Callable[[str, str], str]


def load_vocab(path: Path = Path(__file__).with_name("vocab.json")) -> dict[str, int]:
    """The model's token id of every phoneme and punctuation mark it knows."""
    return json.loads(path.read_text(encoding="utf-8"))


def voice_language(voice: str) -> str | None:
    """The espeak-ng language of a Kokoro voice, None for a voice this runner cannot phonemize."""
    return ESPEAK_LANGUAGES.get(voice[:1])


def sentences(text: str) -> list[str]:
    """Split text after the end of each sentence and at line breaks, dropping the empty pieces."""
    return [piece.strip() for piece in _SENTENCE_END.split(text) if piece.strip()]


def espeak_ipa(text: str, language: str) -> str:
    """Phonemes of a piece of text without punctuation, as espeak-ng writes them in IPA."""
    # The newline ends the text: espeak-ng 1.50 loses the last character of a stdin without one.
    result = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "-v", language, "--stdin"],
        input=text + "\n",
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    # One output line per clause espeak-ng found
    return " ".join(result.stdout.split())


def phonemize(text: str, language: str, espeak: Espeak = espeak_ipa) -> str:
    """Phonemes of text, with its punctuation marks where they were."""
    phonemes = ""
    for index, piece in enumerate(_PUNCTUATION.split(text)):
        if index % 2:
            phonemes += piece
        else:
            words = piece.strip()
            if piece[:1].isspace():
                phonemes += " "
            if words:
                phonemes += _LANGUAGE_SWITCH.sub("", espeak(words, language))
                if piece[-1:].isspace():
                    phonemes += " "
    return phonemes.strip()


def tokenize(phonemes: str, vocab: dict[str, int]) -> list[list[int]]:
    """Token ids of the phonemes, in runs the model takes one at a time.

    The characters the model has no token for are dropped. A sequence longer than
    MAX_TOKENS is cut at the last space that fits, or at MAX_TOKENS without one.
    """
    known = "".join(char for char in phonemes if char in vocab)
    runs: list[list[int]] = []
    while known:
        cut = len(known)
        if cut > MAX_TOKENS:
            space = known.rfind(" ", 0, MAX_TOKENS + 1)
            cut = space if space > 0 else MAX_TOKENS
        runs.append([vocab[char] for char in known[:cut]])
        known = known[cut:].lstrip()
    return runs
