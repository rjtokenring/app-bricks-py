# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Synthesizer: model inputs, one PCM block per sentence, resampling."""

import numpy as np
import pytest

from synthesizer import MODEL_SAMPLE_RATE, UnknownVoiceError, to_pcm16
from text import MAX_TOKENS


def test_only_the_voices_espeak_can_phonemize_are_offered(synthesizer):
    assert synthesizer.voices == {"af_heart": "en-us", "if_sara": "it"}


def test_an_unknown_voice_is_refused_before_any_audio(synthesizer, session):
    with pytest.raises(UnknownVoiceError, match="Available voices: af_heart, if_sara"):
        synthesizer.synthesize("Hello.", voice="jf_alpha")
    assert session.runs == []


def test_each_sentence_is_one_run_between_pad_tokens_with_the_style_of_its_length(synthesizer, session):
    blocks = list(synthesizer.synthesize("Hi. Ciao a te!", voice="if_sara", speed=1.5))

    assert len(blocks) == len(session.runs) == 2
    # fake_espeak: "ihi." (4 tokens) and "iciao a te!" (11 tokens)
    for run, count in zip(session.runs, (4, 11)):
        tokens = run["input_ids"]
        assert tokens.dtype == np.int64 and tokens.shape == (1, count + 2)
        assert tokens[0, 0] == 0 and tokens[0, -1] == 0
        assert run["style"].shape == (1, 256) and run["style"].dtype == np.float32
        assert run["style"][0, 0] == count - 1
        assert run["speed"].dtype == np.float32 and run["speed"][0] == 1.5


def test_the_longest_run_the_model_takes_has_a_style_vector(synthesizer, session):
    # fake_espeak prefixes one character: 510 tokens in all, the last index of the 510 style vectors
    (block,) = synthesizer.synthesize("a" * (MAX_TOKENS - 1))

    assert session.runs[0]["input_ids"].shape == (1, MAX_TOKENS + 2)
    assert session.runs[0]["style"][0, 0] == MAX_TOKENS - 1
    assert len(block) == (MAX_TOKENS + 2) * 240 * 2


def test_a_sentence_longer_than_the_model_takes_is_spoken_in_several_runs(synthesizer, session):
    blocks = list(synthesizer.synthesize("a" * 1200))

    assert len(blocks) == len(session.runs) == 3
    assert all(run["input_ids"].shape[1] <= MAX_TOKENS + 2 for run in session.runs)


def test_audio_is_generated_sentence_by_sentence(synthesizer, session):
    speech = synthesizer.synthesize("One. Two. Three.")
    next(speech)
    assert len(session.runs) == 1


def test_pcm_is_16_bit_little_endian_at_the_model_rate(synthesizer):
    (block,) = synthesizer.synthesize("Hi.")
    samples = np.frombuffer(block, dtype="<i2")
    # 6 tokens with the pads, 240 samples each
    assert samples.size == 6 * 240
    assert set(samples.tolist()) == {16383}


def test_pcm_is_resampled_to_the_requested_rate():
    one_second = np.linspace(-1.0, 1.0, MODEL_SAMPLE_RATE, dtype=np.float32)
    samples = np.frombuffer(to_pcm16(one_second, 44100), dtype="<i2")
    assert samples.size == 44100
    assert samples[0] == -32767 and samples[-1] >= 32766
    assert np.all(np.diff(samples.astype(np.int32)) >= 0)


def test_pcm_clips_samples_out_of_range():
    samples = np.frombuffer(to_pcm16(np.array([-2.0, 2.0], dtype=np.float32), MODEL_SAMPLE_RATE), dtype="<i2")
    assert samples.tolist() == [-32767, 32767]
