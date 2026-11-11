# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Text front end: sentence splitting, punctuation kept around the phonemes, tokens."""

import subprocess

import pytest

import text
from text import MAX_TOKENS, load_vocab, phonemize, sentences, tokenize, voice_language


def upper(piece, language):
    return piece.upper()


def test_sentences_split_after_their_end_and_at_line_breaks():
    assert sentences("One. Two!  Three?\nFour\n\n five") == ["One.", "Two!", "Three?", "Four", "five"]
    assert sentences("  \n ") == []


def test_a_decimal_point_does_not_end_a_sentence():
    assert sentences("Pi is 3.14 or so.") == ["Pi is 3.14 or so."]


def test_phonemize_keeps_the_punctuation_where_it_was():
    assert phonemize("Hello, world! How are you?", "en-us", upper) == "HELLO, WORLD! HOW ARE YOU?"
    assert phonemize('He said: "go" (now)...', "en-us", upper) == 'HE SAID: "GO" (NOW)...'


def test_phonemize_sends_no_punctuation_to_espeak():
    pieces = []

    def record(piece, language):
        pieces.append((piece, language))
        return piece

    phonemize("Ciao, come stai?", "it", record)
    assert pieces == [("Ciao", "it"), ("come stai", "it")]


def test_phonemize_drops_the_language_switch_flags():
    assert phonemize("Una pizza", "it", lambda piece, language: "una (en)pizza(it)") == "una pizza"


def test_espeak_ipa_ends_the_text_with_a_newline_and_joins_the_clauses(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=" həlˈoʊ\n wˈɜːld\n", stderr="")

    monkeypatch.setattr(text.subprocess, "run", fake_run)

    assert text.espeak_ipa("hello world", "en-us") == "həlˈoʊ wˈɜːld"
    command, kwargs = calls[0]
    assert command == ["espeak-ng", "-q", "--ipa", "-v", "en-us", "--stdin"]
    assert kwargs["input"] == "hello world\n"
    assert kwargs["check"] is True


def test_tokenize_drops_the_characters_the_model_does_not_know():
    vocab = load_vocab()
    assert tokenize("a-b_c", vocab) == [[vocab["a"], vocab["b"], vocab["c"]]]
    assert tokenize("", vocab) == []


def test_tokenize_cuts_a_long_sequence_at_a_space():
    vocab = load_vocab()
    runs = tokenize("a" * 400 + " " + "b" * 400, vocab)
    assert [len(run) for run in runs] == [400, 400]
    assert all(len(run) <= MAX_TOKENS for run in tokenize("a" * 1200, vocab))
    assert sum(len(run) for run in tokenize("a" * 1200, vocab)) == 1200


@pytest.mark.parametrize(("voice", "language"), [("af_heart", "en-us"), ("bm_george", "en-gb"), ("if_sara", "it"), ("pf_dora", "pt-br")])
def test_a_voice_speaks_the_language_of_its_first_letter(voice, language):
    assert voice_language(voice) == language


@pytest.mark.parametrize("voice", ["jf_alpha", "zf_xiaobei", ""])
def test_a_voice_espeak_cannot_phonemize_has_no_language(voice):
    assert voice_language(voice) is None
