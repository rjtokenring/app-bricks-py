# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for ``common/gguf_naming.py``."""

import os

import pytest

from common.gguf_naming import GGUF_SUFFIX, catalog_gguf_declarations, declared_gguf_files, gguf_model_name
from common.models_list import load_models_list

REAL_MODELS_LIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "models-list.yaml"))

GEMMA_URL = "https://huggingface.co/google/gemma-gguf/blob/abc123/gemma-Q4_0.gguf"

MODELS = [
    {
        "llamacpp:gemma-Q4_0": {
            "name": "Gemma",
            "deployment": {
                "handler": "hf-handler",
                "platforms": [
                    {"ventunoq": {"variables": {"model_url": GEMMA_URL, "models_repository": "llamacpp", "model_directory": "google/gemma-gguf"}}},
                    # A second board with the same variables must not duplicate the declaration.
                    {"unoq": {"variables": {"model_url": GEMMA_URL, "models_repository": "llamacpp", "model_directory": "google/gemma-gguf"}}},
                ],
            },
        }
    },
    {
        "llamacpp:qwen-by-key": {
            "name": "Qwen via compact key",
            "deployment": {
                "handler": "hf-handler",
                "platforms": [
                    {
                        "unoq": {
                            "variables": {
                                "model_url": "llamacpp:unsloth/Qwen-GGUF:Q4_0",
                                "models_repository": "llamacpp",
                                "model_directory": "unsloth/Qwen-GGUF",
                            }
                        }
                    }
                ],
            },
        }
    },
    # Another handler's entry: not under the llamacpp tree, never a declaration.
    {
        "ei:some-model": {
            "name": "EI",
            "deployment": {
                "handler": "ei-handler",
                "platforms": [{"unoq": {"variables": {"models_repository": "models/edge-impulse", "model_directory": "some-model"}}}],
            },
        }
    },
    # Pre-loaded entry: no platforms, nothing to declare.
    {"builtin:asr": {"name": "ASR", "deployment": {"handler": "asr-handler", "pre-loaded": True}}},
]


def test_declared_gguf_files_extracts_llamacpp_locations():
    assert declared_gguf_files(MODELS) == [
        ("google/gemma-gguf", "gemma-Q4_0.gguf", "llamacpp:gemma-Q4_0"),
        # A compact key naming only a quantization pins no file.
        ("unsloth/Qwen-GGUF", None, "llamacpp:qwen-by-key"),
    ]


def test_gguf_model_name_stems_curated_and_recordless_files():
    # A curated download keeps the stem its fixed entry id uses.
    assert gguf_model_name("google/gemma-gguf/gemma-Q4_0.gguf", {"model_origin": "built_in"}) == "gemma-Q4_0"
    # No record at all: an out-of-the-box model — the fallback — keeps its stem too.
    assert gguf_model_name("google/gemma-gguf/gemma-Q4_0.gguf", None) == "gemma-Q4_0"
    # An unusable record (no origin, or not a mapping) degrades the same way.
    assert gguf_model_name("google/gemma-gguf/gemma-Q4_0.gguf", {}) == "gemma-Q4_0"
    assert gguf_model_name("google/gemma-gguf/gemma-Q4_0.gguf", "junk") == "gemma-Q4_0"


def test_gguf_model_name_user_files_are_path_qualified():
    record = {"model_origin": "user"}
    # Same file name as a curated model, different repository: not that model.
    assert gguf_model_name("bartowski/gemma-clone/gemma-Q4_0.gguf", record) == "bartowski/gemma-clone/gemma-Q4_0"
    assert gguf_model_name("TheBloke/Mistral-GGUF/mistral.Q4_0.gguf", record) == "TheBloke/Mistral-GGUF/mistral.Q4_0"
    # Nested per-quantization folders keep the whole path in the name.
    assert gguf_model_name("unsloth/Qwen-GGUF/Q4_0/Qwen-Q4_0.gguf", record) == "unsloth/Qwen-GGUF/Q4_0/Qwen-Q4_0"


def test_catalog_gguf_declarations_degrades_to_empty(tmp_path):
    assert catalog_gguf_declarations(str(tmp_path / "missing.yaml")) == ()
    broken = tmp_path / "broken.yaml"
    broken.write_text("models: [a: b: c")
    assert catalog_gguf_declarations(str(broken)) == ()


def test_catalog_gguf_declarations_reads_a_models_list(tmp_path):
    yaml_path = tmp_path / "models-list.yaml"
    yaml_path.write_text(
        "models:\n"
        ' - "llamacpp:gemma-Q4_0":\n'
        "    name: Gemma\n"
        "    deployment:\n"
        "      handler: hf-handler\n"
        "      platforms:\n"
        "        - unoq:\n"
        "            variables:\n"
        f'              model_url: "{GEMMA_URL}"\n'
        '              models_repository: "llamacpp"\n'
        '              model_directory: "google/gemma-gguf"\n'
    )
    assert catalog_gguf_declarations(str(yaml_path)) == [("google/gemma-gguf", "gemma-Q4_0.gguf", "llamacpp:gemma-Q4_0")]


@pytest.mark.skipif(not os.path.isfile(REAL_MODELS_LIST), reason="models/models-list.yaml not available")
def test_every_curated_llamacpp_entry_is_named_after_its_gguf_file():
    """Every curated llamacpp entry id must be ``llamacpp:<gguf stem>``.

    A curated download is served by llama-server under the models.ini section named
    after its file stem (``gguf_model_name``), while the LLM/VLM brick asks for the
    part of the entry id after "llamacpp:". An entry whose id no longer matches the
    file its model_url pins — an id left behind when the URL was repointed at another
    repository — installs fine and then answers to nothing at runtime.
    """
    declarations = declared_gguf_files(load_models_list(REAL_MODELS_LIST))
    checked = 0
    for _directory, filename, model_id in declarations:
        if filename is None:  # a compact key pinning only a quantization names no file
            continue
        stem = filename[: -len(GGUF_SUFFIX)]
        assert model_id == f"llamacpp:{stem}", f"{model_id} pins {filename}: id must be llamacpp:{stem}"
        checked += 1
    assert checked > 0
