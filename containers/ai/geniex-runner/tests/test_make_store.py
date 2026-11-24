# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for make_store.py: AI Hub bundles exposed as a GenieX model store."""

import json
from pathlib import Path

import pytest

from make_store import make_store

# metadata.json shapes of the two AI Hub generations seen on the boards
QAIRT_LLM = {
    "model_id": "qwen3_4b_instruct_2507",
    "model_name": "Qwen3-4B-Instruct-2507",
    "runtime": "RUNTIME_GENIEX_QAIRT",
    "precision": "PRECISION_W4A16",
    "model_files": [{"filename": "part1_of_2.bin"}, {"filename": "part2_of_2.bin"}],
    "genie": {"supports_streaming": True},
}
QAIRT_VLM = {
    "model_id": "qwen3_vl_4b_instruct",
    "model_name": "Qwen3-VL-4B-Instruct",
    "runtime": "RUNTIME_GENIEX_QAIRT",
    "precision": "PRECISION_W4A16",
    "model_files": [{"filename": "vision_encoder.bin"}, {"filename": "part1_of_1.bin"}],
    "genie": {"supports_vision": True},
}
OLD_GENIE = {
    "model_id": "qwen3_4b_instruct_2507",
    "model_name": "Qwen3-4B-Instruct-2507",
    "runtime": "genie",
    "precision": "w4a16",
    "model_files": {"ar128_cl512_1_of_1": {"filename": "qwen3_w4a16_part_1_of_1.bin"}},
    "genie": {},
}


def _bundle(models: Path, name: str, metadata: dict, files: list[str]) -> Path:
    bundle = models / name
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for file in files:
        (bundle / file).write_bytes(b"x" * 8)
    return bundle


def _manifest(store: Path, model_id: str) -> dict:
    return json.loads((store / "models" / "qualcomm" / model_id / "geniex.json").read_text(encoding="utf-8"))


def test_llm_bundle_becomes_a_store_entry_of_links(tmp_path: Path):
    models, store = tmp_path / "models", tmp_path / "store"
    bundle = _bundle(
        models,
        "qwen3_4b_instruct_2507-geniex_qairt-w4a16-qualcomm_qcs8275",
        QAIRT_LLM,
        ["part1_of_2.bin", "part2_of_2.bin", "tokenizer.json", "genie_config.json"],
    )

    assert make_store(models, store) == ["qwen3_4b_instruct_2507"]

    manifest = _manifest(store, "qwen3_4b_instruct_2507")
    assert manifest["Name"] == "qualcomm/qwen3_4b_instruct_2507"
    assert manifest["ModelType"] == "llm"
    assert manifest["PluginId"] == "qairt"
    assert manifest["ModelFile"] == {"W4A16": {"Name": "part1_of_2.bin", "Downloaded": True, "Size": 8}}
    assert sorted(f["Name"] for f in manifest["ExtraFiles"]) == ["genie_config.json", "metadata.json", "part2_of_2.bin", "tokenizer.json"]
    entry = store / "models" / "qualcomm" / "qwen3_4b_instruct_2507"
    for name in ["part1_of_2.bin", "part2_of_2.bin", "tokenizer.json", "genie_config.json", "metadata.json"]:
        assert (entry / name).is_symlink()
        assert (entry / name).resolve() == (bundle / name).resolve()


def test_vision_bundle_is_a_vlm(tmp_path: Path):
    models, store = tmp_path / "models", tmp_path / "store"
    _bundle(models, "qwen3_vl", QAIRT_VLM, ["vision_encoder.bin", "part1_of_1.bin"])

    make_store(models, store)

    assert _manifest(store, "qwen3_vl_4b_instruct")["ModelType"] == "vlm"


def test_genie_bundles_are_served_only_on_request(tmp_path: Path):
    models, store = tmp_path / "models", tmp_path / "store"
    _bundle(models, "qwen3-genie", OLD_GENIE, ["qwen3_w4a16_part_1_of_1.bin"])

    assert make_store(models, store) == []
    assert make_store(models, store, ("GENIE",)) == ["qwen3_4b_instruct_2507"]
    assert _manifest(store, "qwen3_4b_instruct_2507")["ModelFile"]["W4A16"]["Name"] == "qwen3_w4a16_part_1_of_1.bin"


@pytest.mark.parametrize("reason", ["downloading", "no_bin"])
def test_incomplete_bundles_are_skipped(tmp_path: Path, reason: str):
    models, store = tmp_path / "models", tmp_path / "store"
    files = ["tokenizer.json"] if reason == "no_bin" else ["part1_of_2.bin", ".download"]
    _bundle(models, "qwen3", QAIRT_LLM, files)

    assert make_store(models, store) == []


def test_dot_files_of_the_downloader_are_not_linked(tmp_path: Path):
    models, store = tmp_path / "models", tmp_path / "store"
    _bundle(models, "qwen3", QAIRT_LLM, ["part1_of_2.bin", ".arduino_metadata.yaml"])

    make_store(models, store)

    assert not (store / "models" / "qualcomm" / "qwen3_4b_instruct_2507" / ".arduino_metadata.yaml").exists()


def test_rebuild_drops_removed_models_and_leaves_the_models_dir_untouched(tmp_path: Path):
    models, store = tmp_path / "models", tmp_path / "store"
    bundle = _bundle(models, "qwen3", QAIRT_LLM, ["part1_of_2.bin"])
    before = sorted(p.name for p in bundle.iterdir())
    make_store(models, store)

    (bundle / "metadata.json").unlink()
    assert make_store(models, store) == []

    assert not (store / "models" / "qualcomm" / "qwen3_4b_instruct_2507").exists()
    assert sorted(p.name for p in bundle.iterdir()) == [n for n in before if n != "metadata.json"]
