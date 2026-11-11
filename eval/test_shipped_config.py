"""Runs the stage code against the SHIPPED package files (data/cdvqa/package/), on CPU.

A key missing from pilot-config.json or model_artifacts.json fails here, not on Kaggle.
Fake processor and model only; no weights, no GPU.
"""
import json
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image

from eval.test_cdvqa_pilot import FakeProcessor, fake_process_vision_info
from scripts import cdvqa_pilot as pilot
from scripts.stage1_runtime import verify_checkpoint
from training.remote_sensing import TrainingExample

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "data/cdvqa/package"
PINNED_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
TRAINING_KEYS = {"split", "seed", "max_samples", "image_size", "epochs", "max_steps", "batch_size",
                 "gradient_accumulation_steps", "learning_rate", "precision", "quantization", "lora_rank",
                 "lora_alpha", "lora_dropout", "lora_target_modules", "max_sequence_length"}


def shipped_root(tmp_path):
    """A throwaway package root holding only the shipped config files and the real manifests."""
    root = tmp_path / "pkg"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(SHIPPED / "pilot-config.json", root / "pilot-config.json")
    shutil.copyfile(SHIPPED / "configs/model_artifacts.json", root / "configs/model_artifacts.json")
    return root


def test_shipped_plan_has_every_key_the_stages_read():
    plan = json.loads((SHIPPED / "pilot-config.json").read_text())
    assert TRAINING_KEYS <= set(plan["training"])
    assert plan["training"]["max_sequence_length"] == 32768
    assert plan["base_revision"] == PINNED_REVISION
    assert set(plan["dependency_pins"]) >= {"torch", "transformers", "peft", "qwen-vl-utils"}
    assert plan["budget"]["ledger_total_seconds"] == {"train": 25200, "evaluate": 7200}
    assert plan["training"]["max_steps"] == 1000 and plan["training"]["max_samples"] == 8000
    assert plan["training"]["gradient_accumulation_steps"] == 8 and plan["training"]["batch_size"] == 1


def test_shipped_model_artifacts_have_every_key_the_stages_read():
    spec = json.loads((SHIPPED / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    assert spec["revision"] == PINNED_REVISION
    assert spec["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert set(spec["weight_sha256"]) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_shipped_provenance_has_every_key_the_stages_read():
    provenance = json.loads((REPO / "data/manifests/cdvqa/provenance.json").read_text())
    for key in ("manifest_sha256", "answer_vocabulary", "answer_choices", "majority_answer"):
        assert key in provenance, key


def test_token_check_runs_on_shipped_plan(tmp_path):
    """token_preflight's core, with the shipped training block: fails with KeyError if a key is missing."""
    plan = json.loads((SHIPPED / "pilot-config.json").read_text())["training"]
    examples = []
    for i in range(2):
        before, after = tmp_path / f"im1-{i}.png", tmp_path / f"im2-{i}.png"
        Image.new("RGB", (512, 512), (i, 0, 0)).save(before)
        Image.new("RGB", (512, 512), (0, i, 0)).save(after)
        examples.append(TrainingExample(before, "Did it change?", "yes", "CDVQA", str(i), "train", f"s{i}",
                                        after_image_path=after, choices=("yes", "no")))
    report = pilot.token_check(plan, examples, FakeProcessor(), process_vision_info=fake_process_vision_info)
    assert report["max_sequence_length"] == 32768
    assert report["examples_checked"] == 2


def test_train_config_loads_from_shipped_plan(tmp_path):
    root = shipped_root(tmp_path)
    config = pilot.train_config(root, tmp_path / "weights", tmp_path / "out", tmp_path / "images")
    config.validate()
    assert (config.max_samples, config.max_steps, config.max_sequence_length) == (8000, 1000, 32768)
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_verify_base_reads_shipped_artifacts_and_rejects_missing_weights(tmp_path):
    root = shipped_root(tmp_path)
    empty_weights = tmp_path / "weights"
    empty_weights.mkdir()
    with pytest.raises(ValueError, match="Pinned base revision"):
        pilot.verify_base(root, empty_weights)


def test_revision_proof_from_restored_hf_metadata(tmp_path):
    """Kaggle workaround: no _commit_hash in config.json, so the pinned commit is read from restored metadata.
    Weights must still match, so this test checks only the revision side."""
    weights = tmp_path / "qwen-pinned"
    (weights / ".cache/huggingface/download").mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    (weights / ".cache/huggingface/download/config.json.metadata").write_text(f"{PINNED_REVISION}\netag\n0\n")
    spec = json.loads((SHIPPED / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    proof = verify_checkpoint(weights, spec["revision"], spec["weight_sha256"])
    assert proof["revision_source"] == "hf_download_metadata"
    assert proof["revision_matches"] is True
    assert proof["passed"] is False  # weight files are absent here, so the full proof must still fail
