"""CPU stochastic model doubles; no Qwen loading or GPU training."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import run_stage1_smoke as smoke


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    config = yaml.safe_load(Path("packages/stage1-t4-smoke/config-last4-vision-fp32.yaml").read_text())
    config.update(device="cpu", precision="fp32", steps=6, optimizer="adamw",
                  gradient_accumulation_steps=2, checkpoint_every_steps=500)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    (tmp_path / "captions.json").write_text(json.dumps({"split_name": "synthetic/train", "captions": [{"output": "test"}]}))
    (tmp_path / "norm_contract.json").write_text('{"synthetic": true}')

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(6)])
            self.visual.patch_embed = torch.nn.Linear(1, 1)
            self.lora_A = torch.nn.Parameter(torch.ones(1))
            self.frozen_lm = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            for block in self.visual.blocks[:2]:
                block.requires_grad_(False)

        def get_base_model(self):
            return self

        def forward(self, input_ids):
            # All three RNG streams influence loss, exposing incomplete reloads.
            target = torch.rand(()) + random.random() + np.random.random()
            value = sum(p.sum() for p in self.parameters() if p.requires_grad)
            return type("Output", (), {"loss": (value - target).square()})()

    def build(*args):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        model = Tiny()
        return model, [
            {"name": "vision", "params": list(model.visual.blocks[2:].parameters()), "lr": 1e-3},
            {"name": "patch_embed", "params": list(model.visual.patch_embed.parameters()), "lr": 2e-3},
            {"name": "lm_lora", "params": [model.lora_A], "lr": 3e-3},
        ]

    monkeypatch.setattr(smoke, "build_model", build)
    monkeypatch.setattr(smoke, "verify_checkpoint", lambda *a: {
        "passed": True, "weights_match": True, "revision_source": "synthetic", "revision_matches": True})
    monkeypatch.setattr(smoke.AutoTokenizer, "from_pretrained", lambda *a, **k:
        type("Tokenizer", (), {"decode": lambda *a, **k: "synthetic"})())
    monkeypatch.setattr(smoke, "load_samples", lambda *a:
        ([{"input_ids": torch.tensor([[1]])}], []))
    # Real CPU scaler with nonempty growth state, rather than disabled {}.
    monkeypatch.setattr(smoke, "make_scaler", lambda *a: torch.amp.GradScaler("cpu", init_scale=16))
    return path, tmp_path


def test_cpu_resume_restores_all_state_without_gpu_checkpoint_copy(tiny_run, monkeypatch):
    path, root = tiny_run
    prefix = smoke.run(path, root, root, root / "prefix.json", stop_after_step=3)
    interrupted = torch.load(root / "prefix.pt", map_location="cpu", weights_only=False)
    assert interrupted["scaler"] and interrupted["optimizer"]["state"]
    assert set(interrupted["model"]) == {"lora_A", "visual.patch_embed.weight", "visual.patch_embed.bias"} | {
        f"visual.blocks.{i}.{name}" for i in range(2, 6) for name in ("weight", "bias")}
    assert interrupted["data_position"] == {"next_micro_sample": 6}
    actual_load = torch.load
    locations = []
    def load(*a, **kw):
        locations.append(kw.get("map_location"))
        return actual_load(*a, **kw)
    monkeypatch.setattr(torch, "load", load)
    resumed = smoke.run(path, root, root, root / "resumed.json", resume_from=root / "prefix.pt")
    assert locations == ["cpu"]
    uninterrupted = smoke.run(path, root, root, root / "whole.json")
    assert [v["loss"] for v in resumed["losses"]] == [v["loss"] for v in uninterrupted["losses"][3:]]
    assert [v["lr"] for v in resumed["losses"]] == [v["lr"] for v in uninterrupted["losses"][3:]]
    assert resumed["optimizer_state_digest_after_load"] == prefix["optimizer_state_digest"]
    final_resume = actual_load(root / "resumed.pt", map_location="cpu", weights_only=False)
    final_whole = actual_load(root / "whole.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler", "torch_rng_state", "numpy_rng_state",
                "python_rng_state", "cuda_rng_state", "data_position"):
        assert smoke.canonical_digest(final_resume[key]) == smoke.canonical_digest(final_whole[key]), key


@pytest.mark.parametrize("change", ["vision_fp32", "vision_trainable_last_n_blocks", "learning_rates",
                                     "seed", "steps", "captions", "contract", "shuffle"])
def test_resume_rejects_changed_identity(tiny_run, change):
    path, root = tiny_run
    smoke.run(path, root, root, root / "prefix.json", stop_after_step=3)
    config = yaml.safe_load(path.read_text())
    kwargs = {}
    if change in ("captions", "contract"):
        file = root / config[change]
        file.write_text(file.read_text() + "\n")
    elif change == "shuffle":
        kwargs["shuffle_images"] = True
    else:
        config[change] = {"vision_fp32": False, "vision_trainable_last_n_blocks": 3,
                          "learning_rates": {"vision": 5e-4}, "seed": 1, "steps": 7}[change]
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="config identity"):
        smoke.run(path, root, root, root / "bad.json", resume_from=root / "prefix.pt", **kwargs)


def test_relocated_identical_config_preserves_identity(tiny_run):
    path, root = tiny_run
    other = root / "relocated"
    other.mkdir()
    for name in (path.name, "captions.json", "norm_contract.json"):
        (other / name).write_bytes((root / name).read_bytes())
    assert smoke.checkpoint_identity(path, total_steps=6) == smoke.checkpoint_identity(other / path.name, total_steps=6)


def test_eval_cpu_load_and_shuffled_arm_keep_training_identity(tiny_run, monkeypatch):
    path, root = tiny_run
    smoke.run(path, root, root, root / "trained.json", max_steps=4)
    actual_load = torch.load
    locations = []
    def load(*a, **kw):
        locations.append(kw.get("map_location"))
        return actual_load(*a, **kw)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(smoke, "shuffle_pixel_streams", lambda *a: [0])
    result = smoke.evaluate_losses(path, root, root, root / "eval.json",
                                   from_checkpoint=root / "trained.pt", shuffle_images=True)
    assert result["passed"] and locations == ["cpu"]
    config = yaml.safe_load(path.read_text())
    config["vision_fp32"] = False
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="config identity"):
        smoke.evaluate_losses(path, root, root, root / "wrong-eval.json", from_checkpoint=root / "trained.pt")


def test_missing_identity_and_bad_data_position_are_rejected(tiny_run):
    path, root = tiny_run
    smoke.run(path, root, root, root / "prefix.json", stop_after_step=3)
    state = torch.load(root / "prefix.pt", map_location="cpu", weights_only=False)
    state.pop("config_identity")
    with pytest.raises(ValueError, match="config identity"):
        smoke.validate_checkpoint_identity(state, path, total_steps=6)
    state["config_identity"] = smoke.checkpoint_identity(path, total_steps=6)
    state["data_position"]["next_micro_sample"] += 1
    with pytest.raises(ValueError, match="data position"):
        smoke.restore_training_state(None, None, None, None, state, path, total_steps=6)
