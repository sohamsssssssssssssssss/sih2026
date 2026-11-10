"""Small integrity check for the prepared, unuploaded T4 evidence package (loop 7)."""

import json
import tarfile
from pathlib import Path

import yaml


def test_t4_smoke_package():
    root = Path(__file__).resolve().parents[1] / "packages/stage1-t4-smoke"
    config = yaml.safe_load((root / "config.yaml").read_text())
    fallback = yaml.safe_load((root / "config-vision-fp32.yaml").read_text())
    captions = json.loads((root / "captions.json").read_text())
    contract = json.loads((root / "norm_contract.json").read_text())
    with tarfile.open(root / "patches.tar.gz") as archive:
        files = [member.name for member in archive if member.isfile()]
    assert len(captions["captions"]) == 16
    assert len(files) == 16 * len(contract["bands"])
    assert not any(name.endswith("_B10.tif") for name in files)
    assert config["steps"] == 50 and config["precision"] == "fp16"
    assert config["attention"] == "sdpa" and config["gradient_checkpointing"]
    assert config["gradient_accumulation_steps"] == 8
    assert config["checkpoint_every_steps"] == 500 and config["resume"]
    assert config["check_vision_finite"]
    assert fallback["vision_fp32"] and fallback["precision"] == "fp16"


def test_t4_package_entry_and_stages():
    root = Path(__file__).resolve().parents[1] / "packages/stage1-t4-smoke"
    revision = json.loads((root / "revision.json").read_text())
    assert revision["revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"
    config = yaml.safe_load((root / "config.yaml").read_text())
    fallback = yaml.safe_load((root / "config-vision-fp32.yaml").read_text())
    assert config["revision"] == fallback["revision"] == revision["revision"]
    assert config["seed"] == fallback["seed"] == 0
    assert config["device"] == fallback["device"] == "cuda"
    assert config["checkpoint_every_steps"] == fallback["checkpoint_every_steps"]
    run_py = (root / "run.py").read_text()
    assert "run_stage1_t4" in run_py and "all" in run_py and "smoke" in run_py
    runner = (root / "src/scripts/run_stage1_t4.py").read_text()
    for stage in ("stage_verify", "stage_smoke", "stage_resume", "stage_overfit", "stage_extrapolate"):
        assert f"def {stage}(" in runner
    assert "config-vision-fp32.yaml" in runner
    readme = (root / "README.md").read_text()
    assert "hf download" in readme and "private Kaggle dataset" in readme
    assert "66285546d2b821cf421d4f5eb2576359d3770cd3" in readme


import pytest


@pytest.mark.parametrize("last_n", [None, 4])
@pytest.mark.parametrize("vision_fp32", [False, True])
@pytest.mark.parametrize("layout", ["direct", "nested", "peft-direct", "peft-nested"])
@pytest.mark.parametrize("initialization", [None, "rgb_preserving_zero9_v1"])
def test_fp32_trainable_masters_and_complete_checkpoint(monkeypatch, last_n, vision_fp32, layout, initialization):
    """Synthetic CPU model: option forwarding, never pretrained evidence."""
    import torch
    from scripts import run_stage1_smoke as smoke
    from models.qwen_vl.stage1 import resolve_vision_module

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.patch_embed = torch.nn.Module()
            self.visual.patch_embed.proj = torch.nn.Conv3d(3, 4, (2, 2, 2), dtype=torch.float16)
            self.visual.other = torch.nn.Linear(4, 4, dtype=torch.float16)
            self.visual.blocks = torch.nn.ModuleList([
                torch.nn.Linear(4, 4, dtype=torch.float16) for _ in range(8)])
            self.frozen_lm = torch.nn.Linear(4, 4, dtype=torch.float16)
            self.lora_A = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.float16))
            self.config = type("Config", (), {})()
        def get_base_model(self):
            return self
        def enable_input_require_grads(self):
            pass

    model = Model()
    tower = model.visual
    original_weight = tower.patch_embed.proj.weight.detach().float().clone()
    if "nested" in layout:
        del model.visual
        model.model = torch.nn.Module()
        model.model.visual = tower
    assert resolve_vision_module(model) is tower
    monkeypatch.setattr(smoke.Qwen2_5_VLForConditionalGeneration, "from_pretrained", lambda *a, **k: model)
    class PeftWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_model = base
        def get_base_model(self):
            return self.base_model
        def enable_input_require_grads(self):
            self.base_model.enable_input_require_grads()

    def peft(model, _config):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.lora_A.requires_grad_(True)
        return PeftWrapper(model) if layout.startswith("peft-") else model
    monkeypatch.setattr(smoke, "get_peft_model", peft)
    config = {"seed": 0, "attention": "sdpa", "gradient_checkpointing": False, "vision_fp32": vision_fp32,
              "lora": {"r": 32, "alpha": 64, "target_modules": ["q_proj"]},
              "learning_rates": {"vision": 1e-5, "patch_embed": 1e-4, "lm_lora": 1e-4}}
    if last_n is not None:
        config["vision_trainable_last_n_blocks"] = last_n
    if initialization is not None:
        config["patch_embed_initialization"] = initialization
    built, groups = smoke.build_model(config, "unused", torch.float16, "cpu")
    if initialization is not None:
        for channel, band in enumerate((3, 2, 1)):
            assert torch.equal(tower.patch_embed.proj.weight[:, band], original_weight[:, channel])
        assert torch.count_nonzero(tower.patch_embed.proj.weight[:, [0, 4, 5, 6, 7, 8, 9, 10, 11]]) == 0
    assert resolve_vision_module(built) is tower
    assert model.frozen_lm.weight.dtype == torch.float16
    assert not model.frozen_lm.weight.requires_grad
    assert {group["name"] for group in groups} == {"vision", "patch_embed", "lm_lora"}
    assert all(p.dtype == torch.float32 for g in groups for p in g["params"])
    saved = smoke.trainable_state(built)
    prefix = ("base_model." if layout.startswith("peft-") else "")
    vision_prefix = prefix + ("model.visual." if "nested" in layout else "visual.")
    assert vision_prefix + "patch_embed.proj.weight" in saved
    assert (vision_prefix + "other.weight" in saved) is (last_n is None)
    by_name = {group["name"]: {id(p) for p in group["params"]} for group in groups}
    assert by_name["patch_embed"] == {id(p) for p in tower.patch_embed.proj.parameters()}
    assert by_name["vision"] == {id(p) for name, p in tower.named_parameters()
                                   if p.requires_grad and not name.startswith("patch_embed.proj.")}
    assert by_name["lm_lora"] == {id(model.lora_A)}
    for index, block in enumerate(tower.blocks):
        assert block.weight.requires_grad is (last_n is None or index >= 4)
        assert block.weight.dtype == (torch.float32 if block.weight.requires_grad or vision_fp32 else torch.float16)
    assert prefix + "lora_A" in saved and prefix + "frozen_lm.weight" not in saved
    if vision_fp32:
        assert all(p.dtype == torch.float32 for p in tower.parameters())
    saved = {name: value.clone() for name, value in saved.items()}
    with torch.no_grad():
        for p in built.parameters():
            if p.requires_grad:
                p.zero_()
    smoke.load_trainable_state(built, saved)
    for name, value in smoke.trainable_state(built).items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert smoke.make_scaler("cuda", torch.float16).is_enabled()
    assert not smoke.make_scaler("cpu", torch.float32).is_enabled()
