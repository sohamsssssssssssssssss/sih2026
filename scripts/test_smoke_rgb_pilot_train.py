"""Synthetic CPU doubles only; no Qwen weights, CUDA calls, or training."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts import smoke_rgb_pilot_train as smoke
from training.remote_sensing import TrainingConfig


def test_frozen_b104d_json_config_accepts_list_without_changing_128_steps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = root / "packages/rgb-vqa-pilot-b10.4d-20261016T175351Z"
    if frozen.exists():
        root = frozen
    cfg = smoke.config(root, tmp_path, tmp_path / "fresh")
    assert cfg.max_steps == 128
    assert cfg.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    smoke.validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="max_steps"):
        smoke.validate_frozen_config(replace(cfg, max_steps=1))
    with pytest.raises(ValueError, match="lora_target_modules"):
        smoke.validate_frozen_config(replace(cfg, lora_target_modules=["q_proj"]))


def test_gradient_hook_rejects_nonfinite_synthetic_backward():
    weight = torch.nn.Parameter(torch.ones(1))
    weight.register_hook(smoke.finite_gradient)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        (weight * torch.tensor(float("inf"))).backward()
    assert weight.grad is None


def test_parent_times_out_and_writes_durable_failure(tmp_path):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 19
        raise subprocess.TimeoutExpired(command, 19)

    output = tmp_path / "fresh"
    assert smoke.run(["--weights", str(tmp_path), "--output", str(output),
                      "--max-seconds", "19"], command_runner=timeout) == 1
    result = json.loads((output / smoke.RESULT).read_text())
    assert result["status"] == "TIMEOUT"
    assert result["optimizer_steps_completed"] == 0
    assert not (output / "checkpoints").exists()
    with pytest.raises(FileExistsError):
        smoke.run(["--weights", str(tmp_path), "--output", str(output),
                   "--max-seconds", "19"], command_runner=timeout)


def test_parent_rejects_incomplete_success_record(tmp_path):
    output = tmp_path / "fresh"
    def forged_success(command, **kwargs):
        child = output / "worker"
        child.mkdir()
        (child / smoke.RESULT).write_text(json.dumps({"status": "PASS", "optimizer_steps_completed": 0}))
        return SimpleNamespace(returncode=0)

    assert smoke.run(["--weights", str(tmp_path), "--output", str(output),
                      "--max-seconds", "19"], command_runner=forged_success) == 1
    assert json.loads((output / smoke.RESULT).read_text())["status"] == "FAIL"


def test_worker_persists_failure_without_reusing_output(tmp_path, monkeypatch):
    def fail(*args):
        raise FloatingPointError("synthetic nonfinite loss")

    monkeypatch.setattr(smoke, "one_update", fail)
    output = tmp_path / "worker"
    assert smoke.run(["--weights", str(tmp_path), "--output", str(output),
                      "--max-seconds", "19", "--worker"]) == 1
    result = json.loads((output / smoke.RESULT).read_text())
    assert result["status"] == "FAIL"
    assert "synthetic nonfinite loss" in result["error"]
    assert result["optimizer_steps_completed"] == 0


@pytest.mark.parametrize("mode", ["pass", "nonfinite_loss", "nonfinite_gradient", "skipped_step", "two_steps", "full_schedule"])
def test_one_update_contract_with_synthetic_trainer(tmp_path, monkeypatch, mode):
    root = tmp_path / "synthetic-package"
    (root / "configs").mkdir(parents=True)
    from eval.rgb_pilot import PILOT_IDENTITY
    spec = {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "a" * 40,
            "weight_sha256": {"synthetic.safetensors": "b" * 64}}
    (root / "configs/model_artifacts.json").write_text(json.dumps({"providers": {"qwen2.5vl-3b": spec}}))
    (root / "pilot-config.json").write_text(json.dumps({
        "base_model_id": spec["model_id"], "base_revision": spec["revision"],
        "base_weight_sha256": spec["weight_sha256"], "manifest_identity": PILOT_IDENTITY
    }))
    cfg = TrainingConfig(
        model_path=tmp_path, dataset_manifest=tmp_path / "synthetic.jsonl",
        image_root=tmp_path, output_dir=tmp_path / "unused", max_samples=1024, max_steps=128
    )
    monkeypatch.setattr(smoke, "check_payload", lambda _: {"source_fingerprint": {"synthetic.py": "a" * 64}})
    monkeypatch.setattr(smoke, "binding", lambda _: "synthetic-binding")
    monkeypatch.setattr(smoke, "config", lambda *a: cfg)
    examples = [SimpleNamespace(sample_id=f"synthetic-{i}") for i in range(1024)]
    monkeypatch.setattr(smoke, "load_manifest", lambda *a: examples)
    monkeypatch.setattr(smoke, "select_examples", lambda *a: examples)
    monkeypatch.setattr(smoke, "QwenVQACollator", lambda *a, **k: object())
    from scripts import stage1_runtime
    monkeypatch.setattr(stage1_runtime, "verify_checkpoint",
                        lambda *a, **k: {"passed": True})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)

    class SyntheticModel:
        def __init__(self):
            self.items = [
                (f"model.layers.0.self_attn.{target}.lora_{kind}.default.weight",
                 torch.nn.Parameter(torch.zeros((16, 32) if kind == "A" else (32, 16))))
                for target in ("q_proj", "k_proj", "v_proj", "o_proj")
                for kind in ("A", "B")
            ]
        def named_parameters(self):
            return self.items
        def parameters(self):
            return [parameter for _, parameter in self.items]

    model = SyntheticModel()
    monkeypatch.setattr(smoke, "load_training_components", lambda _: (model, object()))
    seen = {}
    class FakeTrainer:
        def __init__(self, **kwargs):
            seen["args"] = kwargs["args"]
            seen["dataset"] = kwargs["train_dataset"]
            self.model = kwargs["model"]
            self.callback = kwargs["callbacks"][0]
            self.accelerator = SimpleNamespace(optimizer_step_was_skipped=mode == "skipped_step")
            self.state = SimpleNamespace(global_step=0)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = torch.tensor(float("nan") if mode == "nonfinite_loss" else 1.25)
            return (loss, {}) if return_outputs else loss
        def train(self):
            for _ in range(8):
                self.compute_loss(self.model, {})
            for parameter in self.model.parameters():
                parameter.grad = torch.full_like(parameter, float("inf") if mode == "nonfinite_gradient" else 1.0)
            self.callback.on_pre_optimizer_step(None, self.state, None, model=self.model)
            self.callback.on_optimizer_step(None, self.state, None)
            if mode == "two_steps":
                self.callback.on_optimizer_step(None, self.state, None)
            self.state.global_step = 1
            return SimpleNamespace(metrics={"synthetic": True})

    fake = ModuleType("transformers")
    fake.Trainer = FakeTrainer
    fake.TrainerCallback = object
    fake.TrainingArguments = lambda **kwargs: SimpleNamespace(
        **{**kwargs, "max_steps": 128 if mode == "full_schedule" else kwargs["max_steps"]}
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    if mode == "pass":
        result = smoke.one_update(root, tmp_path, tmp_path / "fresh")
        assert smoke.valid_success(result)
        assert len(seen["dataset"]) == 1024
        assert (seen["args"].max_steps, seen["args"].gradient_accumulation_steps,
                seen["args"].per_device_train_batch_size, seen["args"].save_strategy) == (1, 8, 1, "no")
        assert result["trainable_parameter_count"] == 8 * 16 * 32
        assert result["first_training_loss"] == result["final_training_loss"] == 1.25
    else:
        with pytest.raises((FloatingPointError, RuntimeError, ValueError)):
            smoke.one_update(root, tmp_path, tmp_path / "fresh")
        if mode == "full_schedule":
            assert "dataset" not in seen
