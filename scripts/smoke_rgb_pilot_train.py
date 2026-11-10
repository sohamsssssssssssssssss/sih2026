"""Bounded, one-update RGB pilot smoke; run only from a verified derivative package."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rgb_pilot_package import binding, check_payload, config, trainable_guard
from training.remote_sensing import (
    QwenVQACollator,
    load_manifest,
    load_training_components,
    select_examples,
)

RESULT = "one-step-smoke.json"


def durable_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def valid_success(result: dict) -> bool:
    return (
        result.get("status") == "PASS"
        and result.get("optimizer_steps_completed") == 1
        and result.get("trainer_global_step") == 1
        and result.get("gradient_finite") is True
        and all(isinstance(result.get(key), str) and result[key]
                for key in ("run_identity", "manifest_identity", "model_revision",
                            "model_id", "package_source_fingerprint", "loss_definition"))
        and isinstance(result.get("trainable_parameter_count"), int)
        and result["trainable_parameter_count"] > 0
        and isinstance(result.get("peak_allocated_gpu_bytes"), int)
        and result["peak_allocated_gpu_bytes"] > 0
        and all(isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
                for key in ("first_training_loss", "final_training_loss"))
    )


def finite_gradient(gradient):
    import torch
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite LoRA gradient during backward")
    return gradient


def validate_frozen_config(cfg) -> None:
    cfg.validate()
    actual = {
        "split": cfg.split,
        "max_samples": cfg.max_samples,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "quantization": cfg.quantization,
        "precision": cfg.precision,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": tuple(cfg.lora_target_modules),
    }
    expected = {
        "split": "train", "max_samples": 1024, "max_steps": 128,
        "batch_size": 1, "gradient_accumulation_steps": 8,
        "quantization": "4bit", "precision": "fp16",
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError(f"Frozen training configuration differs: {', '.join(mismatches)}")


def one_update(root: Path, weights: Path, output: Path, progress: dict | None = None) -> dict:
    progress = progress if progress is not None else {}
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    manifest = check_payload(root)
    plan = json.loads((root / "pilot-config.json").read_text(encoding="utf-8"))
    from eval.rgb_pilot import PILOT_IDENTITY
    from scripts.stage1_runtime import verify_checkpoint

    spec = json.loads((root / "configs/model_artifacts.json").read_text(encoding="utf-8"))["providers"]["qwen2.5vl-3b"]
    if (plan["base_model_id"], plan["base_revision"], plan["base_weight_sha256"], plan["manifest_identity"]) != (
        spec["model_id"], spec["revision"], spec["weight_sha256"], PILOT_IDENTITY
    ):
        raise ValueError("Pinned model or pilot identity differs")
    proof = verify_checkpoint(weights, spec["revision"], spec["weight_sha256"], repository_id=spec["model_id"])
    if not proof["passed"]:
        raise ValueError("Pinned base model verification failed")
    cfg = config(root, weights, output)
    validate_frozen_config(cfg)
    examples = select_examples(load_manifest(cfg.dataset_manifest, cfg.image_root), "train", 1024, cfg.seed)
    if len(examples) != 1024 or len({item.sample_id for item in examples}) != 1024:
        raise ValueError("Frozen training selection differs")

    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke requires exactly one visible CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_training_components(cfg)
    names = trainable_guard(model)
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not names or trainable_count <= 0:
        raise ValueError("No intended LoRA parameters are trainable")
    progress["trainable_parameter_count"] = trainable_count
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_hook(finite_gradient)
    collator = QwenVQACollator(processor, cfg.image_size, max_sequence_length=cfg.max_sequence_length)
    losses = []

    class FiniteStep(TrainerCallback):
        gradient_finite = False
        optimizer_steps = 0

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [
                parameter.grad for parameter in kwargs["model"].parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError("Missing or nonfinite LoRA gradients before optimizer update")
            self.gradient_finite = True
            progress["gradient_finite"] = True

        def on_optimizer_step(self, args, state, control, **kwargs):
            if trainer.accelerator.optimizer_step_was_skipped:
                raise RuntimeError("Optimizer step was skipped")
            self.optimizer_steps += 1
            progress["optimizer_steps_completed"] = self.optimizer_steps
            if self.optimizer_steps > 1:
                raise RuntimeError("More than one optimizer update attempted")

    class SmokeTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite training loss before backward")
            losses.append(float(loss.detach().cpu()))
            progress["first_training_loss"] = losses[0]
            progress["final_training_loss"] = losses[-1]
            return result

    args = TrainingArguments(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=1,
        learning_rate=cfg.learning_rate,
        optim=cfg.optimizer,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.precision == "fp16",
        bf16=cfg.precision == "bf16",
        half_precision_backend=cfg.half_precision_backend,
        tf32=cfg.tf32,
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        disable_tqdm=True,
    )
    if (args.max_steps, args.per_device_train_batch_size,
        args.gradient_accumulation_steps) != (1, 1, 8):
        raise ValueError("Smoke must perform one update with batch 1 and accumulation 8")
    guard = FiniteStep()
    trainer = SmokeTrainer(model=model, args=args, train_dataset=examples, data_collator=collator, callbacks=[guard])
    try:
        trained = trainer.train()
    finally:
        progress["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    if (trainer.state.global_step, guard.optimizer_steps, guard.gradient_finite, len(losses)) != (1, 1, True, 8):
        raise RuntimeError("One-update/eight-microbatch contract failed")
    if not all(math.isfinite(loss) for loss in losses):
        raise FloatingPointError("Nonfinite recorded training loss")
    return {
        "status": "PASS",
        "package_source_fingerprint": hashlib.sha256(
            json.dumps(manifest["source_fingerprint"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "run_identity": binding(root),
        "manifest_identity": PILOT_IDENTITY,
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "trainable_parameter_count": trainable_count,
        "first_training_loss": losses[0],
        "final_training_loss": losses[-1],
        "loss_definition": "unscaled model loss on first and eighth training microbatches",
        "gradient_finite": guard.gradient_finite,
        "optimizer_steps_completed": guard.optimizer_steps,
        "trainer_global_step": trainer.state.global_step,
        "peak_allocated_gpu_bytes": progress["peak_allocated_gpu_bytes"],
        "trainer_metrics": trained.metrics,
    }


def run(argv=None, *, command_runner=subprocess.run):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 1200:
        parser.error("--max-seconds must be within 1..1200")
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Smoke output must be outside the immutable package/source tree")
    if args.output.exists():
        raise FileExistsError(f"Fresh output directory required: {args.output}")
    args.output.mkdir()
    began = time.monotonic()
    base = {
        "schema_version": 1,
        "smoke": "rgb-vqa-one-optimizer-update-v1",
        "status": "FAIL",
        "elapsed_seconds": None,
        "runtime_limit_seconds": args.max_seconds,
        "optimizer_steps_completed": 0,
        "gradient_finite": False,
        "first_training_loss": None,
        "final_training_loss": None,
        "trainable_parameter_count": None,
        "peak_allocated_gpu_bytes": None,
    }
    if args.worker:
        try:
            result = one_update(ROOT, args.weights, args.output, base)
            base.update(result)
        except Exception as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
        base["elapsed_seconds"] = time.monotonic() - began
        durable_json(args.output / RESULT, base)
        return 0 if base["status"] == "PASS" else 1
    command = [
        sys.executable, str(Path(__file__).resolve()), "--weights", str(args.weights),
        "--output", str(args.output / "worker"), "--max-seconds", str(args.max_seconds), "--worker"
    ]
    try:
        completed = command_runner(command, timeout=args.max_seconds, check=False)
        worker_result = args.output / "worker" / RESULT
        if worker_result.is_file():
            worker_data = json.loads(worker_result.read_text(encoding="utf-8"))
            base.update(worker_data)
            if worker_data.get("status") == "PASS" and not valid_success(worker_data):
                base["status"] = "FAIL"
                base["error"] = "Worker success result failed integrity checks"
        else:
            base["error"] = f"Worker exited {completed.returncode} without a durable result"
        if completed.returncode != 0:
            base["status"] = "FAIL"
    except subprocess.TimeoutExpired:
        base["status"] = "TIMEOUT"
        base["error"] = "Hard wall-clock limit exceeded; worker terminated"
    base["elapsed_seconds"] = time.monotonic() - began
    durable_json(args.output / RESULT, base)
    return 0 if base["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
