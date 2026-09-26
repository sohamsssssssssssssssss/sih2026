"""Offline, configurable QLoRA adaptation for Qwen2.5-VL remote-sensing VQA."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.remote_sensing import (  # noqa: E402
    BASE_MODEL_ID,
    QwenVQACollator,
    TrainingConfig,
    load_manifest,
    load_training_components,
    select_examples,
    sha256_file,
)

REPORT_NAME = "training-report.json"


def _versions() -> dict[str, object]:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "qwen-vl-utils"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_runtime"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        result.update(cuda_available=False, cuda_runtime=None, gpu=None)
    return result


def _peak_cuda_memory() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None


def build_report(config: TrainingConfig, examples: list, status: str) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": {
            "model_id": BASE_MODEL_ID,
            "local_path": str(config.model_path.resolve()),
            "revision": config.model_revision,
            "sha256": None,
        },
        "dataset": {
            "manifest": str(config.dataset_manifest.resolve()),
            "manifest_sha256": sha256_file(config.dataset_manifest),
            "identities": sorted({item.dataset for item in examples}),
            "sources": sorted({item.source for item in examples}),
            "split": config.split,
            "training_sample_count": len(examples),
            "sample_ids": [item.sample_id for item in examples],
        },
        "configuration": config.serializable(),
        "runtime": _versions(),
        "planned_adapter_path": str((config.output_dir / "adapter").resolve()),
        "adapter_path": (
            str((config.output_dir / "adapter").resolve())
            if status == "training_complete"
            else None
        ),
        "evaluation_claim": None,
    }


def run(
    config: TrainingConfig,
    dry_run: bool = False,
    *,
    component_loader=None,
    collator_factory=None,
) -> dict:
    started = time.perf_counter()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config.validate()
    if config.split != "train":
        raise ValueError("Training entry point only permits the train split")
    all_examples = load_manifest(config.dataset_manifest, config.image_root)
    examples = select_examples(all_examples, config.split, config.max_samples, config.seed)
    model, processor = (component_loader or load_training_components)(config)
    collator = (collator_factory or QwenVQACollator)(processor, config.image_size)
    if dry_run:
        import torch

        model.train()
        batch = collator(examples[: config.batch_size])
        device = next(model.parameters()).device
        batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
        output = model(**batch)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("Dry-run forward pass did not produce a finite training loss")
        output.loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("Dry-run backward pass did not produce finite adapter gradients")
        model.zero_grad(set_to_none=True)
        report = build_report(config, examples, "dry_run_passed")
    else:
        from transformers import Trainer, TrainingArguments

        arguments = TrainingArguments(
            output_dir=str(config.output_dir / "checkpoints"),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=config.precision == "fp16",
            bf16=config.precision == "bf16",
            logging_steps=1,
            save_strategy="steps",
            save_steps=max(1, config.max_steps if config.max_steps > 0 else 100),
            # Checkpoints exist only for --resume-from-checkpoint across Kaggle sessions.
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            seed=config.seed,
            data_seed=config.seed,
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=examples,
            data_collator=collator,
        )
        resume = config.resume_from_checkpoint
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        adapter_dir = config.output_dir / "adapter"
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        report = build_report(config, examples, "training_complete")
        report["trainer_metrics"] = result.metrics
        report["trainer_log_history"] = trainer.state.log_history
        report["adapter_size_bytes"] = sum(
            path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
        )
    report["peak_cuda_memory_bytes"] = _peak_cuda_memory()
    report["wall_clock_seconds"] = time.perf_counter() - started
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv: list[str] | None = None) -> tuple[TrainingConfig, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--model-revision", help="verified Hugging Face commit of --model-path")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = TrainingConfig(
        model_path=args.model_path,
        dataset_manifest=args.dataset_manifest,
        image_root=args.image_root,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
        max_samples=args.max_samples,
        image_size=args.image_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        precision=args.precision,
        quantization=args.quantization,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(item.strip() for item in args.lora_target_modules.split(",") if item.strip()),
        model_revision=args.model_revision,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    return config, args.dry_run


def main() -> int:
    config, dry_run = parse_args()
    report = run(config, dry_run)
    print(json.dumps({"status": report["status"], "report": str(config.output_dir / REPORT_NAME)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
