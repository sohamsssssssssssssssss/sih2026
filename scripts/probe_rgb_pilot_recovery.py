"""Two-phase, bounded checkpoint recovery probe for the frozen RGB pilot."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rgb_pilot_package import (
    REQUIRED_CHECKPOINT_FILES, binding, check_payload, config, trainable_guard, verify_base,
    verify_resume,
)
from scripts.smoke_rgb_pilot_train import durable_json, validate_frozen_config
from training.remote_sensing import (
    QwenVQACollator, load_manifest, load_training_components, select_examples, sha256_file,
)

REQUIRED = REQUIRED_CHECKPOINT_FILES
LIMITS = {"start": 1800, "resume": 900}


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def append_batch(path, ids, step):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step_before_update": step, "sample_ids": ids}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_batches(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_continuity(expected, phase16, phase17):
    if len(expected) != 1024 or len(set(expected)) != 1024:
        raise ValueError("Expected dataloader order is incomplete or duplicated")
    if len(phase16) != 128 or phase16 != expected[:128]:
        raise ValueError("Step-16 sample IDs replayed, skipped, or reordered")
    if phase17 is not None and (len(phase17) != 8 or phase17 != expected[128:136]):
        raise ValueError("Resume boundary replayed, skipped, or reordered")
    return {"last_before_checkpoint": phase16[-8:], "first_after_resume": phase17}


def checkpoint_files(checkpoint):
    files = {}
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Checkpoint symlinks are not accepted")
        if path.is_file() and path.name != "pilot-checkpoint.json":
            files[str(path.relative_to(checkpoint))] = sha256_file(path)
    return files


def verify_checkpoint(checkpoint, identity, *, expected_step=16):
    marker = verify_resume(checkpoint, identity)
    if marker.get("step") != expected_step:
        raise ValueError("Checkpoint step mismatch")
    if type(marker.get("scaler_expected")) is not bool:
        raise ValueError("Checkpoint must record whether the pinned runtime used a scaler")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("max_steps") != 128 or state.get("global_step") != expected_step:
        raise ValueError("Checkpoint Trainer state is not step 16 of 128")
    return marker


def make_backup(checkpoint, backup, identity):
    verify_checkpoint(checkpoint, identity)
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    shutil.copytree(checkpoint, backup)
    verify_checkpoint(backup, identity)
    for path in backup.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    backup.chmod(0o555)
    return backup


def training_arguments(cfg, output, *, use_cpu=False):
    from transformers import TrainingArguments
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
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
        fp16=cfg.precision == "fp16" and not use_cpu,
        bf16=cfg.precision == "bf16" and not use_cpu,
        half_precision_backend=cfg.half_precision_backend,
        fp16_full_eval=cfg.fp16_full_eval,
        bf16_full_eval=cfg.bf16_full_eval,
        tf32=cfg.tf32,
        logging_steps=1,
        save_strategy="steps",
        save_steps=16,
        save_total_limit=None,
        save_only_model=False,
        ignore_data_skip=False,
        report_to="none",
        remove_unused_columns=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=use_cpu,
    )
    if (args.max_steps, args.per_device_train_batch_size, args.gradient_accumulation_steps,
        args.save_steps, args.save_only_model, args.ignore_data_skip) != (128, 1, 8, 16, False, False):
        raise ValueError("Recovery probe changed the frozen training schedule")
    return args


def expected_order(examples, cfg, output):
    import torch
    from transformers import Trainer
    mirror = Trainer(
        model=torch.nn.Linear(1, 1),
        args=training_arguments(cfg, output / "order", use_cpu=True),
        train_dataset=examples,
        data_collator=lambda batch: {"sample_ids": [item.sample_id for item in batch]},
    )
    return [sample_id for batch in mirror.get_train_dataloader() for sample_id in batch["sample_ids"]]


def stop_control(state, control, target):
    if state.global_step > target:
        raise RuntimeError("Recovery probe exceeded its optimizer-step bound")
    if state.global_step == target:
        control.should_training_stop = True
        if target == 16:
            control.should_save = True
    return control


def check_start_step(step, phase):
    expected = 0 if phase == "start" else 16
    if step != expected:
        raise RuntimeError(f"{phase} began at step {step}; expected {expected}")


def worker(phase, weights, output, backup, prior_report):
    import torch
    from transformers import Trainer, TrainerCallback
    from models.qwen_vl.stage1 import resolve_vision_module

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    check_payload(ROOT)
    identity = binding(ROOT)
    cfg = config(ROOT, weights, output)
    validate_frozen_config(cfg)
    proof = verify_base(ROOT, weights)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Recovery probe requires one visible CUDA GPU")
    examples = select_examples(load_manifest(cfg.dataset_manifest, cfg.image_root), "train", 1024, cfg.seed)
    if len(examples) != 1024 or len({item.sample_id for item in examples}) != 1024:
        raise ValueError("Frozen training selection differs")
    target = 16 if phase == "start" else 17
    if phase == "resume":
        marker = verify_checkpoint(backup, identity)
        if prior_report["checkpoint_files_sha256"] != marker["files_sha256"]:
            raise ValueError("Backup differs from phase-16 checkpoint report")
    model, processor = load_training_components(cfg)
    if resolve_vision_module(model).patch_embed.proj.in_channels != 3:
        raise ValueError("Original RGB projection required")
    trainable_guard(model)
    expected = expected_order(examples, cfg, output)
    if len(expected) != 1024 or len(set(expected)) != 1024:
        raise ValueError("CPU mirror dataloader order is invalid")
    if phase == "resume" and prior_report["expected_order_sha256"] != digest_ids(expected):
        raise ValueError("Dataloader order changed across phases")
    collator = QwenVQACollator(processor, cfg.image_size, max_sequence_length=cfg.max_sequence_length)
    batches_path = output / "trained-samples.jsonl"

    def collate(batch):
        encoded = collator(batch)
        encoded["probe_sample_ids"] = [item.sample_id for item in batch]
        return encoded

    microbatches = 0

    class ProbeCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            check_start_step(state.global_step, phase)

        def on_step_end(self, args, state, control, **kwargs):
            return stop_control(state, control, target)

        def on_save(self, args, state, control, **kwargs):
            if phase != "start" or state.global_step != 16:
                return
            checkpoint = Path(args.output_dir) / "checkpoint-16"
            scaler_expected = getattr(trainer.accelerator, "scaler", None) is not None
            files = checkpoint_files(checkpoint)
            if ("scaler.pt" in files) != scaler_expected:
                raise ValueError("Pinned Trainer scaler save behavior differs")
            durable_json(checkpoint / "pilot-checkpoint.json", {
                "run_identity": identity, "step": 16, "scaler_expected": scaler_expected,
                "files_sha256": files,
            })
            verify_checkpoint(checkpoint, identity)

    class ProbeTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            nonlocal microbatches
            if microbatches >= (128 if phase == "start" else 8):
                raise RuntimeError("Recovery probe exceeded its microbatch bound")
            ids = inputs.pop("probe_sample_ids")
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            microbatches += 1
            append_batch(batches_path, ids, self.state.global_step)
            return loss

    trainer = ProbeTrainer(
        model=model, args=training_arguments(cfg, output), train_dataset=examples,
        data_collator=collate, callbacks=[ProbeCallback()],
    )
    if phase == "resume" and (getattr(trainer.accelerator, "scaler", None) is not None) != marker["scaler_expected"]:
        raise ValueError("Resume runtime scaler differs from saved checkpoint")
    trainer.train(resume_from_checkpoint=str(backup) if phase == "resume" else None)
    if trainer.state.global_step != target:
        raise RuntimeError(f"Probe ended at step {trainer.state.global_step}, expected {target}")
    rows = read_batches(batches_path)
    actual = [sample_id for row in rows for sample_id in row["sample_ids"]]
    if phase == "start":
        check_continuity(expected, actual, None)
        checkpoint = output / "checkpoints/checkpoint-16"
        marker = verify_checkpoint(checkpoint, identity)
        result = {"checkpoint": str(checkpoint), "checkpoint_files_sha256": marker["files_sha256"],
                  "scaler_expected": marker["scaler_expected"],
                  "production_resume_compatible": True}
    else:
        check_continuity(expected, prior_report["trained_sample_ids"], actual)
        result = {"resume_from": str(backup), "boundary": check_continuity(
            expected, prior_report["trained_sample_ids"], actual)}
    return {
        "status": "PASS", "phase": phase, "global_step": target, "run_identity": identity,
        "manifest_identity": json.loads((ROOT / "pilot-config.json").read_text())["manifest_identity"],
        "expected_order_sha256": digest_ids(expected), "expected_sample_ids": expected,
        "trained_sample_ids": actual,
        "training_microbatches": len(rows), "base_verification": proof, **result,
    }


def parent(args):
    root = args.recovery_root.resolve()
    state_path = root / "probe-state.json"
    if args.phase == "start":
        # shortcut: restart protection is scoped to this recovery root; use an external run registry across roots.
        if root.exists():
            raise FileExistsError("Recovery root already exists; refusing step-0 restart")
        root.mkdir(parents=True)
        state = {"status": "start_reserved", "run_identity": binding(ROOT)}
        durable_json(state_path, state)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "checkpoint_ready" or state.get("run_identity") != binding(ROOT):
            raise ValueError("Resume requires the completed matching phase-16 state")
    backup = root / "immutable-backup/checkpoint-16"
    prior_path = root / "phase16-report.json"
    if args.phase == "resume":
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        marker = verify_checkpoint(backup, state["run_identity"])
        if prior.get("checkpoint_files_sha256") != marker["files_sha256"]:
            raise ValueError("Resume backup does not match phase-16 report")
    output = root / ("phase16-output" if args.phase == "start" else "phase17-output")
    if output.exists():
        raise FileExistsError(f"Fresh output required: {output}")
    output.mkdir()
    report_path = root / ("phase16-report.json" if args.phase == "start" else "phase17-report.json")
    began = time.monotonic()
    command = [sys.executable, str(Path(__file__).resolve()), args.phase, "--weights", str(args.weights),
               "--recovery-root", str(root), "--max-seconds", str(args.max_seconds), "--worker"]
    try:
        completed = subprocess.run(command, timeout=args.max_seconds, check=False,
                                   env={**os.environ, "SATQUERY_RECOVERY_WORKER": "1"})
        worker_path = output / "worker-report.json"
        report = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.exists() else {
            "status": "FAIL", "error": f"Worker exited {completed.returncode} without a report"}
        if completed.returncode != 0:
            report["status"] = "FAIL"
        if report.get("status") == "PASS" and args.phase == "start":
            if (report.get("run_identity") != state["run_identity"]
                    or report.get("global_step") != 16
                    or report.get("expected_order_sha256") != digest_ids(report["expected_sample_ids"])):
                raise ValueError("Phase-16 worker result identity or step differs")
            check_continuity(report["expected_sample_ids"], report["trained_sample_ids"], None)
            checkpoint = output / "checkpoints/checkpoint-16"
            make_backup(checkpoint, backup, state["run_identity"])
            report["backup"] = str(backup)
        elif report.get("status") == "PASS":
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if (report.get("run_identity") != state["run_identity"]
                    or report.get("global_step") != 17
                    or report.get("expected_order_sha256") != prior["expected_order_sha256"]):
                raise ValueError("Phase-17 worker result identity, step, or order differs")
            check_continuity(prior["expected_sample_ids"], prior["trained_sample_ids"],
                             report["trained_sample_ids"])
    except subprocess.TimeoutExpired:
        report = {"status": "TIMEOUT", "error": "Worker killed at hard wall-clock limit"}
    except Exception as exc:
        report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    report.update(phase=args.phase, elapsed_seconds=time.monotonic()-began,
                  runtime_limit_seconds=args.max_seconds)
    durable_json(report_path, report)
    if report["status"] == "PASS":
        state["status"] = "checkpoint_ready" if args.phase == "start" else "resume_verified"
        durable_json(state_path, state)
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("start", "resume"))
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=int)
    parser.add_argument("--acknowledge-fresh-start", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 121 <= args.max_seconds <= LIMITS[args.phase]:
        parser.error(f"{args.phase} requires 121..{LIMITS[args.phase]} seconds")
    if args.phase == "start" and not args.acknowledge_fresh_start and not args.worker:
        parser.error("start requires --acknowledge-fresh-start and a new recovery root")
    if args.recovery_root.resolve().is_relative_to(ROOT.resolve()):
        parser.error("Recovery outputs must be outside the immutable package")
    if args.worker:
        if os.environ.get("SATQUERY_RECOVERY_WORKER") != "1":
            raise ValueError("Worker must be launched by the bounded parent")
        output = args.recovery_root / ("phase16-output" if args.phase == "start" else "phase17-output")
        began = time.monotonic()
        try:
            prior = (json.loads((args.recovery_root/"phase16-report.json").read_text())
                     if args.phase == "resume" else None)
            report = worker(args.phase, args.weights, output,
                            args.recovery_root/"immutable-backup/checkpoint-16", prior)
        except Exception as exc:
            report = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        report["worker_elapsed_seconds"] = time.monotonic()-began
        durable_json(output/"worker-report.json", report)
        return 0 if report["status"] == "PASS" else 1
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
