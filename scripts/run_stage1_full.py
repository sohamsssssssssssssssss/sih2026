"""Stage-1 full-run training stage: shards in, resumable checkpoints out.

The loop-7 smoke runner trains on 16 packaged patches for a fixed small step
count. A real run has to survive the Kaggle 12-hour session limit, so this stage
adds what a long run needs and nothing else:

* a shard-backed dataset whose sample order is a pure function of (seed, epoch),
  so a checkpoint position of (epoch, cursor) reproduces the sequence exactly;
* a checkpoint every ``checkpoint_every_steps`` optimizer steps holding adapter,
  optimizer, scheduler, data position, and RNG state;
* a wall-clock guard that writes a final checkpoint and exits cleanly instead of
  being killed mid-step;
* periodic caption-only loss on an eval-tile shard, with no optimizer updates.

Model construction, LoRA settings, and the scheduler are imported from
scripts/run_stage1_smoke.py so the full run and the smoke run cannot drift.
Every number written comes from this run; nothing is asserted as a quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from data.bigearthnet_s2 import apply_contract, load_norm_contract, resample_bands
from data.bigearthnet_shards import ShardDataset
from models.qwen_vl.stage1 import pack_s2_pixel_values
from scripts.stage1_runtime import verify_checkpoint, bounded_checkpoint_save, bounded_write_json
from scripts.run_stage1_smoke import (build_model, canonical_digest, amp_context, make_scaler,
                                     rng_state, restore_rng, trainable_state, load_trainable_state)

PROMPT_TEMPLATE = (
    "system\nYou are a helpful assistant.\nuser\n"
    + "<|vision_start|>" + "<|image_pad|>" * 16 + "<|vision_end|>"
    + "Describe this satellite image.\nassistant\n"
)
IMAGE_TOKEN_COUNT = 16
# Keys that may legitimately differ between sessions of the same run: they bound
# the session, not the training. Everything else is part of the config identity
# a checkpoint must match.
SESSION_ONLY_KEYS = ("name", "max_steps_per_session", "wall_clock_limit_seconds",
                     "wall_clock_reserve_seconds", "resume", "eval_patches", "steps",
                     "budget_seconds")


def config_identity(config: dict) -> str:
    """SHA256 over the training-relevant config keys only."""
    relevant = {key: value for key, value in sorted(config.items())
                if key not in SESSION_ONLY_KEYS}
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# plan from measured throughput (loop 9)
# --------------------------------------------------------------------------- #
THROUGHPUT_REQUIRED_FIELDS = ("measured_samples_per_second", "shard_manifest_sha256",
                              "config_sha256", "device", "dtype")


def load_throughput(path: Path, *, shard_manifest_sha256: str, config_sha256: str) -> dict:
    """Load a T4 smoke throughput record, refusing anything unusable.

    The stage must not size a multi-hour run from a placeholder: the record has
    to carry a positive measured rate, and it has to have been measured on the
    same shard manifest and the same training config identity.
    """
    payload = json.loads(Path(path).read_text())
    missing = [field for field in THROUGHPUT_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Throughput JSON is missing required fields: {missing}")
    rate = float(payload["measured_samples_per_second"])
    if not (rate > 0) or rate != rate or rate in (float("inf"),):
        raise ValueError(f"Throughput JSON has an unusable samples_per_second: {rate}")
    if payload["shard_manifest_sha256"] != shard_manifest_sha256:
        raise ValueError("Throughput was measured on a different shard manifest")
    if payload["config_sha256"] != config_sha256:
        raise ValueError("Throughput was measured under a different training config identity")
    return payload


def plan_from_throughput(config: dict, throughput: dict, n_available: int,
                        *, budget_seconds: float | None = None) -> dict:
    """Turn measured throughput into the session plan, with no placeholder steps.

    max_optimizer_steps = floor(budget_seconds x samples_per_second / effective_batch),
    where effective_batch = batch_size x gradient_accumulation_steps. The
    wall-clock guard additionally caps the steps one session may run.
    """
    batch = int(config.get("batch_size", 1)) * int(config["gradient_accumulation_steps"])
    rate = float(throughput["measured_samples_per_second"])
    budget = float(budget_seconds if budget_seconds is not None
                   else config["budget_seconds"])
    if budget <= 0:
        raise ValueError("budget_seconds must be positive")
    total_steps = int(budget * rate // batch)
    if total_steps <= 0:
        raise ValueError(
            f"Computed plan allows zero optimizer steps: budget_seconds={budget}, "
            f"samples_per_second={rate}, effective_batch={batch}")
    limit = float(config.get("wall_clock_limit_seconds", 39600))
    reserve = float(config.get("wall_clock_reserve_seconds", 300))
    session_steps = int(max(limit - reserve, 0) * rate // batch)
    n_used = int(n_available)
    consumed = total_steps * batch
    epochs = consumed / n_used if n_used else None
    return {
        "measured_samples_per_second": rate,
        "effective_batch": batch,
        "batch_size": int(config.get("batch_size", 1)),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "budget_seconds": budget,
        "wall_clock_limit_seconds": limit,
        "wall_clock_reserve_seconds": reserve,
        "max_optimizer_steps": total_steps,
        "max_steps_per_session": session_steps,
        "n_available": int(n_available),
        "n_used": n_used,
        "samples_covered": consumed,
        "epochs": epochs,
        "formula": ("max_optimizer_steps = floor(budget_seconds x measured_samples_per_second / "
                    "effective_batch); effective_batch = batch_size x gradient_accumulation_steps"),
        "throughput_source": throughput.get("source"),
        "throughput_device": throughput.get("device"),
        "throughput_dtype": throughput.get("dtype"),
        "throughput_shard_manifest_sha256": throughput["shard_manifest_sha256"],
        "throughput_config_sha256": throughput["config_sha256"],
        "throughput_timestamp_utc": throughput.get("timestamp_utc"),
        "plan_computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_prompt_ids(tokenizer, template: str = PROMPT_TEMPLATE) -> list[int]:
    prompt_ids = tokenizer(template, add_special_tokens=False).input_ids
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if prompt_ids.count(image_token) != IMAGE_TOKEN_COUNT:
        raise ValueError("Prompt template must carry exactly 16 image tokens")
    return list(prompt_ids)


def make_sample(dataset, entry, prompt_ids, tokenizer, contract, device, dtype,
                vision_fp32: bool):
    """Tokenize one shard row and pack its pixels; identical math to the smoke path.

    The shard stores each band at its native GSD, so the same rasterio bilinear
    upsampling the frozen loader applies to extracted directories is applied
    here, before the contract.
    """
    row = dataset.row(entry)
    native = dataset.raw_patch(entry)
    pixels = torch.from_numpy(np.stack(resample_bands(native)))
    packed, grid = pack_s2_pixel_values(apply_contract(pixels, contract))
    caption_ids = tokenizer(row["caption"] + "<|im_end|>", add_special_tokens=False).input_ids
    ids = prompt_ids + caption_ids
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([[-100] * len(prompt_ids) + list(caption_ids)],
                               dtype=torch.long, device=device),
        "pixel_values": packed.to(device=device, dtype=(torch.float32 if vision_fp32 else dtype)),
        "image_grid_thw": grid.to(device=device),
    }


def evaluate_caption_loss(model, dataset, entries, prompt_ids, tokenizer, contract,
                          device, dtype, vision_fp32, should_stop=None) -> dict:
    """Mean caption-only cross-entropy; no optimizer state is touched."""
    model.eval()
    losses = []
    with torch.no_grad():
        for entry in entries:
            if should_stop is not None and should_stop():
                break
            sample = make_sample(dataset, entry, prompt_ids, tokenizer, contract,
                                 device, dtype, vision_fp32)
            with amp_context(device, dtype):
                value = model(**sample).loss.item()
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite eval caption loss")
            losses.append(value)
    model.train()
    return {"n": len(losses), "mean_loss": sum(losses) / len(losses) if losses else None,
            "min_loss": min(losses) if losses else None, "max_loss": max(losses) if losses else None}


def build_optimizer_scheduler(config, groups, total_steps: int):
    optimizer = (torch.optim.Adafactor(groups, foreach=False)
                 if config["optimizer"] == "adafactor"
                 else torch.optim.AdamW(groups, foreach=False))
    warmup = max(1, math.ceil(total_steps * config["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.0) if step < warmup else
        0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )
    return optimizer, scheduler


def save_checkpoint(path: Path, *, model, optimizer, scheduler, dataset, step: int,
                    config_sha256: str, wall_clock_seconds: float,
                    sample_sequence_digest: str, scaler=None, total_steps: int | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bounded_checkpoint_save({
        "model": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data_position": dataset.state(),
        "step": step,
        "config_sha256": config_sha256,
        "sample_sequence_digest": sample_sequence_digest,
        "scaler": scaler.state_dict() if scaler is not None else {},
        "total_steps": total_steps,
        **rng_state(),
        "wall_clock_seconds": wall_clock_seconds,
    }, path)


def run_full(
    config_path: Path,
    checkpoint_dir: Path,
    train_shards: Path,
    output: Path,
    *,
    eval_shards: Path | None = None,
    resume_from: Path | None = None,
    wall_clock_limit_seconds: float | None = None,
    throughput_json: Path | None = None,
    budget_seconds: float | None = None,
    measure_throughput_steps: int | None = None,
    now=time.time,
) -> dict:
    """Run or resume one full Stage-1 training session and return its record.

    The step count is never taken from the config: ``steps`` and
    ``max_steps_per_session`` are expected to be null and the plan is computed
    from a measured throughput record (loop 9). A session refuses to start
    without one.
    """
    if output.exists() and json.loads(output.read_text()).get("passed") is False:
        raise FileExistsError("Keep failed attempts; choose a new output path")
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("The current sample path requires batch_size=1")
    config_sha256 = config_identity(config)
    config_file_sha256 = hashlib.sha256(config_bytes).hexdigest()
    package_root = config_path.parent
    contract = load_norm_contract(package_root / config["contract"])
    device = config["device"]
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
    vision_fp32 = bool(config.get("vision_fp32", False))
    limit = (wall_clock_limit_seconds if wall_clock_limit_seconds is not None
             else float(config.get("wall_clock_limit_seconds", 39600)))
    reserve = float(config.get("wall_clock_reserve_seconds", 300))
    dataset = ShardDataset(train_shards / "shards-manifest.json", seed=int(config["seed"]))
    eval_dataset = (ShardDataset(eval_shards / "shards-manifest.json", seed=int(config["seed"]))
                    if eval_shards is not None else None)
    if not dataset.split_name.endswith("/train"):
        raise ValueError("Training requires the train split")
    if eval_dataset is not None and (not eval_dataset.split_name.endswith("/eval")
                                     or dataset.tiles & eval_dataset.tiles):
        raise ValueError("Eval shard must be eval-only with disjoint tiles")
    if measure_throughput_steps is not None:
        if not 1 <= measure_throughput_steps <= 50 or resume_from is not None or throughput_json is not None:
            raise ValueError("Shard smoke requires1..50 fresh steps, no resume/throughput input")
        if output.exists() or output.with_suffix(".pt").exists():
            raise FileExistsError("Shard smoke requires a new output; keep old attempts")
        total_steps = steps_to_run_cap = measure_throughput_steps
        plan = {"max_optimizer_steps": total_steps, "max_steps_per_session": total_steps,
                "n_used": len(dataset), "epochs": total_steps * int(config["gradient_accumulation_steps"]) / len(dataset),
                "kind": "bounded sharded throughput smoke; NOT a full training plan"}
    else:
        if throughput_json is None:
            raise ValueError(
                "A measured throughput JSON is required; pass --throughput-json from the T4 smoke stage. "
                "config steps/max_steps_per_session are null on purpose and never used as fallback.")
        throughput = load_throughput(Path(throughput_json),
                                     shard_manifest_sha256=dataset.manifest_sha256,
                                     config_sha256=config_sha256)
        if throughput["device"] != device or throughput["dtype"] != config["precision"]:
            raise ValueError("Throughput device/dtype differs from current config")
        plan = plan_from_throughput(config, throughput, len(dataset), budget_seconds=budget_seconds)
        total_steps = plan["max_optimizer_steps"]
        steps_to_run_cap = plan["max_steps_per_session"]
        if wall_clock_limit_seconds is not None:
            plan["wall_clock_limit_seconds"] = float(wall_clock_limit_seconds)
            steps_to_run_cap = int(max(float(wall_clock_limit_seconds) - plan["wall_clock_reserve_seconds"], 0)
                                   * throughput["measured_samples_per_second"] // plan["effective_batch"])
    started_at = now()
    result = {
        "kind": "stage-1 full run",
        "git_sha": config["git_sha"],
        "revision": config["revision"],
        "split_name": dataset.split_name,
        "n": len(dataset),
        "seed": int(config["seed"]),
        "device": device,
        "dtype": config["precision"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weight_sha256": config["weight_sha256"],
        "config": str(config_path),
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "shard_manifest_sha256": dataset.manifest_sha256,
        "eval_shard_manifest_sha256": eval_dataset.manifest_sha256 if eval_dataset else None,
        "steps_requested": total_steps,
        "steps_completed": 0,
        "wall_clock_limit_seconds": limit,
        "wall_clock_reserve_seconds": reserve,
        "plan": plan,
        "n_used": plan["n_used"],
        "steps_planned": total_steps,
        "max_steps_per_session_planned": steps_to_run_cap,
        "epochs_planned": plan["epochs"],
        "checkpoint_every_steps": int(config["checkpoint_every_steps"]),
        "eval_every_steps": int(config.get("eval_every_steps", 2000)),
        "patches_per_epoch": len(dataset),
        "vision_fp32": vision_fp32,
        "attention": config["attention"],
        "optimizer": config["optimizer"],
        "losses": [],
        "eval_losses": [],
        "sample_sequence_digest": "",
        "status": "started",
    }
    # A resumable hash chain over (step, patch_id): a running SHA256 cannot be
    # continued from its own hex digest, so each link folds in the previous one.
    sequence = {"chain": ""}

    def feed(step: int, patch_id: str) -> None:
        sequence["chain"] = hashlib.sha256(
            f"{sequence['chain']}|{step}:{patch_id}".encode()).hexdigest()

    def save() -> None:
        result["sample_sequence_digest"] = sequence["chain"]
        output.parent.mkdir(parents=True, exist_ok=True)
        bounded_write_json(output, result)

    checkpoint_path = Path(output).with_suffix(".pt")
    save()
    if total_steps <= 0:
        result["status"] = "plan_refused"
        result["passed"] = False
        save()
        raise ValueError(f"Computed plan allows zero optimizer steps: {plan}")
    try:
        proof = verify_checkpoint(checkpoint_dir, config["revision"], config["weight_sha256"])
        result["checkpoint_verification"] = proof
        result.update({key: proof[key] for key in ("weights_match", "revision_source", "revision_matches", "passed")})
        save()
        if not proof["passed"]:
            raise ValueError("Checkpoint revision proof or weight SHA256 mismatch")
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, groups = build_model(config, checkpoint_dir, dtype, device)
        def check_vision(_module, _inputs, value):
            if not torch.isfinite(value).all():
                raise FloatingPointError("Vision tower produced inf/NaN; use vision-fp32 variant")

        if hasattr(model, "get_base_model"):
            model.get_base_model().visual.register_forward_hook(check_vision)
        result["group_parameter_counts"] = {group["name"]: sum(p.numel() for p in group["params"])
                                            for group in groups}
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        prompt_ids = build_prompt_ids(tokenizer, config.get("prompt_template", PROMPT_TEMPLATE))
        optimizer, scheduler = build_optimizer_scheduler(config, groups, total_steps)
        scaler = make_scaler(device, dtype)
        start_step = 0
        resume_state = (resume_from if resume_from is not None else
                        (checkpoint_path if measure_throughput_steps is None and config.get("resume", True) and checkpoint_path.exists()
                         else None))
        if resume_state is not None:
            saved = torch.load(resume_state, map_location=device, weights_only=False)
            if saved.get("config_sha256") not in (None, config_sha256):
                raise ValueError("Checkpoint was written by a different config")
            load_trainable_state(model, saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            dataset.load_state(saved["data_position"])
            if saved.get("total_steps") != total_steps:
                raise ValueError("Resume must retain the original total-step schedule")
            scaler.load_state_dict(saved["scaler"])
            restore_rng(saved)
            start_step = int(saved["step"])
            result["resumed_from"] = str(resume_state)
            result["resumed_from_step"] = start_step
            result["resumed_from_data_position"] = saved["data_position"]
            result["optimizer_state_digest_after_load"] = canonical_digest(optimizer.state_dict())
            sequence["chain"] = str(saved.get("sample_sequence_digest", ""))
        model.train()
        eval_entries = None
        if eval_dataset is not None and len(eval_dataset):
            rng = torch.Generator().manual_seed(int(config["seed"]) + 1)
            order = torch.randperm(len(eval_dataset), generator=rng).tolist()
            eval_entries = order[:int(config.get("eval_patches", 2000))]

        def evaluate(tag: str) -> None:
            entry = {
                "step": step + 1,
                "wall_clock_seconds": now() - started_at,
                "eval_split_name": eval_dataset.split_name,
                **evaluate_caption_loss(model, eval_dataset, eval_entries, prompt_ids,
                                        tokenizer, contract, device, dtype, vision_fp32,
                                        should_stop=lambda: now() - started_at >= limit - reserve),
            }
            result["eval_losses"].append(entry)
            result["status"] = f"{tag}; eval completed {entry['n']} samples at step {step + 1}"
            save()

        steps_to_run = min(total_steps, start_step + steps_to_run_cap)
        for step in range(start_step, steps_to_run):
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            used: list[int] = []
            token_counts = []
            for _micro in range(int(config["gradient_accumulation_steps"])):
                position, entry = dataset.next_entry()
                sample = make_sample(dataset, entry, prompt_ids, tokenizer, contract,
                                     device, dtype, vision_fp32)
                token_counts.append(sample["input_ids"].numel())
                with amp_context(device, dtype):
                    loss = model(**sample).loss
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                scaler.scale(loss / int(config["gradient_accumulation_steps"])).backward()
                losses.append(value)
                used.append(entry)
            for entry in used:
                feed(step, dataset.entries[entry][2])
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            if not math.isfinite(grad_norm.item()):
                raise FloatingPointError(f"Nonfinite gradient norm at step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            peak = (torch.cuda.max_memory_allocated() if device == "cuda" else
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            accumulation = int(config["gradient_accumulation_steps"])
            result["losses"].append({
                "step": step + 1, "loss": sum(losses) / len(losses),
                "seconds": elapsed, "samples_per_second": accumulation / elapsed,
                "peak_memory_bytes": peak, "lr": scheduler.get_last_lr(),
                "tokens_per_sample": token_counts,
                "epoch": dataset.epoch, "cursor": dataset.cursor,
                "wall_clock_seconds": now() - started_at,
            })
            result["steps_completed"] = step + 1
            result["status"] = f"completed {step + 1}/{steps_to_run} steps"
            save()

            if (step + 1) % int(config["checkpoint_every_steps"]) == 0 or step + 1 == steps_to_run:
                result["checkpoint_storage"] = save_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                                scheduler=scheduler, dataset=dataset, step=step + 1,
                                config_sha256=config_sha256,
                                wall_clock_seconds=now() - started_at,
                                sample_sequence_digest=sequence["chain"],
                                scaler=scaler, total_steps=total_steps)
                result["optimizer_state_digest"] = canonical_digest(optimizer.state_dict())
                result["last_checkpoint"] = str(checkpoint_path)
                save()
            if eval_entries is not None and (step + 1) % result["eval_every_steps"] == 0:
                evaluate("running")
                save()

            if now() - started_at >= limit - reserve:
                result["checkpoint_storage"] = save_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                                scheduler=scheduler, dataset=dataset, step=step + 1,
                                config_sha256=config_sha256,
                                wall_clock_seconds=now() - started_at,
                                sample_sequence_digest=sequence["chain"],
                                scaler=scaler, total_steps=total_steps)
                result["status"] = "wall_clock_guard_exit"
                result["wall_clock_stopped_at_step"] = step + 1
                result["wall_clock_seconds"] = now() - started_at
                result["last_checkpoint"] = str(checkpoint_path)
                result["finished"] = True
                save()
                return result

        result["status"] = "completed"
        result["finished"] = True
        result["wall_clock_seconds"] = now() - started_at
        save()
    except Exception:
        result["passed"] = False
        result["oom"] = "out of memory" in traceback.format_exc().lower()
        if str(device).startswith("cuda") and torch.cuda.is_initialized():
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        save()
        raise
    return result


def measure_shard_throughput(config_path, checkpoint_dir, train_shards, output, *, steps=50):
    """Smoke the actual shard/full path; write a consumer-compatible record."""
    run_output = Path(output).with_name(Path(output).stem + "-run.json")
    result = run_full(Path(config_path), Path(checkpoint_dir), Path(train_shards), run_output,
                      measure_throughput_steps=steps)
    if result["steps_completed"] != steps or result["status"] != "completed":
        raise ValueError("Incomplete shard smoke: no usable throughput record")
    losses = result["losses"]
    tokens = [n for item in losses for n in item["tokens_per_sample"]]
    elapsed = sum(item["seconds"] for item in losses)
    record = {key: result[key] for key in ("git_sha", "revision", "weight_sha256", "split_name",
              "n", "seed", "device", "dtype", "timestamp_utc", "config_sha256", "shard_manifest_sha256")}
    record.update(kind="stage-1 sharded smoke throughput", source=str(run_output),
                  measured_samples_per_second=len(tokens) / elapsed,
                  peak_memory_bytes=max(item["peak_memory_bytes"] for item in losses),
                  gpu=torch.cuda.get_device_name() if str(result["device"]).startswith("cuda") else None,
                  sequence_lengths=tokens, steps_measured=len(losses), samples_measured=len(tokens),
                  measurement_seconds=elapsed, checkpoint_verification=result["checkpoint_verification"],
                  torch_version=torch.__version__, passed=True)
    bounded_write_json(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--eval-shards", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--wall-clock-limit-seconds", type=float, default=None)
    parser.add_argument("--throughput-json", type=Path, default=None,
                        help="measured T4 smoke throughput; required, never optional")
    parser.add_argument("--budget-seconds", type=float, default=None)
    args = parser.parse_args()
    run_full(args.config, args.checkpoint_dir, args.train_shards, args.output,
             eval_shards=args.eval_shards, resume_from=args.resume_from,
             wall_clock_limit_seconds=args.wall_clock_limit_seconds,
             throughput_json=args.throughput_json, budget_seconds=args.budget_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())