"""Run the small, explicitly configured Stage-1 caption smoke/trend job.

Loop 7: the body is exposed as ``run()`` so the T4 package runner
(scripts/run_stage1_t4.py) can drive individual training runs for the smoke,
resume-continuity, and overfit stages. Command-line behavior is unchanged, with
three additions: ``--stop-after`` (early stop; the LR schedule stays pinned to
the configured step count so a resumed run is comparable), ``--resume-from``
(explicit checkpoint path), and ``--steps`` (redefine the total step count; the
overfit stage sizes N from measured throughput).

``evaluate_losses()`` measures caption-only losses at a fixed trained state
(loaded from a checkpoint) without any optimizer update; the loop-7 image-
shuffle control uses it for both arms so the comparison is pure inference.
"""

import argparse
import gc
import hashlib
import json
import math
import resource
import random
from contextlib import nullcontext

import numpy as np
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from scripts.stage1_runtime import verify_checkpoint, bounded_checkpoint_save, bounded_write_json
from data.bigearthnet_s2 import load_s2_patch
from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values, stage1_parameter_groups


def canonical_digest(obj) -> str:
    """SHA256 over a canonical serialization of optimizer/scheduler state.

    torch.save output is not byte-stable across runs (zip container), so
    continuity checks hash a deterministic walker instead: sorted mapping keys,
    tensor dtype/shape/bytes in order, scalars by repr.
    """
    digest = hashlib.sha256()

    def walk(node) -> None:
        if torch.is_tensor(node):
            payload = node.detach().cpu().contiguous()
            digest.update(f"tensor|{payload.dtype}|{tuple(payload.shape)}|".encode())
            digest.update(payload.view(torch.uint8).numpy().tobytes() if payload.dtype in (torch.bfloat16,) else payload.numpy().tobytes())
        elif isinstance(node, dict):
            for key in sorted(node.keys(), key=repr):
                digest.update(f"key:{key!r};".encode())
                walk(node[key])
        elif isinstance(node, (list, tuple)):
            digest.update(f"seq:{len(node)};".encode())
            for item in node:
                walk(item)
        else:
            digest.update(f"leaf:{node!r};".encode())

    walk(obj)
    return digest.hexdigest()


def amp_context(device, dtype):
    return (torch.autocast("cuda", dtype=torch.float16)
            if str(device).startswith("cuda") and dtype == torch.float16 else nullcontext())


def make_scaler(device, dtype):
    return torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda") and dtype == torch.float16)


def rng_state():
    return {"torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(saved):
    torch.set_rng_state(saved["torch_rng_state"].cpu())
    np.random.set_state(saved["numpy_rng_state"])
    random.setstate(saved["python_rng_state"])
    if saved["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])


def trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


def load_trainable_state(model, saved):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved) != expected:
        raise ValueError("Checkpoint trainable parameter names differ from the model")
    model.load_state_dict(saved, strict=False)


def checkpoint_identity(config_path, *, total_steps, shuffle_images=False):
    """Bind a state to its config, schedule, captions and normalization bytes."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    return canonical_digest({
        "config": config,
        "total_steps": total_steps,
        "shuffle_images": bool(shuffle_images or config.get("shuffle_images", False)),
        "captions_sha256": hashlib.sha256((config_path.parent / config["captions"]).read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256((config_path.parent / config["contract"]).read_bytes()).hexdigest(),
    })


def validate_checkpoint_identity(saved, config_path, *, total_steps=None, shuffle_images=False):
    total_steps = saved["total_steps"] if total_steps is None else total_steps
    expected = checkpoint_identity(config_path, total_steps=total_steps, shuffle_images=shuffle_images)
    if saved.get("config_identity") != expected or saved.get("total_steps") != total_steps:
        raise ValueError("Checkpoint config identity differs; retain the original variant/config and schedule")
    return expected


def restore_training_state(model, optimizer, scheduler, scaler, saved, config_path,
                           *, total_steps, shuffle_images=False):
    """Validate before mutation; reload every state at an optimizer-step boundary."""
    validate_checkpoint_identity(saved, config_path, total_steps=total_steps,
                                 shuffle_images=shuffle_images)
    config = yaml.safe_load(Path(config_path).read_text())
    if saved["data_position"]["next_micro_sample"] != saved["step"] * config["gradient_accumulation_steps"]:
        raise ValueError("Checkpoint data position differs from its optimizer step")
    load_trainable_state(model, saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved)
    return saved["step"]


def build_model(config, checkpoint, dtype, device):
    """Deterministically construct the trainable Stage-1 model from the pin.

    Shared by training runs, eval-only loss measurement, and generation so all
    three see identical construction (same seed, dtype, conversion, LoRA).
    """
    # Clear cycles from the preceding model before allocating its replacement.
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map=device,
        attn_implementation=config["attention"],
    )
    if config.get("vision_fp32", False):
        model.visual.float()
    convert_patch_embed(model)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=0,
        bias="none", task_type="CAUSAL_LM", target_modules=lora["target_modules"],
    ))
    visual = model.get_base_model().visual
    last_n = config.get("vision_trainable_last_n_blocks")
    if last_n is None:
        for parameter in visual.parameters():
            parameter.requires_grad_(True)
    else:
        if not isinstance(last_n, int) or not 1 <= last_n <= len(visual.blocks):
            raise ValueError("vision_trainable_last_n_blocks outside tower block count")
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        for parameter in visual.patch_embed.proj.parameters():
            parameter.requires_grad_(True)
        for block in visual.blocks[-last_n:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    # Frozen language weights retain checkpoint dtype; optimizer masters are fp32.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.enable_input_require_grads()
    if config.get("vision_fp32", False) and str(device).startswith("cuda"):
        # Disable autocast only for the explicit fp32 vision fallback.
        def enter_fp32(module, _inputs):
            module._stage1_amp_context = torch.autocast("cuda", enabled=False)
            module._stage1_amp_context.__enter__()

        def exit_fp32(module, _inputs, _output):
            module._stage1_amp_context.__exit__(None, None, None)

        visual = model.get_base_model().visual
        visual.register_forward_pre_hook(enter_fp32)
        visual.register_forward_hook(exit_fp32, always_call=True)
    groups = stage1_parameter_groups(model)
    for group in groups:
        group["lr"] = config["learning_rates"][group["name"]]
    if {group["name"] for group in groups} != {"vision", "patch_embed", "lm_lora"}:
        raise ValueError("Missing a trainable parameter group")
    return model, groups


def load_samples(config, captions, contract, checkpoint, patch_root, device, dtype, tokenizer):
    """Tokenize the 16 package captions and pack their S2 pixels."""
    prompt_ids = tokenizer(captions["prompt_template"], add_special_tokens=False).input_ids
    image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if prompt_ids.count(image_token) != 16:
        raise ValueError("Expected 16 image tokens")
    samples = []
    for row in captions["captions"]:
        patch_id = row["patch_id"]
        patch_dir = patch_root / patch_id.rsplit("_", 2)[0] / patch_id
        pixels, grid = pack_s2_pixel_values(load_s2_patch(patch_dir, norm_contract=contract))
        caption_ids = tokenizer(row["output"] + "<|im_end|>", add_special_tokens=False).input_ids
        ids = prompt_ids + caption_ids
        labels = [-100] * len(prompt_ids) + caption_ids
        samples.append({
            "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
            "labels": torch.tensor([labels], dtype=torch.long, device=device),
            "pixel_values": pixels.to(device=device, dtype=(torch.float32 if config.get("vision_fp32") else dtype)),
            "image_grid_thw": grid.to(device=device),
        })
    return samples, prompt_ids


def shuffle_pixel_streams(samples, seed):
    """Permute pixel streams across patches; captions stay aligned."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    images = [sample["pixel_values"].clone() for sample in samples]
    for sample, source in zip(samples, order):
        sample["pixel_values"] = images[source]
    return order


def run(
    config_path: Path,
    checkpoint: Path,
    patch_root: Path,
    output: Path,
    *,
    stop_after_step: int | None = None,
    max_steps: int | None = None,
    resume_from: Path | None = None,
    shuffle_images: bool = False,
) -> dict:
    """Execute one configured training run and return its result record.

    ``stop_after_step`` stops early without changing the LR schedule, which is
    built on ``max_steps or config["steps"]``; a resumed run therefore follows
    the identical schedule as an uninterrupted one. ``shuffle_images`` permutes
    the pixel streams across patches while captions stay aligned (the loop-7
    image-shuffle control); the order is recorded in the result.
    """
    if output.exists() and json.loads(output.read_text()).get("passed") is False:
        raise FileExistsError("Keep failed attempts; choose a new output path")
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("The current sample path requires batch_size=1")
    captions = json.loads((config_path.parent / config["captions"]).read_text())
    contract = config_path.parent / config["contract"]
    device = config["device"]
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
    rows = captions["captions"]
    total_steps = int(max_steps if max_steps is not None else config["steps"])
    steps_to_run = total_steps if stop_after_step is None else min(total_steps, int(stop_after_step))
    result = {
        "kind": config["name"], "git_sha": config["git_sha"],
        "split_name": captions["split_name"], "n": len(rows), "seed": config["seed"],
        "device": device, "dtype": str(dtype),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": config["revision"], "weight_sha256": config["weight_sha256"],
        "config": str(config_path), "steps_requested": total_steps,
        "vision_trainable_last_n_blocks": config.get("vision_trainable_last_n_blocks"),
        "steps_to_run": steps_to_run,
        "attention": config["attention"], "vision_fp32": bool(config.get("vision_fp32", False)),
        "optimizer": config["optimizer"], "losses": [], "status": "started",
    }

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        bounded_write_json(output, result)

    save()
    checkpoint_file = output.with_suffix(".pt")
    try:
        proof = verify_checkpoint(checkpoint, config["revision"], config["weight_sha256"])
        result["checkpoint_verification"] = proof
        result.update({key: proof[key] for key in ("weights_match", "revision_source", "revision_matches", "passed")})
        save()
        if not proof["passed"]:
            raise ValueError("Checkpoint revision proof or weight SHA256 mismatch")
        result["weight_sha256_verified"] = True
        identity = checkpoint_identity(config_path, total_steps=total_steps, shuffle_images=shuffle_images)
        result["config_identity"] = identity
        if resume_from is not None:
            saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        elif config["resume"] and checkpoint_file.exists():
            saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        else:
            saved = None
        if saved is not None:
            validate_checkpoint_identity(saved, config_path, total_steps=total_steps,
                                         shuffle_images=shuffle_images)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, groups = build_model(config, checkpoint, dtype, device)
        result["group_parameter_counts"] = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}

        def check_vision(_module, _inputs, output):
            if not torch.isfinite(output).all():
                raise FloatingPointError("Vision tower produced inf/NaN; use vision-fp32 variant")

        hook = model.get_base_model().visual.register_forward_hook(check_vision)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        samples, _prompt_ids = load_samples(config, captions, contract, checkpoint, patch_root, device, dtype, tokenizer)
        if config.get("shuffle_images", False) or shuffle_images:
            result["shuffle_order"] = shuffle_pixel_streams(samples, config["seed"])
        result["tokens_per_sample"] = [sample["input_ids"].numel() for sample in samples]
        result["decoded_sample"] = tokenizer.decode(samples[0]["input_ids"][0].cpu(), skip_special_tokens=False)
        optimizer = torch.optim.Adafactor(groups, foreach=False) if config["optimizer"] == "adafactor" else torch.optim.AdamW(groups, foreach=False)
        accumulation = config["gradient_accumulation_steps"]
        warmup = max(1, math.ceil(total_steps * config["warmup_fraction"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / warmup, 1.0) if step < warmup else
            0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
        )
        scaler = make_scaler(device, dtype)
        start_step = 0
        if saved is not None:
            start_step = restore_training_state(model, optimizer, scheduler, scaler, saved,
                                               config_path, total_steps=total_steps,
                                               shuffle_images=shuffle_images)
            result["resumed_from_step"] = start_step
            result["optimizer_state_digest_after_load"] = canonical_digest(optimizer.state_dict())
            del saved  # CPU checkpoint copies are not needed during forward/backward.
        model.train()
        for step in range(start_step, steps_to_run):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro in range(accumulation):
                sample = samples[(step * accumulation + micro) % len(samples)]
                with amp_context(device, dtype):
                    loss = model(**sample).loss
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite loss at step {step}, micro {micro}")
                scaler.scale(loss / accumulation).backward()
                micro_losses.append(value)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            if not math.isfinite(grad_norm.item()):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = (torch.cuda.max_memory_allocated() if device == "cuda" else
                    torch.mps.driver_allocated_memory() if device == "mps" else
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result["losses"].append({"step": step + 1, "loss": sum(micro_losses) / len(micro_losses),
                                     "seconds": elapsed, "samples_per_second": accumulation / elapsed,
                                     "peak_memory_bytes": peak, "lr": scheduler.get_last_lr()})
            result["status"] = f"completed {step + 1}/{steps_to_run} steps"
            save()
            if (step + 1) % config["checkpoint_every_steps"] == 0 or step + 1 == steps_to_run:
                optimizer.zero_grad(set_to_none=True)
                result["optimizer_state_digest"] = canonical_digest(optimizer.state_dict())
                result["checkpoint_storage"] = bounded_checkpoint_save({"model": trainable_state(model), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "step": step + 1, "total_steps": total_steps,
                            "config_identity": identity,
                            "data_position": {"next_micro_sample": (step + 1) * accumulation},
                            **rng_state()}, checkpoint_file)
        hook.remove()
        result["status"] = "completed; trend only" if config.get("trend_only") else "completed smoke"
        result["passed"] = True
        save()
    except Exception:
        result["oom"] = "out of memory" in traceback.format_exc().lower()
        if str(device).startswith("cuda") and torch.cuda.is_initialized():
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "failed"
        result["passed"] = False
        result["traceback"] = traceback.format_exc()
        save()
        raise
    return result


def evaluate_losses(
    config_path: Path,
    checkpoint: Path,
    patch_root: Path,
    output: Path,
    *,
    from_checkpoint: Path,
    steps: int = 1,
    shuffle_images: bool = False,
) -> dict:
    """Measure caption-only losses at a fixed trained state; no optimizer step.

    Loads the model exactly as a training run would (same seed/dtype/conversion/
    LoRA), applies the checkpoint's weights, and records ``steps x
    gradient_accumulation_steps`` micro-sample losses. With the default one
    step, every one of the 16 package images is measured exactly once.
    """
    if output.exists() and json.loads(output.read_text()).get("passed") is False:
        raise FileExistsError("Keep failed attempts; choose a new output path")
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("The current sample path requires batch_size=1")
    captions = json.loads((config_path.parent / config["captions"]).read_text())
    contract = config_path.parent / config["contract"]
    device = config["device"]
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
    result = {
        "kind": "stage-1 eval-only loss measurement", "git_sha": config["git_sha"],
        "split_name": captions["split_name"], "n": len(captions["captions"]), "seed": config["seed"],
        "device": device, "dtype": str(dtype),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": config["revision"], "weight_sha256": config["weight_sha256"],
        "config": str(config_path), "from_checkpoint": str(from_checkpoint),
        "steps": steps, "shuffle_images": shuffle_images,
        "losses": [], "status": "started",
    }

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        bounded_write_json(output, result)

    save()
    try:
        proof = verify_checkpoint(checkpoint, config["revision"], config["weight_sha256"])
        result["checkpoint_verification"] = proof
        result.update({key: proof[key] for key in ("weights_match", "revision_source", "revision_matches", "passed")})
        if not proof["passed"]:
            raise ValueError("Checkpoint revision proof or weight SHA256 mismatch")
        model, _groups = build_model(config, checkpoint, dtype, device)
        saved = torch.load(from_checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(saved, config_path)
        load_trainable_state(model, saved["model"])
        del saved
        model.eval()

        def check_vision(_module, _inputs, vision_output):
            if not torch.isfinite(vision_output).all():
                raise FloatingPointError("Vision tower produced inf/NaN during evaluation")

        hook = model.get_base_model().visual.register_forward_hook(check_vision)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        samples, _prompt_ids = load_samples(config, captions, contract, checkpoint, patch_root, device, dtype, tokenizer)
        if shuffle_images:
            result["shuffle_order"] = shuffle_pixel_streams(samples, config["seed"])
        accumulation = config["gradient_accumulation_steps"]
        total_micro = steps * len(samples)
        with torch.no_grad():
            for index in range(total_micro):
                sample = samples[index % len(samples)]
                started = time.perf_counter()
                with amp_context(device, dtype):
                    loss = model(**sample).loss
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite loss at micro-sample {index}")
                if device == "cuda":
                    torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() if device == "cuda" else
                        torch.mps.driver_allocated_memory() if device == "mps" else
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                result["losses"].append({
                    "step": index + 1, "loss": value, "seconds": time.perf_counter() - started,
                    "samples_per_second": 1.0 / max(time.perf_counter() - started, 1e-12),
                    "peak_memory_bytes": peak, "lr": None,
                })
                result["status"] = f"measured {index + 1}/{total_micro} micro-samples"
                save()
        hook.remove()
        result["passed"] = True
        result["status"] = "completed eval-only measurement"
        save()
    except Exception:
        result["oom"] = "out of memory" in traceback.format_exc().lower()
        if str(device).startswith("cuda") and torch.cuda.is_initialized():
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "failed"
        result["passed"] = False
        result["traceback"] = traceback.format_exc()
        save()
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=None, dest="stop_after")
    parser.add_argument("--steps", type=int, default=None, dest="steps")
    parser.add_argument("--resume-from", type=Path, default=None, dest="resume_from")
    parser.add_argument("--memory-matrix", action="store_true", help="50-step control and last4 variants; separate evidence")
    parser.add_argument("--train-shards", type=Path, default=None, help="produce full-config-compatible sharded throughput")
    args = parser.parse_args()
    if args.train_shards is not None:
        if args.memory_matrix or args.resume_from or args.stop_after:
            parser.error("sharded throughput must be a fresh separate smoke")
        from scripts.run_stage1_full import measure_shard_throughput
        measure_shard_throughput(args.config, args.checkpoint, args.train_shards, args.output,
                                 steps=args.steps if args.steps is not None else 50)
        return
    if args.patch_root is None:
        parser.error("--patch-root required for packaged-patch smoke")
    if args.memory_matrix:
        if args.steps is not None or args.stop_after or args.resume_from:
            parser.error("memory matrix uses fixed50-step configs and fresh attempts")
        from scripts.run_stage1_t4 import stage_memory_matrix
        stage_memory_matrix(args.config.parent, args.checkpoint, args.patch_root, args.output)
        return
    run(args.config, args.checkpoint, args.patch_root, args.output,
        stop_after_step=args.stop_after, max_steps=args.steps, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
