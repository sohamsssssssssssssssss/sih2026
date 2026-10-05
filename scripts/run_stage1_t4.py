"""Stage-1 T4 package runner: one Kaggle session, five evidence stages.

Stages run in a fixed order and the session stops at the first failed stage:
verify -> smoke -> resume -> overfit -> extrapolate. Each stage writes its own
JSON with full provenance (git SHA, pinned revision, split, n, seed, device,
dtype, timestamp). No threshold is evaluated here that is not pre-registered in
results/stage1-loop7-preregistration.json in the source repository.

Heavy work is delegated to scripts.run_stage1_smoke.run(); this module stages
the runs, applies the pre-registered gates, and owns the resume-continuity
comparison and the greedy-decode reproduction check. Each problem gets at most
two attempts; failed attempts stay as separate files.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


def _load_sibling(name: str):
    """Load a sibling module by file path.

    The package is run with src/ on sys.path (see run.py), but loading by path
    keeps this deterministic even when a different ``scripts`` package is
    already importable (the source repository has one too).
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"stage1_t4_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke_module = _load_sibling("run_stage1_smoke")
run = _smoke_module.run
evaluate_losses = _smoke_module.evaluate_losses
build_model = _smoke_module.build_model

LOSS_CONTINUITY_THRESHOLD = 0.05
FINAL_LOSS_RATIO_THRESHOLD = 0.2
EARLY_SPIKE_RATIO_THRESHOLD = 2.0
SHUFFLE_RATIO_THRESHOLD = 1.10
DECODE_REPRODUCTION_MIN = 12
OVERFIT_BUDGET_SECONDS = 20 * 60
TRAIN_CAPTION_COUNT = 463932


def stage_record(kind, config_path, extra):
    config = yaml.safe_load(config_path.read_text())
    captions = json.loads((config_path.parent / config["captions"]).read_text())
    record = {
        "kind": kind,
        "git_sha": config["git_sha"],
        "revision": config["revision"],
        "split_name": captions["split_name"],
        "n": len(captions["captions"]),
        "seed": config["seed"],
        "device": config["device"],
        "dtype": config["precision"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weight_sha256": config["weight_sha256"],
        "vision_fp32": bool(config.get("vision_fp32", False)),
        "vision_trainable_last_n_blocks": config.get("vision_trainable_last_n_blocks"),
    }
    record.update(extra)
    return record


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _smoke_module.bounded_write_json(path, payload)


def step_loss(losses, step):
    entry = next(item for item in losses if item["step"] == step)
    return float(entry["loss"])


def mean_step_seconds(losses):
    return sum(float(entry["seconds"]) for entry in losses) / len(losses)


def stage_verify(checkpoint, revision_json, package_root, out_dir):
    """Abort on any revision or weight-hash mismatch before any stage runs."""
    if (out_dir / "stage-verify.json").exists():
        raise FileExistsError("Keep prior verification attempt; choose a new output directory")
    expected = json.loads(revision_json.read_text())
    proof = _smoke_module.verify_checkpoint(checkpoint, expected["revision"], expected["weight_sha256"])
    result = stage_record("stage-verify", package_root / "config.yaml", {
        "checkpoint": str(checkpoint), "expected_revision": expected["revision"],
        "expected_weight_sha256": expected["weight_sha256"], **proof,
    })
    result["status"] = "verified" if result["passed"] else "mismatch; abort before any training stage"
    write_json(out_dir / "stage-verify.json", result)
    return result


def stage_smoke(package_root, checkpoint, patch_root, out_dir, *, config_names=None):
    """50 optimizer steps; pre-registered gates: finite losses, finite vision.

    Attempt 1 uses config.yaml (fp16 vision). If the vision tower emits inf/NaN
    (FloatingPointError from the forward hook) or anything else fails, attempt 2
    reruns with config-vision-fp32.yaml. The config that passes is recorded and
    reused by every later stage so all stages see the same vision dtype.
    """
    attempts = []
    for attempt, config_name in enumerate(config_names or ("config.yaml", "config-vision-fp32.yaml"), start=1):
        import gc
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        config_path = package_root / config_name
        output = out_dir / f"stage-smoke-attempt{attempt}.json"
        if output.exists():
            raise FileExistsError("Keep prior attempts; choose a new output directory")
        record = stage_record(f"stage-smoke-attempt{attempt}", config_path,
                              {"attempt": attempt, "config": config_name})
        try:
            run(config_path, checkpoint, patch_root, output)
            run_result = json.loads(output.read_text())
            record["vision_dtype_used"] = "float32" if yaml.safe_load(config_path.read_text()).get("vision_fp32") else "float16"
            record["samples_per_second"] = sum(e["samples_per_second"] for e in run_result["losses"]) / len(run_result["losses"])
            record["peak_memory_bytes"] = max(e["peak_memory_bytes"] for e in run_result["losses"])
            record["tokens_per_sample"] = run_result["tokens_per_sample"]
            record["first_loss"] = run_result["losses"][0]["loss"]
            record["last_loss"] = run_result["losses"][-1]["loss"]
            record["attempts"] = attempts
            record["passed"] = True
            record["status"] = "completed"
            write_json(out_dir / "stage-smoke.json", record)
            return record
        except Exception as error:  # noqa: BLE001 - recorded, then retried once
            attempts.append({"attempt": attempt, "config": config_name, "error": repr(error)})
            record["attempts"] = attempts
            record["passed"] = False
            record["status"] = f"attempt {attempt} failed"
            record["traceback"] = getattr(error, "stage_traceback", traceback.format_exc())
            failed = json.loads(output.read_text()) if output.exists() else {}
            record["oom"] = failed.get("oom", "out of memory" in record["traceback"].lower())
            record["peak_memory_bytes"] = failed.get("peak_memory_bytes")
            write_json(out_dir / f"stage-smoke-attempt{attempt}-failure.json", record)
            write_json(out_dir / "stage-smoke.json", record)
            import gc
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
    return record


def stage_memory_matrix(package_root, checkpoint, patch_root, out_dir):
    """Separate control/variant evidence, up to two attempts EACH; no quality claim."""
    records = {}
    for name, configs in (("control", ("config.yaml", "config-vision-fp32.yaml")),
                          ("last4", ("config-last4.yaml", "config-last4-vision-fp32.yaml"))):
        records[name] = stage_smoke(package_root, checkpoint, patch_root, out_dir / name,
                                   config_names=configs)
    record = stage_record("memory-matrix", package_root / "config.yaml",
                          {"variants": records, "passed": all(r["passed"] for r in records.values()),
                           "limitation": "finite/peak/OOM comparison only; not tiny-overfit gate"})
    write_json(out_dir / "memory-matrix.json", record)
    return record


def _stage_resume_once(package_root, config_name, checkpoint, patch_root, out_dir, attempt):
    config_path = package_root / config_name
    total = yaml.safe_load(config_path.read_text())["steps"]
    cut = total // 2
    suffix = "" if attempt == 1 else f"-attempt{attempt}"
    run(config_path, checkpoint, patch_root, out_dir / f"stage-resume-interrupted{suffix}.json",
        stop_after_step=cut)
    prefix = json.loads((out_dir / f"stage-resume-interrupted{suffix}.json").read_text())
    ckpt = (out_dir / f"stage-resume-interrupted{suffix}.json").with_suffix(".pt")

    run(config_path, checkpoint, patch_root, out_dir / f"stage-resume-resumed{suffix}.json",
        resume_from=ckpt)
    resumed = json.loads((out_dir / f"stage-resume-resumed{suffix}.json").read_text())

    run(config_path, checkpoint, patch_root, out_dir / f"stage-resume-uninterrupted{suffix}.json",
        stop_after_step=total)
    uninterrupted = json.loads((out_dir / f"stage-resume-uninterrupted{suffix}.json").read_text())

    max_abs_diff = max(
        abs(step_loss(resumed["losses"], s) - step_loss(uninterrupted["losses"], s))
        for s in range(cut + 1, total + 1)
    )
    resumed_by_step = {entry["step"]: entry["lr"] for entry in resumed["losses"]}
    uninterrupted_by_step = {entry["step"]: entry["lr"] for entry in uninterrupted["losses"]}
    lr_mismatches = [
        {"step": s, "resumed": resumed_by_step[s], "uninterrupted": uninterrupted_by_step[s]}
        for s in sorted(set(resumed_by_step) & set(uninterrupted_by_step))
        if resumed_by_step[s] != uninterrupted_by_step[s]
    ]
    result = stage_record("stage-resume", config_path, {
        "attempt": attempt, "config": config_name, "cut_step": cut, "total_steps": total,
        "loss_continuity": {
            "steps_compared": list(range(cut + 1, total + 1)),
            "max_abs_loss_diff": max_abs_diff,
            "threshold_max": LOSS_CONTINUITY_THRESHOLD,
            "passed": max_abs_diff <= LOSS_CONTINUITY_THRESHOLD,
        },
        "lr_schedule_continuity": {
            "mismatches": lr_mismatches,
            "passed": not lr_mismatches,
        },
        "optimizer_state_continuity": {
            "digest_resumed_after_load": resumed.get("optimizer_state_digest_after_load"),
            "digest_uninterrupted_final": uninterrupted.get("optimizer_state_digest"),
            "note": ("digests are canonical SHA256 over the optimizer state; the resumed "
                     "digest is taken immediately after load, before its own updates"),
        },
        "runs_all_passed": bool(prefix["passed"] and resumed["passed"] and uninterrupted["passed"]),
    })
    result["passed"] = (
        result["loss_continuity"]["passed"]
        and result["lr_schedule_continuity"]["passed"]
        and result["runs_all_passed"]
    )
    result["status"] = "completed" if result["passed"] else "continuity gate failed"
    write_json(out_dir / f"stage-resume{suffix}.json", result)
    if result["passed"]:
        write_json(out_dir / "stage-resume.json", result)
    return result


def stage_resume(package_root, config_name, checkpoint, patch_root, out_dir):
    """Checkpoint at half the schedule, reload, finish; compare to uninterrupted."""
    result = None
    for attempt in (1, 2):
        try:
            result = _stage_resume_once(package_root, config_name, checkpoint, patch_root, out_dir, attempt)
        except Exception as error:
            # Preserve the underlying run's evidence and stop on execution failure.
            result = stage_record("stage-resume", package_root / config_name, {
                "attempt": attempt, "config": config_name, "passed": False,
                "status": "resume execution failed", "error": repr(error),
                "traceback": traceback.format_exc(),
            })
            write_json(out_dir / f"stage-resume-attempt{attempt}-failure.json", result)
            return result
        if result["passed"]:
            return result
    return result


def _greedy_decode(package_root, config_name, checkpoint, patch_root, captions, ckpt, out_dir):
    """Greedy-decode all 16 captions from the trained checkpoint.

    Keep the trained variant/dtype identity. A nonfinite decode is a failed
    gate; changing configuration would no longer evaluate the trained variant.
    """
    from data.bigearthnet_s2 import load_s2_patch
    from models.qwen_vl.stage1 import pack_s2_pixel_values
    from transformers import AutoTokenizer

    attempts = []
    for attempt, gen_config_name in enumerate((config_name,), start=1):
        config_path = package_root / gen_config_name
        config = yaml.safe_load(config_path.read_text())
        dtype = {"fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
        device = config["device"]
        try:
            model, _groups = build_model(config, checkpoint, dtype, device)
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            _smoke_module.validate_checkpoint_identity(saved, config_path)
            _smoke_module.load_trainable_state(model, saved["model"])
            del saved
            model.config.use_cache = True
            model.eval()

            def check_vision(_module, _inputs, output):
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Vision tower produced inf/NaN during generation")

            hook = model.get_base_model().visual.register_forward_hook(check_vision)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            prompt_ids = tokenizer(captions["prompt_template"], add_special_tokens=False).input_ids
            contract = package_root / "norm_contract.json"
            reproduced = 0
            outputs = []
            with torch.inference_mode():
                for row in captions["captions"]:
                    patch_id = row["patch_id"]
                    patch_dir = patch_root / patch_id.rsplit("_", 2)[0] / patch_id
                    pixels, grid = pack_s2_pixel_values(load_s2_patch(patch_dir, norm_contract=contract))
                    inputs = {
                        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
                        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
                        "pixel_values": pixels.to(device=device, dtype=(torch.float32 if config.get("vision_fp32") else dtype)),
                        "image_grid_thw": grid.to(device=device),
                    }
                    with _smoke_module.amp_context(device, dtype):
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=200)
                    text = tokenizer.decode(generated[0, len(prompt_ids):], skip_special_tokens=True)
                    match = " ".join(text.lower().split()) == " ".join(row["output"].lower().split())
                    reproduced += int(match)
                    outputs.append({"patch_id": patch_id, "match": match, "generated_head": text[:120]})
            hook.remove()
            return {"reproduced": reproduced, "threshold_min": DECODE_REPRODUCTION_MIN,
                    "config_used": gen_config_name, "attempt": attempt,
                    "outputs_head": outputs, "attempts": attempts}
        except FloatingPointError as error:
            attempts.append({"attempt": attempt, "config": gen_config_name, "error": repr(error)})
    return {"reproduced": 0, "threshold_min": DECODE_REPRODUCTION_MIN,
            "attempts": attempts, "failed": True,
            "error": "vision output nonfinite in the trained configuration"}


def _stage_overfit_once(package_root, config_name, checkpoint, patch_root, out_dir, steps, attempt):
    config_path = package_root / config_name
    suffix = "" if attempt == 1 else f"-attempt{attempt}"
    train_output = out_dir / f"stage-overfit-train{suffix}.json"
    run(config_path, checkpoint, patch_root, train_output, max_steps=steps)
    train = json.loads(train_output.read_text())
    ckpt = train_output.with_suffix(".pt")
    losses = [entry["loss"] for entry in train["losses"]]
    first_loss, final_loss = losses[0], losses[-1]

    # Shuffle control, both arms at the trained state with identical decoding:
    # eval-only measurement (no optimizer updates) of every one of the 16
    # images exactly once. The true arm keeps image-caption pairing; the
    # shuffle arm permutes pixel streams across patches.
    true_eval_output = out_dir / f"stage-overfit-true-eval{suffix}.json"
    evaluate_losses(config_path, checkpoint, patch_root, true_eval_output, from_checkpoint=ckpt)
    true_eval = json.loads(true_eval_output.read_text())
    shuffle_eval_output = out_dir / f"stage-overfit-shuffle-eval{suffix}.json"
    evaluate_losses(config_path, checkpoint, patch_root, shuffle_eval_output,
                    from_checkpoint=ckpt, shuffle_images=True)
    shuffle_eval = json.loads(shuffle_eval_output.read_text())

    true_losses = [entry["loss"] for entry in true_eval["losses"]]
    shuffle_losses = [entry["loss"] for entry in shuffle_eval["losses"]]
    mean_true = sum(true_losses) / len(true_losses)
    mean_shuffle = sum(shuffle_losses) / len(shuffle_losses)

    captions = json.loads((package_root / "captions.json").read_text())
    generation = _greedy_decode(package_root, config_name, checkpoint, patch_root,
                                captions, ckpt, out_dir)

    gates = {
        "final_loss_ratio": {
            "measured": final_loss / first_loss,
            "threshold_max": FINAL_LOSS_RATIO_THRESHOLD,
            "passed": final_loss <= FINAL_LOSS_RATIO_THRESHOLD * first_loss,
        },
        "no_nan": {"passed": bool(train["passed"]) and all(math.isfinite(v) for v in losses)},
        "no_early_spike": {
            "measured_max_ratio_first50": max(losses[:50]) / first_loss,
            "threshold_max": EARLY_SPIKE_RATIO_THRESHOLD,
            "passed": max(losses[:50]) <= EARLY_SPIKE_RATIO_THRESHOLD * first_loss,
        },
        "greedy_decode_reproduction": {
            "measured": generation["reproduced"],
            "threshold_min": DECODE_REPRODUCTION_MIN,
            "of": 16,
            "passed": generation["reproduced"] >= DECODE_REPRODUCTION_MIN,
        },
        "image_shuffle_control": {
            "mean_true_image_loss": mean_true,
            "mean_shuffled_image_loss": mean_shuffle,
            "measured_ratio": mean_shuffle / mean_true,
            "threshold_min": SHUFFLE_RATIO_THRESHOLD,
            "passed": mean_shuffle > SHUFFLE_RATIO_THRESHOLD * mean_true,
            "shuffle_order": shuffle_eval.get("shuffle_order"),
        },
    }
    result = stage_record("stage-overfit", config_path, {
        "attempt": attempt, "config": config_name,
        "sizing": {"steps": steps,
                   "rule": "floor(budget_seconds / smoke_seconds_per_optimizer_step); recorded, not tuned"},
        "first_loss": first_loss, "final_loss": final_loss,
        "losses": train["losses"],
        "greedy_decode": generation,
        "gates": gates,
    })
    result["passed"] = all(gate["passed"] for gate in gates.values())
    result["status"] = "completed" if result["passed"] else "one or more overfit gates failed"
    write_json(out_dir / f"stage-overfit{suffix}.json", result)
    if result["passed"]:
        write_json(out_dir / "stage-overfit.json", result)
    return result


def stage_overfit(package_root, config_name, checkpoint, patch_root, out_dir, smoke):
    """N steps sized from measured smoke throughput to a 20-minute budget."""
    smoke_losses = json.loads((out_dir / f"stage-smoke-attempt{smoke['attempt']}.json").read_text())["losses"]
    smoke_seconds = mean_step_seconds(smoke_losses)
    steps = max(1, int(OVERFIT_BUDGET_SECONDS // smoke_seconds))
    result = None
    for attempt in (1, 2):
        result = _stage_overfit_once(package_root, config_name, checkpoint, patch_root,
                                     out_dir, steps, attempt)
        if result["passed"]:
            return result
    return result


def stage_extrapolate(package_root, config_name, out_dir, smoke):
    """Epoch-hours for all 463,932 captions from measured smoke throughput."""
    epoch_hours = TRAIN_CAPTION_COUNT / smoke["samples_per_second"] / 3600
    result = stage_record("stage-extrapolate", package_root / config_name, {
        "config": config_name,
        "train_caption_count": TRAIN_CAPTION_COUNT,
        "measured_samples_per_second": smoke["samples_per_second"],
        "measured_from": "smoke stage; gradient_accumulation_steps included in the rate",
        "epoch_hours_1x_t4": epoch_hours,
        "epoch_hours_2x_t4_ddp": None,
        "ddp_note": "no measured DDP run exists; a 2xT4 DDP figure is NOT extrapolated",
        "label": "extrapolation from a 50-step measurement; not a measurement",
        "passed": True,
        "status": "recorded",
    })
    write_json(out_dir / "stage-extrapolate.json", result)
    return result


def _fail(out_dir, stages, stage_result):
    write_json(out_dir / "session-summary.json", {
        "stages_attempted": stages, "all_passed": False, "last_stage": stage_result,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="directory with the pinned Qwen weights (HF download or Kaggle dataset)")
    parser.add_argument("--patch-root", type=Path, required=True,
                        help="extracted patches.tar.gz root containing BigEarthNet-S2/")
    parser.add_argument("--out-dir", type=Path, default=Path("stage-results"))
    parser.add_argument("--package-root", type=Path, default=None,
                        help="package directory holding config.yaml/captions.json/revision.json "
                             "(defaults to the directory two levels above this file)")
    parser.add_argument("--mode", choices=("all", "verify", "resume"), default="all")
    parser.add_argument("--config", choices=("config.yaml", "config-vision-fp32.yaml",
                                            "config-last4.yaml", "config-last4-vision-fp32.yaml"),
                        default=None, help="explicit evidence variant; default retains control/fp32 fallback")
    args = parser.parse_args()

    package_root = args.package_root or Path(__file__).resolve().parents[2]
    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Keep prior evidence; choose a fresh output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    verify = stage_verify(args.checkpoint, package_root / "revision.json", package_root, out_dir)
    if not verify["passed"]:
        _fail(out_dir, ["verify"], verify)
    if args.mode == "verify":
        write_json(out_dir / "session-summary.json", {
            "stages_attempted": ["verify"], "all_passed": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        return

    completed = ["verify"]
    if args.mode == "resume":
        config_name = args.config or "config.yaml"
        resume = stage_resume(package_root, config_name, args.checkpoint, args.patch_root, out_dir)
        if not resume["passed"]:
            _fail(out_dir, completed + ["resume"], resume)
        write_json(out_dir / "session-summary.json", {
            "stages_attempted": ["verify", "resume"], "all_passed": True,
            "config": config_name, "scope": "resume continuity only; overfit not tested",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        return
    smoke = stage_smoke(package_root, args.checkpoint, args.patch_root, out_dir,
                        config_names=(args.config,) if args.config else None)
    if not smoke["passed"]:
        _fail(out_dir, completed + ["smoke"], smoke)
    completed.append("smoke")
    config_name = smoke["config"]

    resume = stage_resume(package_root, config_name, args.checkpoint, args.patch_root, out_dir)
    if not resume["passed"]:
        _fail(out_dir, completed + ["resume"], resume)
    completed.append("resume")

    overfit = stage_overfit(package_root, config_name, args.checkpoint, args.patch_root, out_dir, smoke)
    if not overfit["passed"]:
        _fail(out_dir, completed + ["overfit"], overfit)
    completed.append("overfit")

    extrapolate = stage_extrapolate(package_root, config_name, out_dir, smoke)
    completed.append("extrapolate")
    write_json(out_dir / "session-summary.json", {
        "stages_attempted": completed,
        "all_passed": bool(verify["passed"] and smoke["passed"] and resume["passed"]
                           and overfit["passed"] and extrapolate["passed"]),
        "smoke_config_used": config_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
