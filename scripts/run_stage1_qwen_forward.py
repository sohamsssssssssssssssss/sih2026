"""Run the gated eight-patch Stage-1 Qwen forward and gradient check."""

import argparse
import json
import math
import resource
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from data.bigearthnet_s2 import load_s2_patch
from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values, stage1_parameter_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt", choices=("mps_eager", "cpu_fp32"), required=True)
    args = parser.parse_args()
    diagnostic = json.loads(Path("results/stage1-loop4b-diagnostics.json").read_text())
    if not diagnostic.get("overall_pass"):
        raise ValueError("Loop 4b equivalence gates did not pass")
    selected = json.loads(Path("results/stage1-loop4-selected-captions.json").read_text())
    pre = json.loads(Path("data/manifests/bigearthnet/stage1-loop4-preregistration.json").read_text())
    rows = selected["captions"][:8]
    device = "mps" if args.attempt == "mps_eager" else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    attention = "eager" if device == "mps" else "sdpa"
    result = {
        "kind": "engineering checkpoint (pinned rev); loop 6 forward",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": "caption-geo-split.v1/train", "n": len(rows), "seed": 0,
        "device": device, "dtype": str(dtype), "attempt": args.attempt,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": diagnostic["revision"], "weight_sha256": diagnostic["weight_sha256"],
        "patch_ids": [row["patch_id"] for row in rows], "status": "started",
        "losses": [], "tokens": [],
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        torch.manual_seed(0)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map=device, attn_implementation=attention,
        )
        result["vision_attention_class"] = type(model.visual.blocks[0].attn).__name__
        result["vision_attention_requested"] = attention
        if attention == "eager" and result["vision_attention_class"] != "Qwen2_5_VLVisionAttention":
            raise ValueError("Eager vision attention configuration was not applied")
        convert_patch_embed(model)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = pre["planned_tiny_run"]
        model = get_peft_model(model, LoraConfig(
            r=lora["lora_rank"], lora_alpha=lora["lora_alpha"], lora_dropout=0,
            bias="none", task_type="CAUSAL_LM", target_modules=lora["lora_target_modules"],
        ))
        for parameter in model.get_base_model().visual.parameters():
            parameter.requires_grad_(True)
        model.enable_input_require_grads()
        groups = stage1_parameter_groups(model)
        result["parameter_groups"] = {
            group["name"]: {"lr": group["lr"], "parameters": sum(p.numel() for p in group["params"])}
            for group in groups
        }
        if set(result["parameter_groups"]) != {"vision", "patch_embed", "lm_lora"}:
            raise ValueError("Missing a required trainable parameter group")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        prompt_ids = tokenizer(selected["prompt_template"], add_special_tokens=False).input_ids
        result["prompt_tokens"] = len(prompt_ids)
        result["image_pad_tokens"] = prompt_ids.count(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        if result["image_pad_tokens"] != 16:
            raise ValueError("Expected sixteen Qwen image tokens")
        samples = []
        for row in rows:
            patch_id = row["patch_id"]
            directory = args.patch_root / patch_id.rsplit("_", 2)[0] / patch_id
            pixels, grid = pack_s2_pixel_values(load_s2_patch(
                directory, norm_contract=Path("data/manifests/bigearthnet/norm_contract.json")))
            caption_ids = tokenizer(row["output"] + "<|im_end|>", add_special_tokens=False).input_ids
            ids = prompt_ids + caption_ids
            labels = [-100] * len(prompt_ids) + caption_ids
            if len(ids) != len(labels) or labels[:len(prompt_ids)] != [-100] * len(prompt_ids):
                raise ValueError("Caption-only loss mask mismatch")
            samples.append((row, {
                "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
                "labels": torch.tensor([labels], dtype=torch.long, device=device),
                "pixel_values": pixels.to(device=device, dtype=dtype),
                "image_grid_thw": grid.to(device=device),
            }))
            result["tokens"].append({"patch_id": patch_id, "prompt": len(prompt_ids),
                                     "caption": len(caption_ids), "total": len(ids)})
        grad_index = min(range(len(samples)), key=lambda i: result["tokens"][i]["total"])
        result["gradient_patch_id"] = samples[grad_index][0]["patch_id"]
        result["decoded_sample"] = tokenizer.decode(samples[grad_index][1]["input_ids"][0].cpu(), skip_special_tokens=False)
        print(result["decoded_sample"], flush=True)
        result["status"] = "model and eight caption-masked inputs prepared"
        save()
        model.train()
        for index, (row, batch) in enumerate(samples):
            if index == grad_index:
                if device == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                loss = model(**batch).loss
                if not math.isfinite(loss.item()):
                    raise ValueError("Nonfinite forward loss")
                loss.backward()
                if device == "mps":
                    torch.mps.synchronize()
                result["gradient_step_seconds"] = time.perf_counter() - start
                result["peak_memory_bytes"] = (
                    torch.mps.driver_allocated_memory() if device == "mps"
                    else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                gradient = {}
                for group in groups:
                    gradient[group["name"]] = sum(
                        parameter.grad.detach().float().abs().sum().item()
                        for parameter in group["params"] if parameter.grad is not None)
                result["gradient_abs_sum"] = gradient
                if not all(value > 0 and math.isfinite(value) for value in gradient.values()):
                    raise ValueError("Missing or nonfinite gradient in a required group")
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    loss = model(**batch).loss
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite forward loss: {row['patch_id']}")
            result["losses"].append({"patch_id": row["patch_id"], "loss": value})
            result["status"] = f"{len(result['losses'])}/{len(rows)} forward losses finite"
            save()
        result["passed"] = len(result["losses"]) == 8 and all(
            value > 0 for value in result["gradient_abs_sum"].values())
        result["status"] = "forward gate passed" if result["passed"] else "forward gate failed"
        save()
    except Exception as exc:
        result["passed"] = False
        result["status"] = "forward gate failed; no tiny overfit"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        save()
        raise
    print(json.dumps({"passed": result["passed"], "losses": len(result["losses"]),
                      "gradient_abs_sum": result["gradient_abs_sum"]}))


if __name__ == "__main__":
    main()
