"""Run the preregistered loop-4b Qwen vision equivalence diagnostics."""

import argparse
import gc
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
)

from data.bigearthnet_s2 import load_s2_patch
from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values


def max_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (actual.double() - reference.double()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--patch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    amendment = json.loads(Path("results/stage1-loop4b-preregistration.json").read_text())
    original_path = Path("results/stage1-loop4-equivalence.json")
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != amendment["original_failed_result_sha256"]:
        raise ValueError("Original failed result changed after preregistration")
    result = {
        "kind": "engineering checkpoint (pinned rev); loop 4b diagnostics",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": amendment["split_name"], "n": 4, "seed": 0,
        "device": "CPU float32 and float64; pinned bfloat16 vision weights",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": amendment["revision"], "weight_sha256": {},
        "preregistration": "results/stage1-loop4b-preregistration.json",
        "original_failed_result_sha256": amendment["original_failed_result_sha256"],
        "input": amendment["input"], "status": "started",
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for name, expected in amendment["weight_sha256"].items():
            digest = hashlib.sha256()
            with (args.checkpoint / name).open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            result["weight_sha256"][name] = actual
            if actual != expected:
                raise ValueError(f"Weight SHA256 mismatch: {name}")
        result["status"] = "weight hashes verified"
        save()

        config = Qwen2_5_VLConfig.from_pretrained(args.checkpoint, local_files_only=True)
        config.vision_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            vision = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        vision.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        weight_map = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        visual_keys = [key for key in weight_map if key.startswith("visual.")]
        if {weight_map[key] for key in visual_keys} != {"model-00001-of-00002.safetensors"}:
            raise ValueError("Visual weights are not wholly in the verified first shard")
        with safe_open(args.checkpoint / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as source:
            weights = {key.removeprefix("visual."): source.get_tensor(key) for key in visual_keys}
        vision.load_state_dict(weights, strict=True, assign=True)
        del weights
        gc.collect()
        wrapper = nn.Module()
        wrapper.visual = vision
        wrapper.config = config
        vision.eval().float()
        result["visual_parameter_count"] = sum(parameter.numel() for parameter in vision.parameters())
        result["visual_weight_source_dtype"] = "bfloat16"
        result["status"] = "full vision tower loaded on CPU float32"
        save()

        torch.manual_seed(0)
        gray120 = torch.randn(1, 120, 120)
        gray112 = F.interpolate(gray120[None], size=(112, 112), mode="bicubic", align_corners=False)[0]
        twelve_flat, grid = pack_s2_pixel_values(gray120.repeat(12, 1, 1))
        rgb_flat = twelve_flat.reshape(64, 12, 2, 14, 14)[:, :3].reshape(64, 1176)
        old = vision.patch_embed.proj
        old_input32 = gray112[None, :, None].repeat(1, 3, 2, 1, 1)
        new_input32 = gray112[None, :, None].repeat(1, 12, 2, 1, 1)
        with torch.no_grad():
            old_proj32 = old(old_input32)
            old_tower32 = vision(rgb_flat, grid)
            new = convert_patch_embed(wrapper)
            new_proj32 = new(new_input32)
            new_tower32 = vision(twelve_flat, grid)
        result["fp32_magnitude"] = {
            "input": "seed-0 gray [1,120,120], bicubic 112, repeated across channels and temporal frames",
            "projection": {
                "max_abs_activation": old_proj32.abs().max().item(),
                "mean_abs_activation": old_proj32.abs().mean().item(),
                "max_abs_difference": max_error(new_proj32, old_proj32),
            },
            "tower": {
                "max_abs_activation": old_tower32.abs().max().item(),
                "mean_abs_activation": old_tower32.abs().mean().item(),
                "max_abs_difference": max_error(new_tower32, old_tower32),
            },
        }
        for key in ("projection", "tower"):
            item = result["fp32_magnitude"][key]
            item["relative_max_error"] = item["max_abs_difference"] / item["max_abs_activation"]
        result["status"] = "fp32 magnitude measured"
        save()

        vision.patch_embed.proj = old
        vision.patch_embed.in_channels = 3
        config.vision_config.in_channels = 3
        del new
        vision.double()
        result["status"] = "full vision tower cast to CPU float64"
        save()
        with torch.no_grad():
            old_proj64 = old(old_input32.double())
            old_tower64 = vision(rgb_flat.double(), grid)
            new64 = convert_patch_embed(wrapper)
            new_proj64 = new64(new_input32.double())
            new_tower64 = vision(twelve_flat.double(), grid)
        result["gray_fp64"] = {
            "projection_max_abs_error": max_error(new_proj64, old_proj64),
            "tower_max_abs_error": max_error(new_tower64, old_tower64),
        }
        result["status"] = "gray fp64 projection and full tower measured"
        save()

        real = []
        for patch_id in amendment["input"]["real_patch_ids"]:
            directory = args.patch_root / patch_id.rsplit("_", 2)[0] / patch_id
            tensor = load_s2_patch(directory, norm_contract=Path("data/manifests/bigearthnet/norm_contract.json"))
            flat, _ = pack_s2_pixel_values(tensor)
            x = flat.double().reshape(64, 12, 2, 14, 14)
            mean = x.mean(dim=1, keepdim=True)
            with torch.no_grad():
                reference = old(mean.repeat(1, 3, 1, 1, 1))
                actual = new64(x)
            real.append({"patch_id": patch_id, "projection_max_abs_error": max_error(actual, reference),
                         "reference_max_abs_activation": reference.abs().max().item()})
        result["real_fp64"] = real
        result["status"] = "real-patch analytic projection measured"
        save()

        result["fp32_noise_floor"] = {
            "projection": {"old32_vs_old64_max_abs_error": max_error(old_proj32, old_proj64),
                           "new32_vs_old64_max_abs_error": max_error(new_proj32, old_proj64)},
            "tower": {"old32_vs_old64_max_abs_error": max_error(old_tower32, old_tower64),
                      "new32_vs_old64_max_abs_error": max_error(new_tower32, old_tower64)},
        }
        for item in result["fp32_noise_floor"].values():
            old_error = item["old32_vs_old64_max_abs_error"]
            item["new_to_old_error_ratio"] = item["new32_vs_old64_max_abs_error"] / old_error if old_error else None
        thresholds = amendment["gates"]
        result["gates"] = {
            "gray_fp64_projection": {"measured": result["gray_fp64"]["projection_max_abs_error"],
                                     "threshold": thresholds["gray_fp64_projection_max_abs_error"]},
            "gray_fp64_tower": {"measured": result["gray_fp64"]["tower_max_abs_error"],
                                "threshold": thresholds["gray_fp64_vision_max_abs_error"]},
            "real_fp64_projection": {"measured": max(item["projection_max_abs_error"] for item in real),
                                     "threshold": thresholds["real_fp64_projection_max_abs_error_each_patch"]},
        }
        for key in ("projection", "tower"):
            item = result["fp32_noise_floor"][key]
            result["gates"][f"fp32_noise_floor_{key}"] = {
                "measured": item["new32_vs_old64_max_abs_error"],
                "threshold": thresholds["fp32_noise_floor_ratio_ceiling"] * item["old32_vs_old64_max_abs_error"],
            }
        for gate in result["gates"].values():
            gate["passed"] = gate["measured"] <= gate["threshold"]
        result["overall_pass"] = all(gate["passed"] for gate in result["gates"].values())
        result["status"] = "equivalent; original fp32 absolute threshold below measured rounding floor" if result["overall_pass"] else "diagnostic gate failed; stop before forward"
        save()
    except Exception as exc:
        result["status"] = "diagnostic error; stop before forward"
        result["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    if not result["overall_pass"]:
        raise SystemExit("Loop 4b gate failed")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
