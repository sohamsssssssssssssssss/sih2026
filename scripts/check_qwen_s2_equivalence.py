"""Verify pinned Qwen weights and the fp32 3-to-12-channel vision equivalence gate."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact = json.loads(Path("configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    result = {
        "kind": "engineering checkpoint (pinned rev)",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": "caption-geo-split.v1/train", "n": 1, "seed": 0,
        "device": "MPS checkpoint; CPU fp32 vision equivalence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": artifact["revision"], "weight_sha256": {},
        "atol": 1e-5, "rtol": 0,
    }
    try:
        for name, expected in artifact["weight_sha256"].items():
            digest = hashlib.sha256()
            with (args.checkpoint / name).open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            result["weight_sha256"][name] = digest.hexdigest()
            if digest.hexdigest() != expected:
                raise ValueError(f"Weight SHA256 mismatch: {name}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa",
        )
        model.eval()
        model.visual.to(device="cpu", dtype=torch.float32)
        old = model.visual.patch_embed.proj
        torch.manual_seed(0)
        gray120 = torch.randn(1, 120, 120)
        gray112 = F.interpolate(gray120[None], size=(112, 112), mode="bicubic", align_corners=False)[0]
        twelve_flat, grid = pack_s2_pixel_values(gray120.repeat(12, 1, 1))
        rgb_flat = twelve_flat.reshape(64, 12, 2, 14, 14)[:, :3].reshape(64, 1176)
        with torch.no_grad():
            old_projection = old(gray112[None, :, None].repeat(1, 3, 2, 1, 1))
            old_vision = model.visual(rgb_flat, grid)
            new = convert_patch_embed(model)
            expected_weight = old.weight.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1, 1) * (3.0 / 12.0)
            new_projection = new(gray112[None, :, None].repeat(1, 12, 2, 1, 1))
            new_vision = model.visual(twelve_flat, grid)
        projection_max_abs = (new_projection - old_projection).abs().max().item()
        vision_max_abs = (new_vision - old_vision).abs().max().item()
        result.update({
            "old_projection_shape": list(old.weight.shape),
            "new_projection_shape": list(new.weight.shape),
            "initialization_exact": torch.equal(new.weight, expected_weight),
            "projection_max_abs_error": projection_max_abs,
            "vision_max_abs_error": vision_max_abs,
            "projection_output_shape": list(new_projection.shape),
            "vision_output_shape": list(new_vision.shape),
        })
        result["passed"] = result["initialization_exact"] and projection_max_abs <= result["atol"] and vision_max_abs <= result["atol"]
    except Exception as exc:
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("Hard equivalence gate failed")
    print(json.dumps({key: result[key] for key in ("initialization_exact", "projection_max_abs_error", "vision_max_abs_error", "passed")}))


if __name__ == "__main__":
    main()
