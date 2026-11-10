"""CPU-only A1 diagnostic; candidate weights exist in memory only.

Run with: python -m scripts.diagnose_qwen_rgb_initialization --checkpoint DIR --output REPORT.json
Requires the pinned local base weights. Never downloads or loads a training pickle.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values
from scripts.stage1_runtime import verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RGB_INDICES = (3, 2, 1)  # B04, B03, B02 in the frozen twelve-band order.
ATOL = RTOL = 1e-10


def pack_frames(frames):
    """Independent slicing oracle: [T,C,H,W] -> merge-grouped [patch,C*T*14*14]."""
    if frames.ndim != 4 or frames.shape[0] != 2 or frames.shape[2:] != (112, 112):
        raise ValueError("Expected [2,C,112,112] frames")
    patches = []
    for macro_y in range(0, 8, 2):
        for macro_x in range(0, 8, 2):
            for dy in range(2):
                for dx in range(2):
                    y, x = (macro_y + dy) * 14, (macro_x + dx) * 14
                    patches.append(frames[:, :, y:y + 14, x:x + 14].permute(1, 0, 2, 3).reshape(-1))
    return torch.stack(patches)


def errors(actual, reference):
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Projection shapes differ or outputs are nonfinite")
    maximum = (actual - reference).abs().max().item()
    scale = reference.abs().max().item()
    return {"max_abs_error": maximum, "relative_max_error": maximum / scale if scale else None,
            "reference_max_abs": scale,
            "equivalent": bool(torch.allclose(actual, reference, atol=ATOL, rtol=RTOL))}


def load_projection(checkpoint):
    """Verify pinned shards before reading only the real projection tensor."""
    spec = json.loads((ROOT / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    required = [checkpoint / name for name in spec["weight_sha256"]]
    required += [checkpoint / "config.json", checkpoint / "model.safetensors.index.json"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Pinned local base weights/config/index are unavailable; no random fallback")
    proof = verify_checkpoint(checkpoint, spec["revision"], spec["weight_sha256"])
    if not proof["passed"]:
        raise ValueError("Pinned base checkpoint provenance or weight verification failed")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    names = [key for key in ("visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.weight") if key in index]
    if len(names) != 1 or index[names[0]] not in spec["weight_sha256"]:
        raise ValueError("Expected exactly one Qwen projection in a pinned shard")
    with safe_open(str(checkpoint / index[names[0]]), framework="pt", device="cpu") as source:
        weight = source.get_tensor(names[0])
    if tuple(weight.shape) != (1280, 3, 2, 14, 14) or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("Unexpected pretrained Qwen projection tensor")
    config = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    defaults = {"in_channels": 3, "temporal_patch_size": 2, "patch_size": 14, "spatial_merge_size": 2}
    if any(config.get(key, defaults.get(key)) != value for key, value in
           {"hidden_size": 1280, "in_channels": 3, "temporal_patch_size": 2,
            "patch_size": 14, "spatial_merge_size": 2}.items()):
        raise ValueError("Base vision configuration differs from the diagnostic contract")
    return weight, {"tensor_key": names[0], "source_dtype": str(weight.dtype), "checkpoint_verification": proof}


@torch.no_grad()
def diagnose(weight):
    """Exercise the actual Qwen wrapper; supplied weights determine evidence identity."""
    if weight.ndim != 5 or tuple(weight.shape[1:]) != (3, 2, 14, 14) or not torch.isfinite(weight).all():
        raise ValueError("Expected finite [out,3,2,14,14] projection weights")
    original = Qwen2_5_VisionPatchEmbed(patch_size=14, temporal_patch_size=2,
                                      in_channels=3, embed_dim=weight.shape[0]).double().cpu().eval()
    original.proj.weight.copy_(weight.detach().to(device="cpu", dtype=torch.float64))
    wrapper = nn.Module()
    wrapper.visual = nn.Module()
    wrapper.visual.patch_embed = copy.deepcopy(original)
    convert_patch_embed(wrapper)
    current = wrapper.visual.patch_embed
    candidate_wrapper = nn.Module()
    candidate_wrapper.visual = nn.Module()
    candidate_wrapper.visual.patch_embed = copy.deepcopy(original)
    convert_patch_embed(candidate_wrapper, initialization="rgb_preserving_zero9_v1")
    candidate = candidate_wrapper.visual.patch_embed
    generator = torch.Generator(device="cpu").manual_seed(0)
    cases = {}

    def compare(name, rgb_frames, s2_packed):
        rgb_packed = pack_frames(rgb_frames).double()
        # Full-image Conv3d is an independent oracle for temporal and patch order.
        full = original.proj(rgb_frames.permute(1, 0, 2, 3)[None].double())[0, :, 0]
        reference = torch.stack([full[:, my + dy, mx + dx]
                                 for my in range(0, 8, 2) for mx in range(0, 8, 2)
                                 for dy in range(2) for dx in range(2)])
        wrapped = original(rgb_packed)
        oracle = errors(wrapped, reference)
        if not oracle["equivalent"]:
            raise ValueError("Qwen wrapper disagrees with temporal/spatial packing oracle")
        if not torch.equal(s2_packed.reshape(64, 12, 2, 14, 14)[:, RGB_INDICES].reshape(64, 1176).double(), rgb_packed):
            raise ValueError("S2 RGB channel packing differs from independent RGB oracle")
        cases[name] = {"reference_wrapper_vs_full_conv3d": oracle,
                       "current": errors(current(s2_packed.double()), wrapped),
                       "candidate": errors(candidate(s2_packed.double()), wrapped)}

    rgb = torch.randn(3, 120, 120, generator=generator)
    fixtures = {"independent_rgb": rgb,
                "red_green_opponent": torch.stack([rgb[0], -rgb[0], torch.zeros_like(rgb[0])]),
                "rgb_only_grayscale": rgb[:1].repeat(3, 1, 1)}
    for channel, name in enumerate(("red", "green", "blue")):
        impulse = torch.zeros(3, 120, 120)
        impulse[channel, 35, 49] = 1
        fixtures[name + "_impulse"] = impulse
    for name, source in fixtures.items():
        image = torch.zeros(12, 120, 120)
        image[list(RGB_INDICES)] = source
        packed, grid = pack_s2_pixel_values(image)
        if grid.tolist() != [[1, 8, 8]]:
            raise ValueError("Unexpected production image grid")
        frames = F.interpolate(source[None], (112, 112), mode="bicubic", align_corners=False)[0].repeat(2, 1, 1, 1)
        compare(name, frames, packed)
    gray = rgb[0]
    packed, _ = pack_s2_pixel_values(gray[None].repeat(12, 1, 1))
    frames = F.interpolate(gray[None, None], (112, 112), mode="bicubic", align_corners=False)[0].repeat(2, 3, 1, 1)
    compare("all_12_grayscale", frames, packed)
    temporal = torch.randn(2, 3, 112, 112, generator=generator)
    for frame in (0, 1):
        frames = temporal.clone()
        frames[1 - frame].zero_()
        image = torch.zeros(2, 12, 112, 112)
        image[:, list(RGB_INDICES)] = frames
        compare(f"temporal_frame_{frame}_only", frames, pack_frames(image))
    return {"device": "cpu", "dtype": "float64", "atol": ATOL, "rtol": RTOL,
            "weight_shapes": {"original": list(original.proj.weight.shape), "converted": list(current.proj.weight.shape)},
            "packed_shapes": {"rgb": [64, 1176], "s2": [64, 4704]},
            "conv3d_input_shapes": {"rgb": [64, 3, 2, 14, 14], "s2": [64, 12, 2, 14, 14]},
            "projection_output_shape": [64, weight.shape[0]], "grid_thw": [[1, 8, 8]],
            "cases": cases, "candidate_all_equivalent": all(row["candidate"]["equivalent"] for row in cases.values()),
            "scope": "Projection only; source weight values promoted before initialization. Not a full-tower or fp16 runtime test."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.suffix != ".json" or args.output.resolve().is_relative_to(args.checkpoint.resolve()):
        parser.error("Output must be a JSON report outside the checkpoint directory")
    report = {"evidence": "pretrained weights required", "status": "BLOCKED"}
    code = 2
    try:
        weight, proof = load_projection(args.checkpoint)
        report.update(diagnose(weight), **proof, evidence="pinned pretrained projection", status="COMPLETE")
        code = 0 if report["candidate_all_equivalent"] else 1
    except FileNotFoundError as exc:
        report["error"] = str(exc)
    except (ValueError, RuntimeError) as exc:
        report.update(status="FAILED", error=str(exc))
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
