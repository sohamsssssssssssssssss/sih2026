"""Check that the fp64 full-tower equivalence comparison is sensitive."""

import argparse
import copy
import gc
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
)

from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values


def module_copy(module):
    clone = copy.copy(module)
    clone._modules = module._modules.copy()
    return clone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    pre = json.loads(Path("results/stage1-loop5-preregistration.json").read_text())
    amendment = json.loads(args.amendment.read_text()) if args.amendment else None
    result = {
        "kind": "engineering checkpoint (pinned rev); loop 5 canary",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": pre["split_name"], "n": 1, "seed": 0,
        "device": "CPU float64", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": pre["revision"], "weight_sha256": pre["weight_sha256"],
        "preregistration": "results/stage1-loop5-preregistration.json", "status": "started",
    }
    if amendment:
        result["amendment"] = str(args.amendment)
        result["attempt"] = amendment["attempt"]

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        config = Qwen2_5_VLConfig.from_pretrained(args.checkpoint, local_files_only=True)
        config.vision_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            original = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        original.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        keys = [key for key in index if key.startswith("visual.")]
        if {index[key] for key in keys} != {"model-00001-of-00002.safetensors"}:
            raise ValueError("Visual weights are not in the pinned first shard")
        with safe_open(args.checkpoint / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as source:
            state = {key.removeprefix("visual."): source.get_tensor(key) for key in keys}
        original.load_state_dict(state, strict=True, assign=True)
        del state
        gc.collect()
        original.eval().double()
        converted = module_copy(original)
        converted.patch_embed = module_copy(original.patch_embed)
        wrapper = torch.nn.Module()
        wrapper.visual = converted
        wrapper.config = copy.deepcopy(config)
        convert_patch_embed(wrapper)
        converted.eval()
        result["distinct_tower_objects"] = original is not converted
        result["distinct_patch_embed_objects"] = original.patch_embed is not converted.patch_embed
        result["distinct_patch_weight_storage"] = (
            original.patch_embed.proj.weight.data_ptr() != converted.patch_embed.proj.weight.data_ptr())
        result["shared_other_weight_storage"] = (
            original.blocks[0].attn.qkv.weight.data_ptr() == converted.blocks[0].attn.qkv.weight.data_ptr())
        if not all(result[key] for key in ("distinct_tower_objects", "distinct_patch_embed_objects", "distinct_patch_weight_storage")):
            raise ValueError("Canary towers or patch weights alias")
        result["status"] = "distinct towers and patch storage verified"
        save()

        torch.manual_seed(0)
        gray = torch.randn(1, 120, 120)
        twelve, grid = pack_s2_pixel_values(gray.repeat(12, 1, 1))
        twelve = twelve.double()
        rgb = twelve.reshape(64, 12, 2, 14, 14)[:, :3].reshape(64, 1176)
        with torch.no_grad():
            old_output = original(rgb, grid)
            new_output = converted(twelve, grid)
        result["distinct_output_storage"] = old_output.data_ptr() != new_output.data_ptr()
        result["tower_fp64_max_abs_error"] = (new_output - old_output).abs().max().item()
        result["tower_reference_max_abs_activation"] = old_output.abs().max().item()
        result["status"] = "distinct full-tower outputs measured"
        save()

        perturbed = module_copy(converted)
        perturbed.patch_embed = module_copy(converted.patch_embed)
        perturbed.patch_embed.proj = copy.deepcopy(converted.patch_embed.proj)
        index = tuple(pre["canary"]["perturb_weight_index"])
        delta1, delta2 = pre["canary"]["perturbations"]
        with torch.no_grad():
            perturbed.patch_embed.proj.weight[index] += delta1
            output1 = perturbed(twelve, grid)
            perturbed.patch_embed.proj.weight[index] += delta2 - delta1
            output2 = perturbed(twelve, grid)
        diff1 = (output1 - new_output).abs().max().item()
        diff2 = (output2 - new_output).abs().max().item()
        l2_1 = torch.linalg.vector_norm(output1 - new_output).item()
        l2_2 = torch.linalg.vector_norm(output2 - new_output).item()
        result["sensitivity"] = {
            "weight_index": list(index), "delta1": delta1, "delta2": delta2,
            "tower_max_abs_change_delta1": diff1, "tower_max_abs_change_delta2": diff2,
            "change_ratio": diff2 / diff1 if diff1 else None,
            "tower_l2_change_delta1": l2_1, "tower_l2_change_delta2": l2_2,
            "l2_change_ratio": l2_2 / l2_1 if l2_1 else None,
        }
        gate = pre["canary"]
        result["gates"] = {
            "distinct_towers_and_patch_storage": result["distinct_tower_objects"] and result["distinct_patch_weight_storage"] and result["distinct_output_storage"],
            "fp64_tower_equivalence": result["tower_fp64_max_abs_error"] <= gate["fp64_tower_equivalence_max_abs_error"],
            "sensitivity_nonzero": diff1 > gate["sensitivity_first_diff_gt"],
            "sensitivity_scales": diff1 > 0 and gate["sensitivity_scale_ratio_min"] <= diff2 / diff1 <= gate["sensitivity_scale_ratio_max"],
        }
        if amendment:
            result["amended_gate"] = {
                "sensitivity_l2_nonzero": l2_1 > 0,
                "sensitivity_l2_scales": l2_1 > 0 and amendment["new_l2_ratio_gate"][0] <= l2_2 / l2_1 <= amendment["new_l2_ratio_gate"][1],
            }
            result["passed"] = (all(value for key, value in result["gates"].items() if key != "sensitivity_scales")
                                and all(result["amended_gate"].values()))
        else:
            result["passed"] = all(result["gates"].values())
        result["status"] = "canary passed" if result["passed"] else "canary failed; stop"
        save()
    except Exception as exc:
        result["passed"] = False
        result["status"] = "canary error; stop"
        result["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    if not result["passed"]:
        raise SystemExit("Canary failed")
    print(json.dumps({"gates": result["gates"], "tower_fp64_max_abs_error": result["tower_fp64_max_abs_error"],
                      "sensitivity": result["sensitivity"]}))


if __name__ == "__main__":
    main()
