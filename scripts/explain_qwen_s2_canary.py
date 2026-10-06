"""Inspect precision boundaries in the pinned Qwen vision tower."""

import argparse
import copy
import gc
import inspect
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen2_5_VLConfig, __version__ as transformers_version
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as vision_source

from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values


def module_copy(module):
    clone = copy.copy(module)
    clone._modules = module._modules.copy()
    return clone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(Path("results/stage1-loop6-preregistration.json").read_text())
    result = {
        "kind": "loop 6 canary dtype explanation", "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": pre["split_name"], "n": 1, "seed": 0, "device": "CPU fp64",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": pre["revision"], "weight_sha256": pre["weight_sha256"],
        "transformers_version": transformers_version, "status": "started",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    handles = []
    original_rotary = vision_source.apply_rotary_pos_emb_vision
    original_sdpa = vision_source.F.scaled_dot_product_attention
    try:
        source_file = inspect.getsourcefile(vision_source.apply_rotary_pos_emb_vision)
        source = Path(source_file).read_text().splitlines()
        result["source_file"] = str(Path(source_file).relative_to(Path.cwd()))
        result["apply_rotary_pos_emb_vision_source"] = inspect.getsource(original_rotary)
        result["vision_precision_source_lines"] = [
            {"line": i, "text": line.strip()} for i, line in enumerate(source, 1)
            if 89 <= i <= 555 and (".float()" in line or ".to(" in line or "softmax(" in line)
        ]
        print(result["apply_rotary_pos_emb_vision_source"], flush=True)
        for line in result["vision_precision_source_lines"]:
            print(f"{line['line']}: {line['text']}", flush=True)
        config = Qwen2_5_VLConfig.from_pretrained(args.checkpoint, local_files_only=True)
        config.vision_config._attn_implementation = "sdpa"
        with torch.device("meta"):
            tower = vision_source.Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        tower.rotary_pos_emb = vision_source.Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        result["rotary_inv_freq_dtype_after_meta_repair"] = str(tower.rotary_pos_emb.inv_freq.dtype)
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        keys = [key for key in index if key.startswith("visual.")]
        with safe_open(args.checkpoint / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as source_weights:
            state = {key.removeprefix("visual."): source_weights.get_tensor(key) for key in keys}
        tower.load_state_dict(state, strict=True, assign=True)
        del state
        gc.collect()
        tower.eval().double()
        result["rotary_inv_freq_dtype_after_tower_double"] = str(tower.rotary_pos_emb.inv_freq.dtype)
        converted = module_copy(tower)
        converted.patch_embed = module_copy(tower.patch_embed)
        wrapper = torch.nn.Module()
        wrapper.visual = converted
        wrapper.config = copy.deepcopy(config)
        convert_patch_embed(wrapper)
        converted.eval()
        result["hook_dtypes"] = {"blocks": []}

        def record(name):
            def hook(_module, inputs, output):
                entry = {"name": name, "input": str(inputs[0].dtype), "output": str(output.dtype)}
                result["hook_dtypes"]["blocks"].append(entry)
            return hook

        handles.append(converted.patch_embed.register_forward_hook(record("patch_embed")))
        handles.append(converted.rotary_pos_emb.register_forward_hook(record("rotary_pos_emb")))
        for i, block in enumerate(converted.blocks):
            handles.append(block.attn.qkv.register_forward_hook(record(f"block_{i}.qkv")))
            handles.append(block.register_forward_hook(record(f"block_{i}")))
        handles.append(converted.merger.register_forward_hook(record("merger")))

        def rotary_probe(q, k, cos, sin):
            if "rotary_operands" not in result["hook_dtypes"]:
                q_out, k_out = original_rotary(q, k, cos, sin)
                result["hook_dtypes"]["rotary_operands"] = {
                    "q_in": str(q.dtype), "k_in": str(k.dtype),
                    "q_after_explicit_float": str(q.float().dtype),
                    "k_after_explicit_float": str(k.float().dtype),
                    "cos": str(cos.dtype), "sin": str(sin.dtype),
                    "q_after_rotary": str(q_out.dtype), "k_after_rotary": str(k_out.dtype),
                }
                return q_out, k_out
            return original_rotary(q, k, cos, sin)

        def sdpa_probe(q, k, v, *rest, **kwargs):
            if "sdpa_operands" not in result["hook_dtypes"]:
                result["hook_dtypes"]["sdpa_operands"] = {
                    "q": str(q.dtype), "k": str(k.dtype), "v": str(v.dtype),
                    "mask": str(rest[0].dtype) if rest and rest[0] is not None else None,
                    "softmax": "internal to SDPA; dtype not exposed by a forward hook",
                }
            return original_sdpa(q, k, v, *rest, **kwargs)

        torch.manual_seed(0)
        gray = torch.randn(1, 120, 120)
        pixels, grid = pack_s2_pixel_values(gray.repeat(12, 1, 1))
        pixels = pixels.double()
        vision_source.apply_rotary_pos_emb_vision = rotary_probe
        vision_source.F.scaled_dot_product_attention = sdpa_probe
        with torch.no_grad():
            full_output = converted(pixels, grid)
        result["hook_dtypes"]["tower_output"] = str(full_output.dtype)
        result["hook_dtypes"]["block_count"] = len(converted.blocks)
        vision_source.apply_rotary_pos_emb_vision = original_rotary
        vision_source.F.scaled_dot_product_attention = original_sdpa

        projection = converted.patch_embed.proj
        projection_copy = copy.deepcopy(projection)
        image = pixels.reshape(-1, 12, 2, 14, 14)
        index = (0, 0, 0, 0, 0)
        delta1, delta2 = pre["canary"]["projection_perturbations"]
        with torch.no_grad():
            baseline = projection(image)
            projection_copy.weight[index] += delta1
            output1 = projection_copy(image)
            projection_copy.weight[index] += delta2 - delta1
            output2 = projection_copy(image)
        diff1 = (output1 - baseline).abs().max().item()
        diff2 = (output2 - baseline).abs().max().item()
        ratio = diff2 / diff1 if diff1 else None
        result["projection_linearity"] = {
            "weight_index": index, "deltas": [delta1, delta2],
            "max_abs_change_1": diff1, "max_abs_change_2": diff2, "ratio": ratio,
            "projection_weight_dtype": str(projection.weight.dtype),
        }
        result["source_confirms_fp32_path"] = (
            "q, k = q.float(), k.float()" in result["apply_rotary_pos_emb_vision_source"]
            and result["hook_dtypes"]["rotary_operands"]["q_after_explicit_float"] == "torch.float32"
            and result["hook_dtypes"]["rotary_operands"]["k_after_explicit_float"] == "torch.float32"
        )
        low, high = pre["canary"]["projection_change_ratio_range"]
        result["projection_linearity_pass"] = diff1 > 0 and low <= ratio <= high
        result["confirmed"] = result["source_confirms_fp32_path"] and result["projection_linearity_pass"]
        result["verdict"] = (
            "confirmed: q/k are quantized to fp32 in every vision attention block; full-tower fp64 comparison has a fp32 precision boundary"
            if result["confirmed"] else "not confirmed; stop")
        result["limitations"] = (
            "Projection fp64 equivalence remains genuine; downstream tower gate includes fp32 q/k quantization. "
            "Equivalence relies on projection match, shared unchanged downstream weights, and prior fp32 noise-floor gate. "
            "This does not prove the fp32 casts are the sole cause of the nonlinear perturbation ratios."
        )
        result["status"] = "completed"
        save()
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        vision_source.apply_rotary_pos_emb_vision = original_rotary
        vision_source.F.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    if not result["confirmed"]:
        raise SystemExit("Canary explanation not confirmed")
    print(json.dumps({"verdict": result["verdict"], "projection_linearity": result["projection_linearity"]}))


if __name__ == "__main__":
    main()
