"""Offline held-out evaluation of a base Qwen checkpoint and, optionally, a trained adapter.

Samples come from a subset file produced by ``python -m eval.rsvqa_research subset``;
metrics are defined in ``eval.rsvqa_research.METRICS``.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rsvqa_research import METRICS, paired_comparison, score, summarise  # noqa: E402
from training.remote_sensing import (  # noqa: E402
    BASE_MODEL_ID,
    TrainingExample,
    load_manifest,
    messages,
    sha256_file,
)


def _answer(
    model, processor, process_vision_info, example: TrainingExample, image_size: int, max_new_tokens: int
) -> str:
    import torch

    conversation = messages(example, image_size, include_target=False)
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(conversation)
    inputs = processor(
        text=[prompt], images=images, videos=videos, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def select_subset(examples: list[TrainingExample], subset: dict) -> list[tuple[TrainingExample, dict]]:
    """Resolve every subset sample in the validated manifest, in subset order."""
    by_id = {item.sample_id: item for item in examples if item.split == subset["split"]}
    missing = [sample["sample_id"] for sample in subset["samples"] if sample["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} subset samples are not in the {subset['split']} manifest: {missing[:3]}")
    return [(by_id[sample["sample_id"]], sample) for sample in subset["samples"]]


def evaluate(selected: list[tuple[TrainingExample, dict]], subset: dict, answer: Callable) -> dict:
    started = time.perf_counter()
    rows = []
    for example, sample in selected:
        sample_started = time.perf_counter()
        prediction = answer(example)
        rows.append(
            {
                "sample_id": example.sample_id,
                "image_id": sample["image_id"],
                "type": sample["type"],
                "expected": example.response,
                "prediction": prediction,
                "latency_seconds": round(time.perf_counter() - sample_started, 4),
                **score(prediction, example.response, sample["type"]),
            }
        )
    result = {
        "summary": summarise(rows, subset["full_split_type_counts"]),
        "latency_seconds": time.perf_counter() - started,
        "results": rows,
    }
    flagged = set(subset.get("leakage_flagged_image_ids", []))
    if flagged:
        result["footprint_disjoint_summary"] = summarise(
            [row for row in rows if row["image_id"] not in flagged], subset["full_split_type_counts"]
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-revision", help="verified Hugging Face commit of --model-path")
    parser.add_argument("--adapter-path", type=Path, help="omit to evaluate the base model only")
    parser.add_argument("--skip-base", action="store_true", help="evaluate only the adapter")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--expected-subset-sha256", help="refuse to run unless the subset matches")
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.image_size < 28 or args.image_size % 28:
        parser.error("image-size must be a positive multiple of 28")
    if args.skip_base and args.adapter_path is None:
        parser.error("--skip-base needs --adapter-path")
    if args.out.exists():
        parser.error(f"{args.out} exists; evaluation reports are never overwritten")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    subset_sha256 = sha256_file(args.subset)
    if args.expected_subset_sha256 and subset_sha256 != args.expected_subset_sha256:
        raise RuntimeError(f"subset sha256 {subset_sha256} != locked {args.expected_subset_sha256}")
    if not args.model_path.is_dir() or (args.adapter_path and not args.adapter_path.is_dir()):
        raise FileNotFoundError("Local base model (and adapter, if given) directories are required")
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    selected = select_subset(load_manifest(args.dataset_manifest, args.image_root), subset)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    load_seconds = time.perf_counter() - load_started

    def run(current_model) -> dict:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = evaluate(
            selected,
            subset,
            lambda example: _answer(
                current_model, processor, process_vision_info, example, args.image_size, args.max_new_tokens
            ),
        )
        result["peak_cuda_memory_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        return result

    base = None if args.skip_base else run(model)
    adapted = None
    adapter_weights = None
    if args.adapter_path:
        from peft import PeftModel

        adapter_file = args.adapter_path / "adapter_model.safetensors"
        adapter_weights = sha256_file(adapter_file) if adapter_file.is_file() else None
        adapted = run(
            PeftModel.from_pretrained(model, str(args.adapter_path), local_files_only=True).eval()
        )
    report = {
        "schema_version": 2,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": args.model_revision,
        "base_model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "adapter_weights_sha256": adapter_weights,
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "subset_path": str(args.subset.resolve()),
        "subset_sha256": subset_sha256,
        "split": subset["split"],
        "sample_count": len(selected),
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "image_size": args.image_size,
            "torch_dtype": "float16",
            "base_quantization_at_eval": "none",
            "prompt": "{question} + training.remote_sensing.ANSWER_INSTRUCTION",
        },
        "model_load_seconds": load_seconds,
        "metrics": METRICS,
        "base": base,
        "adapted": adapted,
        "paired": paired_comparison(base["results"], adapted["results"]) if base and adapted else None,
        "claim": "Measured on the listed held-out samples only; no broader accuracy claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"report": str(args.out), "samples": len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
