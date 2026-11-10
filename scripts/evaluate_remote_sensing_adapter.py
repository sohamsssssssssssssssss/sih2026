"""Offline held-out comparison of a base Qwen checkpoint and a trained adapter."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.remote_sensing import (  # noqa: E402
    BASE_MODEL_ID,
    ANSWER_INSTRUCTION,
    TrainingExample,
    load_manifest,
    messages,
    select_examples,
    sha256_file,
)


def _answer(model, processor, process_vision_info, example: TrainingExample, image_size: int, *, blind=False) -> str:
    import torch

    conversation = messages(example, image_size, include_target=False)
    if blind:
        from PIL import Image
        with Image.open(example.image_path) as source:
            control = Image.new("RGB", source.size, (127, 127, 127))
        # Retain the image content entry and identical explicit resize settings.
        conversation[0]["content"][0]["image"] = control
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(conversation)
    inputs = processor(
        text=[prompt], images=images, videos=videos, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=32)
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _evaluate(model, processor, process_vision_info, examples, image_size: int) -> dict:
    started = time.perf_counter()
    results = []
    for example in examples:
        prediction = _answer(model, processor, process_vision_info, example, image_size)
        correct = prediction.casefold().strip() == example.response.casefold().strip()
        results.append(
            {
                "sample_id": example.sample_id,
                "expected": example.response,
                "prediction": prediction,
                "exact_match": correct,
            }
        )
    return {
        "sample_count": len(results),
        "exact_match_accuracy": sum(item["exact_match"] for item in results) / len(results),
        "latency_seconds": time.perf_counter() - started,
        "results": results,
    }


def evaluate_pilot_conditions(rows, examples, model, processor, vision, attach_adapter,
                              *, image_size=392, seed=17, repetitions=2000):
    """One fixed row order and input path for all four paired conditions."""
    from eval.rgb_pilot import CONDITIONS, paired_differences, score, summarise
    if image_size < 28 or image_size % 28 or len(rows) != len(examples):
        raise ValueError("Invalid image size or row alignment")
    conditions = {}
    generation_settings = model.generation_config.to_dict() if hasattr(model, "generation_config") else None
    for stage in ("base", "adapter"):
        if stage == "adapter":
            model = attach_adapter(model)
            if generation_settings is not None and model.generation_config.to_dict() != generation_settings:
                raise ValueError("Adapter decoding configuration differs from the base")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for mode in ("real", "constant"):
            records = []
            for row, example in zip(rows, examples):
                prediction = _answer(model, processor, vision, example, image_size,
                                     blind=mode == "constant")
                records.append({"sample_id": row["sample_id"], "question_id": row["question_id"],
                    "image_id": row["image_id"], "image": row["image"],
                    "question_type": row["question_type"], "question": row["question"],
                    "expected": row["answer"], "prediction": prediction,
                    **score(prediction, row["answer"], row["question_type"])})
            conditions[f"{stage}_{mode}"] = {"summary": summarise(records), "records": records}
    assert tuple(conditions) == CONDITIONS
    return {"conditions": conditions,
            "paired_differences": paired_differences(conditions, seed=seed, repetitions=repetitions)}


def run_pilot(args):
    """Validate the frozen data and local artifacts before any model loading."""
    from eval.rgb_pilot import (PILOT_IDENTITY, SCORING_VERSION, adapter_identity,
                               validate_pilot, versions)
    from scripts.stage1_runtime import verify_checkpoint
    rows, examples, provenance = validate_pilot(args.pilot_provenance, args.dataset_manifest,
                                                args.image_root, PILOT_IDENTITY)
    spec = json.loads((ROOT / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    proof = verify_checkpoint(args.model_path, spec["revision"], spec["weight_sha256"])
    if not proof["passed"]:
        raise ValueError("Base model revision or weight verification failed")
    base_files = {n: sha256_file(args.model_path / n) for n in spec["required_files"]}
    adapter = adapter_identity(args.adapter_path)
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from models.qwen_vl.stage1 import resolve_vision_module
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path), local_files_only=True, torch_dtype=torch.float16, device_map="auto")
    if resolve_vision_module(model).patch_embed.proj.in_channels != 3:
        raise ValueError("RGB pilot requires the original three-channel projection")
    result = evaluate_pilot_conditions(rows, examples, model, processor, process_vision_info,
        lambda base: PeftModel.from_pretrained(base, str(args.adapter_path), local_files_only=True),
        image_size=args.image_size, seed=args.seed, repetitions=args.bootstrap_samples)
    # Fail rather than publish a report if selected data or adapter bytes changed.
    validate_pilot(args.pilot_provenance, args.dataset_manifest, args.image_root, PILOT_IDENTITY)
    if adapter_identity(args.adapter_path) != adapter:
        raise ValueError("Adapter files changed during evaluation")
    return {"schema_version": 2, "manifest_identity": provenance["manifest_identity"],
        "validation_manifest_sha256": sha256_file(args.dataset_manifest),
        "selected_image_sha256": provenance["selected_image_sha256"],
        "base_model_id": BASE_MODEL_ID, "base_revision": spec["revision"],
        "base_verification": proof, "base_config_processor_sha256": base_files,
        "adapter_identity": adapter, "versions": versions(),
        "scoring": {"version": SCORING_VERSION,
            "policy": "Our pilot policy: stripped case-insensitive exact match; count uses canonical nonnegative raw decimal integers (0 or [1-9][0-9]*), no bins or aliases. Malformed answers score incorrect and are reported separately."},
        "preprocessing": {"image_size": args.image_size, "resize": "same Qwen process_vision_info and processor in all conditions",
            "image_processor": processor.image_processor.to_dict(),
            "constant_control": {"mode": "RGB", "value": [127, 127, 127],
                "dimensions": "source image dimensions before identical resize", "image_token_path_retained": True}},
        "prompt": {"construction": "training.remote_sensing.messages; user image then original question with shared answer instruction",
                   "answer_instruction": ANSWER_INSTRUCTION,
                   "chat_template": processor.chat_template},
        "decoding": {"do_sample": False, "max_new_tokens": 32, "skip_special_tokens": True,
                     "clean_up_tokenization_spaces": False,
                     "generation_config": model.generation_config.to_dict()},
        "claim": "Exploratory paired pilot on the frozen validation subset; no official benchmark or geographic buffer claim.",
        **result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pilot-provenance", type=Path,
                        help="Use the frozen pilot.v1 validation rows and four controls; no subsampling")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.image_size < 28 or args.image_size % 28:
        raise ValueError("image-size must be a positive multiple of 28")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not args.model_path.is_dir() or not args.adapter_path.is_dir():
        raise FileNotFoundError("Local base model and adapter directories are required")
    if args.pilot_provenance is not None:
        if args.split != "validation" or args.image_root is None or args.bootstrap_samples < 1:
            raise ValueError("Pilot requires validation, image-root and positive bootstrap-samples")
        if args.out.exists():
            raise FileExistsError("Choose a fresh pilot output path")
        report = run_pilot(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"report": str(args.out), "samples": len(report["conditions"]["base_real"]["records"])}))
        return 0
    examples = select_examples(
        load_manifest(args.dataset_manifest, args.image_root),
        args.split,
        args.max_samples,
        args.seed,
    )
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    base = _evaluate(model, processor, process_vision_info, examples, args.image_size)
    adapted_model = PeftModel.from_pretrained(
        model, str(args.adapter_path), local_files_only=True
    ).eval()
    adapted = _evaluate(
        adapted_model, processor, process_vision_info, examples, args.image_size
    )
    report = {
        "schema_version": 1,
        "base_model_id": BASE_MODEL_ID,
        "base_model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "metric": "case-insensitive exact match",
        "base": base,
        "adapted": adapted,
        "accuracy_delta": adapted["exact_match_accuracy"] - base["exact_match_accuracy"],
        "claim": "Measured comparison on the selected held-out samples; no broader accuracy claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out), "samples": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
