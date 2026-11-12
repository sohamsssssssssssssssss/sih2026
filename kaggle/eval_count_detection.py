"""CUDA-only, offline detection-count evaluation on frozen B13 validation.

This CLI never falls back to a VLM and never trains. Provision inputs separately.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grounding_dino.count import CONFIG, count_objects, load_thresholds
from models.qwen_vl.b13 import count_target

VALIDATION_SHA256 = "b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c"


def count_rows(manifest):
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VALIDATION_SHA256:
        raise ValueError("Expected the frozen 389-question B13 validation manifest")
    rows = [json.loads(line) for line in raw.splitlines()]
    selected = [row for row in rows if row["question_type"] == "count"]
    if len(rows) != 389 or len(selected) != 100 or len({r["sample_id"] for r in selected}) != 100:
        raise ValueError("Frozen validation row count/identity differs")
    for row in selected:
        if count_target(row["question"]) is None or not re.fullmatch(r"0|[1-9][0-9]*", row["answer"]):
            raise ValueError(f"Invalid count row: {row['sample_id']}")
    return selected


def metrics(records):
    if not records:
        raise ValueError("Cannot score an empty count evaluation")
    return {"n": len(records),
            "accuracy": sum(r["prediction"] == r["expected"] for r in records) / len(records),
            "mean_absolute_error": sum(abs(r["prediction"] - r["expected"]) for r in records) / len(records)}


def evaluate(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from models.artifacts import validate_artifact
    from models.grounding_dino import GroundingDINOModel

    if not torch.cuda.is_available():
        raise RuntimeError("Count evaluation requires CUDA; CPU/VLM fallback is disabled")
    if args.output.exists():
        raise FileExistsError("Choose a fresh evaluation output")
    settings = load_thresholds(args.config)
    thresholds = sorted(set(args.thresholds))
    if not thresholds or settings.box_threshold not in thresholds:
        raise ValueError("Include the configured app box threshold in the sweep")
    for threshold in thresholds:
        replace(settings, box_threshold=threshold)  # Apply the production config validation.
    rows = count_rows(args.manifest)
    image_root = args.image_root.resolve()
    images = {}
    for row in rows:
        image = (image_root / row["image"]).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError(f"Missing or unsafe count image: {row['image']}")
        if row["image"] not in images:
            with image.open("rb") as handle:
                images[row["image"]] = hashlib.file_digest(handle, "sha256").hexdigest()
    model = GroundingDINOModel(box_threshold=min(thresholds), text_threshold=settings.text_threshold)
    readiness = model.readiness()
    if not readiness.available:
        raise RuntimeError(f"Detector unavailable: {readiness.reason_code}: {readiness.detail}")
    checkpoint = validate_artifact(model.name)
    if not checkpoint.available or checkpoint.path is None:
        raise RuntimeError("Detector checkpoint unavailable")
    with checkpoint.path.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    conditions = {threshold: [] for threshold in thresholds}
    for row in rows:
        paths = [str(image_root / row["image"])]
        noun = count_target(row["question"])
        proposals = model.infer(paths, noun)
        for threshold in thresholds:
            result = count_objects(paths, noun, lambda *_: proposals,
                                   replace(settings, box_threshold=threshold))
            conditions[threshold].append({
                "sample_id": row["sample_id"], "image": row["image"], "question": row["question"],
                "object_prompt": noun, "expected": int(row["answer"]), "prediction": result["count"],
                "evidence": result["evidence"],
            })
    report = {"manifest_sha256": VALIDATION_SHA256, "device": "cuda",
              "detector": {"name": model.name, "version": model.version, "checkpoint_sha256": checkpoint_sha256},
              "image_sha256": images, "count_config": asdict(settings),
              "metrics": metrics(conditions[settings.box_threshold]),
              "threshold_sweep": {str(key): {**metrics(records), "records": records}
                                  for key, records in conditions.items()},
              "scope": "Detection counts only on frozen B13 validation; no VLM fallback or training."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True,
                        help="Box-score sweep; include the app's configured threshold")
    report = evaluate(parser.parse_args())
    print(json.dumps({"metrics": report["metrics"], "sweep": {
        key: {metric: condition[metric] for metric in ("n", "accuracy", "mean_absolute_error")}
        for key, condition in report["threshold_sweep"].items()}}, indent=2))


if __name__ == "__main__":
    main()
