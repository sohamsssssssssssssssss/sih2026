"""Offline, stratified B13 app-path parity; never provisions model weights."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def frozen_questions(reference):
    if reference["manifest_identity"] != "1eb622b1ca4bec1d2d849f049623f015f3e015c423ce8fab146e91e5b86f43ea":
        raise ValueError("Expected frozen B13 validation identity")
    rows = reference["conditions"]["adapter_real"]["records"]
    selected = []
    for kind in ("comp", "count", "presence", "rural_urban"):
        group = [row for row in rows if row["question_type"] == kind][:10]
        if len(group) != 10:
            raise ValueError(f"Need ten frozen questions of type {kind}")
        selected.extend(group)
    if len({row["sample_id"] for row in selected}) != 40:
        raise ValueError("Duplicate selected sample IDs")
    return selected


def summarise(records):
    summary = {}
    for kind in ("comp", "count", "presence", "rural_urban"):
        group = [row for row in records if row["question_type"] == kind]
        completed = [row for row in group if "prediction" in row]
        summary[kind] = {
            "attempted": len(group), "completed": len(completed),
            "agreement": sum(row["agrees"] for row in completed) / len(completed) if completed else None,
            "accuracy": sum(row["correct"] for row in completed) / len(completed) if completed else None,
            "kaggle_accuracy": sum(row["kaggle_correct"] for row in group) / len(group),
            "seconds_per_question": sum(row["seconds"] for row in completed) / len(completed) if completed else None,
        }
    return summary


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      SATQUERY_VQA_ADAPTER=str(args.adapter.resolve()),
                      SATQUERY_ALLOW_MPS="0" if args.cpu else "1")
    from models.qwen_vl import b13
    from orchestrator import router, trace
    from orchestrator.registry import get

    reference = json.loads(args.reference.read_text())
    identity = b13.adapter_identity(args.adapter)
    if reference["adapter_identity"]["files_sha256"] != identity["files_sha256"]:
        raise ValueError("Reference adapter differs")
    selected = frozen_questions(reference)
    for row in selected:
        image = args.images / row["image"]
        with image.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != reference["selected_image_sha256"][row["image"]]:
            raise ValueError(f"Frozen image hash mismatch: {image}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace.TRACE_PATH = args.out.with_suffix(".trace.jsonl")
    if args.out.exists() or trace.TRACE_PATH.exists():
        raise FileExistsError("Choose a fresh report/trace path")
    model = get("qwen2.5vl-3b")
    from models.artifacts import validate_artifact
    artifact = validate_artifact(model.name, model_id=model.model_id)
    readiness = model.readiness()
    load_seconds, load_error = None, None
    if readiness.available:
        began = time.perf_counter()
        try:
            model._load()
            load_seconds = time.perf_counter() - began
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
    else:
        load_error = f"{readiness.reason_code}: {readiness.detail}"
    records = []
    peak_mps_bytes = 0
    for row in selected:
        record = {key: row[key] for key in ("sample_id", "question_type", "question", "expected", "image")}
        record.update(kaggle_prediction=row["prediction"], kaggle_correct=row["correct"])
        began = time.perf_counter()
        try:
            if load_error and row["question_type"] != "count":
                raise RuntimeError(load_error)
            result = router.route("single_image_vqa", [str(args.images / row["image"])],
                                  row["question"], params={"execution_mode": "live"})
            prediction = result["answer"]
            record.update(prediction=prediction,
                          agrees=prediction.casefold().strip() == row["prediction"].casefold().strip(),
                          correct=prediction.casefold().strip() == row["expected"].casefold().strip(),
                          answer_source=result.get("answer_source"),
                          model_provenance=result.get("model_provenance"))
            if model.device == "mps":
                import torch
                torch.mps.synchronize()
                peak_mps_bytes = max(peak_mps_bytes, torch.mps.driver_allocated_memory())
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - began
        records.append(record)
    completed = sum("prediction" in row for row in records)
    report = {
        "status": "complete" if completed == 40 else "incomplete",
        "requested_device": "cpu" if args.cpu else "mps_with_cpu_fallback",
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "qwen-vl-utils")},
        "manifest_identity": reference["manifest_identity"],
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "selection": "First ten records of each type in frozen reference order; no cherry-picking",
        "adapter_identity": identity, "base_load_seconds": load_seconds, "load_error": load_error,
        "base_artifact": {"available": artifact.available, "detail": artifact.detail},
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "sampled_peak_mps_driver_bytes": peak_mps_bytes or None,
        "memory_scope": "In-process app router; RSS includes interpreter; MPS sampled after each question, not a kernel peak",
        "inference_seconds_per_question": sum(row["seconds"] for row in records if "prediction" in row) / completed if completed else None,
        "comparison": "Kaggle 4-bit adapter versus local fp16 app; count uses Grounding DINO and is not adapter parity",
        "summary": summarise(records), "records": records,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("status", "load_error", "summary")}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
