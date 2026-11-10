"""Frozen RGB pilot integrity, scoring and paired image-cluster statistics."""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from scripts.prepare_rsvqa_pilot import digest, TYPES
from training.remote_sensing import load_manifest, sha256_file

PILOT_IDENTITY = "1eb622b1ca4bec1d2d849f049623f015f3e015c423ce8fab146e91e5b86f43ea"
SCORING_VERSION = "rsvqa-rgb-pilot-exact-raw-integer-v1"
CONDITIONS = ("base_real", "base_constant", "adapter_real", "adapter_constant")
COMPARISONS = (("adapter_real", "base_real"), ("base_real", "base_constant"),
               ("adapter_real", "adapter_constant"), ("adapter_constant", "base_constant"))


def score(prediction, answer, question_type):
    """Our pilot policy: canonical nonnegative raw integers, no count bins."""
    pred, gold = prediction.strip().casefold(), answer.strip().casefold()
    if question_type == "count":
        valid = lambda s: re.fullmatch(r"0|[1-9][0-9]*", s) is not None
    elif question_type in ("presence", "comp"):
        valid = lambda s: s in ("yes", "no")
    elif question_type == "rural_urban":
        valid = lambda s: s in ("rural", "urban")
    else:
        raise ValueError("Unknown question type")
    if not valid(gold):
        raise ValueError("Malformed gold answer")
    malformed = not valid(pred)
    return {"correct": not malformed and pred == gold, "malformed": malformed}


def summarise(records):
    counts = Counter(r["question_type"] for r in records)
    per_type = {t: {"n": counts[t],
                    "accuracy": sum(r["correct"] for r in records if r["question_type"] == t) / counts[t] if counts[t] else None,
                    "malformed_n": sum(r["malformed"] for r in records if r["question_type"] == t)} for t in TYPES}
    return {"n": len(records), "per_type": per_type,
            "macro_accuracy": float(np.mean([v["accuracy"] for v in per_type.values()])) if all(counts[t] for t in TYPES) else None,
            "malformed_n": sum(r["malformed"] for r in records)}


def validate_pilot(provenance, manifest, image_root, expected_identity=PILOT_IDENTITY):
    report = json.loads(Path(provenance).read_text())
    identity = report.pop("manifest_identity")
    if digest(report) != identity or identity != expected_identity:
        raise ValueError("Pilot manifest identity mismatch")
    directory = Path(provenance).parent
    for name, expected in report["manifest_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Manifest hash mismatch: {name}")
    if Path(manifest).resolve() != (directory / "validation.jsonl").resolve():
        raise ValueError("Pilot evaluation requires the frozen validation manifest")
    root = Path(image_root).resolve()
    for name, expected in report["selected_image_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != expected:
            raise ValueError(f"Image hash mismatch: {name}")
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    examples = load_manifest(Path(manifest), root)
    qids, sample_ids = set(), set()
    for row, example in zip(rows, examples):
        if row["split"] != "validation" or row["question_id"] in qids or row["sample_id"] in sample_ids:
            raise ValueError("Duplicate identity or non-validation pilot row")
        qids.add(row["question_id"]); sample_ids.add(row["sample_id"])
        if (row["question"] != example.instruction or row["answer"] != example.response
                or row["image"] != f"{row['image_id']}.tif"):
            raise ValueError("Pilot question/answer/image identity mismatch")
        score(row["answer"], row["answer"], row["question_type"])
    if not rows:
        raise ValueError("Empty pilot validation manifest")
    report["manifest_identity"] = identity
    return rows, examples, report


def adapter_identity(path):
    """Hash safe adapter files without loading tensors or pickle objects."""
    root = Path(path)
    names = ("adapter_config.json", "adapter_model.safetensors")
    files = {n: sha256_file(root / n) for n in names}
    return {"sha256": digest(files), "files_sha256": files}


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "qwen-vl-utils", "Pillow", "numpy", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def paired_differences(conditions, *, seed=17, repetitions=2000):
    """Resample image clusters with replacement, keeping every QA and pairing."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    reference = conditions[CONDITIONS[0]]["records"]
    identity = lambda r: (r["sample_id"], r["question_id"], r["image_id"], r["question_type"], r["expected"])
    for name in CONDITIONS:
        if [identity(r) for r in conditions[name]["records"]] != [identity(r) for r in reference]:
            raise ValueError("Condition alignment mismatch")
    images = sorted({r["image_id"] for r in reference})
    if not images:
        raise ValueError("No image clusters")
    image_index = {image: i for i, image in enumerate(images)}
    counts = np.zeros((len(images), len(TYPES)))
    correct = {name: np.zeros_like(counts) for name in CONDITIONS}
    for j, r in enumerate(reference):
        i, t = image_index[r["image_id"]], TYPES.index(r["question_type"])
        counts[i, t] += 1
        for name in CONDITIONS:
            correct[name][i, t] += conditions[name]["records"][j]["correct"]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(images), size=(repetitions, len(images)))
    denominators = counts[draws].sum(axis=1)
    result = {}
    for a, b in COMPARISONS:
        delta = correct[a] - correct[b]
        totals = counts.sum(axis=0)
        point = np.divide(delta.sum(axis=0), totals, out=np.full(len(TYPES), np.nan), where=totals > 0)
        boot = np.divide(delta[draws].sum(axis=1), denominators,
                         out=np.full_like(denominators, np.nan), where=denominators > 0)
        def interval(values):
            valid = values[np.isfinite(values)]
            return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) and len(images) >= 2 else None,
                    "valid_replicates": len(valid)}
        types = {t: {"difference": float(point[k]) if np.isfinite(point[k]) else None,
                     **interval(boot[:, k])} for k, t in enumerate(TYPES)}
        # Replicates lacking any category cannot define the four-type macro.
        macros = boot.mean(axis=1)
        result[f"{a}_minus_{b}"] = {"per_type": types,
            "macro": {"difference": float(point.mean()) if np.isfinite(point).all() else None, **interval(macros)}}
    return {"method": "paired percentile bootstrap of image clusters; QA-weighted within each type; equal-weight four-type macro",
            "seed": seed, "repetitions": repetitions, "image_clusters": len(images), "comparisons": result}
