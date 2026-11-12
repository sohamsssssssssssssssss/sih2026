"""Unrun CUDA evaluator: original DIOR val counting, or LR gate negative control."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites.grounding_dior_rsvg import DIOR_CATEGORIES
from kaggle.run_grounding_dior_rsvg import _find_dataset_root, _safe_extract
from models.grounding_dino.count import CONFIG, count_objects, load_thresholds
from models.grounding_dino.resolution import resolution_gate
from models.qwen_vl.b13 import count_target

MIRROR_URL = "https://zenodo.org/records/11213149/files/DIOR-VOC.zip"
SEED = 17
VALIDATION_SHA256 = "b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c"
PROMPTS = {"baseballfield": "baseball field", "basketballcourt": "basketball court",
           "golffield": "golf field", "groundtrackfield": "ground track field",
           "storagetank": "storage tank", "tenniscourt": "tennis court", "trainstation": "train station"}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def layout_error(root, message):
    tree = sorted((str(p.relative_to(root)) for p in root.rglob("*")), key=lambda p: (p.count("/"), p))[:60]
    raise ValueError(f"{message}. Found tree under {root}:\n" + "\n".join(tree))


def read_splits(root):
    splits = {}
    for split in ("train", "val", "test"):
        path = root / "ImageSets/Main" / f"{split}.txt"
        if not path.is_file():
            layout_error(root, f"Missing official image-ID split {path}")
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in ids):
            layout_error(root, f"Invalid or duplicate image IDs in {path}")
        splits[split] = sorted(ids)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[left]) & set(splits[right]):
            raise ValueError(f"Image-level split leakage: {left} overlaps {right}")
    return splits


def annotations(root, ids):
    known = {re.sub(r"[^a-z0-9]", "", c): c for c in DIOR_CATEGORIES}
    records = {}
    for image_id in ids:
        xml = root / "Annotations" / f"{image_id}.xml"
        if not xml.is_file():
            layout_error(root, f"Missing exhaustive train/val annotation {xml}")
        try:
            annotation = ET.parse(xml).getroot()
            name = annotation.findtext("filename")
            if not name:
                raise ValueError("Missing VOC filename")
            if Path(name).name != name or Path(name).stem != image_id:
                raise ValueError("XML filename does not match image ID")
            if (int(annotation.findtext("size/width", "0")), int(annotation.findtext("size/height", "0"))) != (800, 800):
                raise ValueError("Expected original DIOR 800x800 annotations")
            counts = Counter()
            for obj in annotation.findall("object"):
                category = known.get(re.sub(r"[^a-z0-9]", "", (obj.findtext("name") or "").lower()))
                if category is None:
                    raise ValueError("Unknown DIOR class")
                box = [float(obj.findtext(f"bndbox/{key}", "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(v) and 0 <= v <= 800 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Missing or malformed VOC bounding box")
                counts[category] += 1  # Include difficult/truncated objects; no annotation filtering.
            image = root / "JPEGImages" / name
            if not image.is_file():
                raise ValueError(f"Missing image {image}")
            records[image_id] = {"filename": name, "counts": dict(counts), "xml_sha256": sha256(xml)}
        except (ET.ParseError, ValueError) as exc:
            layout_error(root, f"Invalid VOC annotation {xml}: {exc}")
    return records


def majority_counts(train, classes):
    modes = {}
    for category in classes:
        histogram = Counter(record["counts"].get(category, 0) for record in train.values())
        if not histogram:
            raise ValueError("Train split is empty")
        modes[category] = min(histogram, key=lambda count: (-histogram[count], count))
    return modes


def sample_pairs(val, classes, sample_size=1000):
    if sample_size < 2 * len(classes) or sample_size % 2:
        raise ValueError("Sample size must be even and permit a positive/zero pair per class")
    base, remainder = divmod(sample_size // 2, len(classes))
    selected = []
    for index, category in enumerate(classes):
        quota = base + int(index < remainder)
        for positive in (True, False):
            pool = [image_id for image_id in sorted(val) if (val[image_id]["counts"].get(category, 0) > 0) == positive]
            if len(pool) < quota:
                raise ValueError(f"Insufficient {category} {'positive' if positive else 'zero'} image/class pairs: {len(pool)} < {quota}")
            random.Random(f"{SEED}:{category}:{positive}").shuffle(pool)
            for image_id in pool[:quota]:
                item = val[image_id]
                selected.append({"image_id": image_id, "class": category, "image": item["filename"],
                                 "expected": item["counts"].get(category, 0), "xml_sha256": item["xml_sha256"],
                                 "prompt": PROMPTS.get(category, category.replace("_", " "))})
    return sorted(selected, key=lambda row: (row["image_id"], row["class"]))


def freeze_manifest(path, manifest):
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen sample differs; choose a new sample manifest path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return sha256(path)


def metrics(records):
    if not records:
        raise ValueError("Cannot score empty count evaluation")
    errors = [abs(r["prediction"] - r["expected"]) for r in records]
    return {"n": len(errors), "accuracy": sum(e == 0 for e in errors) / len(errors),
            "mean_absolute_error": sum(errors) / len(errors), "within_one": sum(e <= 1 for e in errors) / len(errors)}


def summary(records, classes):
    return {"overall": metrics(records), "per_class": {
        category: metrics([row for row in records if row["class"] == category]) for category in classes}}


def count_rows(manifest):
    if sha256(manifest) != VALIDATION_SHA256:
        raise ValueError("Expected frozen B13 LR validation manifest")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = [row for row in rows if row["question_type"] == "count"]
    if len(rows) != 389 or len(selected) != 100:
        raise ValueError("Wrong frozen LR cohort")
    return selected


def lr_negative_control(args):
    if args.manifest is None:
        raise ValueError("lr_negative_control requires --manifest")
    rows = count_rows(args.manifest)
    gates = [{"sample_id": row["sample_id"], **resolution_gate(count_target(row["question"]), 10.0,
              "RSVQA-LR dataset GSD", args.config)} for row in rows]
    if any(gate["count_status"] != "not_resolvable" for gate in gates):
        raise ValueError("LR negative control expected refusal for every count question")
    return {"mode": "lr_negative_control", "n": len(gates), "ground_truth_zero_count": sum(row["answer"] == "0" for row in rows),
            "expected_outcome": "not_resolvable; no detector accuracy measured", "gates": gates}


def evaluate_dior(args):
    from PIL import Image
    from models.grounding_dino import GroundingDINOModel
    from models.artifacts import validate_artifact
    if args.archive is None or args.data_root is None or args.sample_manifest is None:
        raise ValueError("DIOR requires --archive, --data-root and --sample-manifest")
    settings = load_thresholds(args.config)
    thresholds = sorted(set(args.thresholds))
    if settings.box_threshold not in thresholds:
        raise ValueError("Include the app box threshold in the sweep")
    for threshold in thresholds:
        replace(settings, box_threshold=threshold)
    classes = tuple(c for c in DIOR_CATEGORIES if c in args.classes)
    if len(classes) != len(args.classes) or not classes:
        raise ValueError("Select unique known DIOR classes")
    archive_hash = sha256(args.archive)
    source = {"kind": "third_party_mirror", "url": MIRROR_URL,
              "zip_sha256": archive_hash, "zip_bytes": args.archive.stat().st_size}
    root = _find_dataset_root(args.data_root, "ImageSets/Main")
    if root is None:
        args.data_root.mkdir(parents=True, exist_ok=True)
        _safe_extract(args.archive, args.data_root)
        root = _find_dataset_root(args.data_root, "ImageSets/Main")
    if root is None:
        layout_error(args.data_root, "Expected ImageSets/Main/{train,val,test}.txt, Annotations/*.xml and JPEGImages")
    splits = read_splits(root)
    train = annotations(root, splits["train"])
    val = annotations(root, splits["val"])
    modes = majority_counts(train, classes)
    rows = sample_pairs(val, classes, args.sample_size)
    image_hashes = {}
    for row in rows:
        image = root / "JPEGImages" / row["image"]
        if row["image"] not in image_hashes:
            with Image.open(image) as opened:
                if opened.size != (800, 800):
                    raise ValueError(f"Expected an original 800x800 DIOR image: {image}")
            image_hashes[row["image"]] = sha256(image)
        row["image_sha256"] = image_hashes[row["image"]]
    frozen = {"dataset": "original DIOR VOC mirror", "source": source, "eval_split": "val", "baseline_split": "train",
              "seed": SEED, "sample_size": args.sample_size, "positive_zero_ratio": "1:1 per class",
              "classes": list(classes), "train_majority_counts": modes,
              "split_sha256": {name: sha256(root / "ImageSets/Main" / f"{name}.txt") for name in splits},
              "train_annotations_identity": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest(),
              "thresholds": thresholds, "count_config": asdict(settings), "pairs": rows}
    # Publish and fsync the sample before constructing/loading the detector or running inference.
    manifest_hash = freeze_manifest(args.sample_manifest, frozen)
    model = GroundingDINOModel(box_threshold=min(thresholds), text_threshold=settings.text_threshold)
    ready = model.readiness()
    if not ready.available:
        raise RuntimeError(f"Detector unavailable: {ready.reason_code}: {ready.detail}")
    artifact = validate_artifact(model.name)
    if not artifact.available or artifact.path is None:
        raise RuntimeError("Detector checkpoint unavailable")
    checkpoint_hash = sha256(artifact.path)
    conditions = {threshold: [] for threshold in thresholds}
    for row in rows:
        paths = [str(root / "JPEGImages" / row["image"])]
        proposals = model.infer(paths, row["prompt"])
        for threshold in thresholds:
            result = count_objects(paths, row["prompt"], lambda *_: proposals, replace(settings, box_threshold=threshold))
            conditions[threshold].append({**row, "prediction": result["count"], "evidence": result["evidence"]})
    baselines = {"always_zero": summary([{**row, "prediction": 0} for row in rows], classes),
                 "train_majority_count": summary([{**row, "prediction": modes[row["class"]]} for row in rows], classes)}
    return {"mode": "dior", "source": source, "sample_manifest_sha256": manifest_hash, "sample_size": len(rows),
            "split_sizes": {name: len(ids) for name, ids in splits.items()}, "count_config": asdict(settings),
            "detector": {"name": model.name, "version": model.version, "checkpoint_sha256": checkpoint_hash},
            "metrics": summary(conditions[settings.box_threshold], classes), "baselines": baselines,
            "threshold_sweep": {str(key): {**summary(records, classes), "records": records} for key, records in conditions.items()},
            "scope": "Detector counting only, not resolution-gate performance. DIOR GSD is not in VOC annotations.",
            "annotation_assumption": "VOC annotations exhaustively label instances of the 20 DIOR classes; zero-count negatives rely on this."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dior", "lr_negative_control"), default="dior")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--classes", nargs="+", default=list(DIOR_CATEGORIES))
    parser.add_argument("--manifest", type=Path, help="Only for lr_negative_control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40])
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires CUDA; no CPU/VLM fallback")
    if args.output.exists():
        raise FileExistsError("Choose a fresh evaluation output")
    report = lr_negative_control(args) if args.mode == "lr_negative_control" else evaluate_dior(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("mode", "metrics") if key in report}, indent=2))


if __name__ == "__main__":
    main()
