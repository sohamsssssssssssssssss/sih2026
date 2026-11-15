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
from kaggle.run_grounding_dior_rsvg import _safe_extract
from models.grounding_dino.count import CONFIG, count_objects, load_thresholds
from models.grounding_dino.resolution import resolution_gate
from models.qwen_vl.b13 import count_target

MIRROR_URL = "https://zenodo.org/records/11213149/files/DIOR-VOC.zip"
# Verified on Kaggle CPU, 13 Nov 2026. Any other bytes are a hard stop.
MIRROR_SHA256 = "7a386a56ba815333627d89d3b8b372d1af2a07b78955c017530ec62b7f1e0540"
MIRROR_BYTES = 7_423_852_627
DATASET_SUBDIR = Path("DIOR-VOC/VOC2007")
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


def verify_archive(path, expected_sha256=MIRROR_SHA256, expected_bytes=MIRROR_BYTES):
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"DIOR mirror size {size} != pinned {expected_bytes}; refusing to evaluate")
    digest = sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"DIOR mirror sha256 {digest} != pinned {expected_sha256}; refusing to evaluate")
    return digest


def dataset_root(data_root, archive=None):
    """The verified mirror's DIOR-VOC/VOC2007 root; extract into an empty data_root, else fail with the tree."""
    root = data_root / DATASET_SUBDIR

    def complete():
        return all((root / sub).is_dir() for sub in ("ImageSets/Main", "Annotations", "JPEGImages"))
    if not complete() and archive is not None and not (data_root.exists() and any(data_root.iterdir())):
        data_root.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, data_root)
    if not complete():
        layout_error(data_root if data_root.exists() else data_root.parent,
                     f"Expected {DATASET_SUBDIR}/{{ImageSets/Main,Annotations,JPEGImages}}")
    return root


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


INVALID_ANNOTATION_CAP = 0.01  # More invalid XML than this per split means a wrong/corrupt mirror: hard stop.


class InvalidAnnotation(ValueError):
    def __init__(self, reason, detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def voc_counts(xml, image_id, known):
    """(filename, per-class counts) of one VOC XML, or InvalidAnnotation(reason, detail)."""
    try:
        annotation = ET.parse(xml).getroot()
    except ET.ParseError as exc:
        raise InvalidAnnotation("parse_error", str(exc)) from exc
    name = annotation.findtext("filename")
    if not name or Path(name).name != name or Path(name).stem != image_id:
        raise InvalidAnnotation("bad_filename", {"filename": name})
    width, height = annotation.findtext("size/width"), annotation.findtext("size/height")
    try:
        size = (int(width), int(height))
    except (TypeError, ValueError):
        size = None
    if size != (800, 800):
        raise InvalidAnnotation("nonstandard_size", {"width": width, "height": height})
    counts = Counter()
    for obj in annotation.findall("object"):
        label = obj.findtext("name")
        category = known.get(re.sub(r"[^a-z0-9]", "", (label or "").lower()))
        if category is None:
            raise InvalidAnnotation("unknown_class", {"name": label})
        raw = [obj.findtext(f"bndbox/{key}") for key in ("xmin", "ymin", "xmax", "ymax")]
        try:
            box = [float(v) for v in raw]
        except (TypeError, ValueError):
            box = [math.nan] * 4
        if not all(math.isfinite(v) and 0 <= v <= 800 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise InvalidAnnotation("bad_bbox", {"bndbox": raw})
        counts[category] += 1  # Include difficult/truncated objects; no annotation filtering.
    return name, counts


def annotations(root, ids, require_image=True):
    """Parse VOC counts. Ids whose XML (or, when required, JPEG) is missing, or whose XML is invalid, are
    dropped and returned; invalid XML above INVALID_ANNOTATION_CAP of the requested ids stops the run."""
    known = {re.sub(r"[^a-z0-9]", "", c): c for c in DIOR_CATEGORIES}
    records, dropped = {}, {"missing_xml": [], "missing_image": [], "invalid_annotation": []}
    for image_id in ids:
        xml = root / "Annotations" / f"{image_id}.xml"
        if not xml.is_file():
            dropped["missing_xml"].append(image_id)
            continue
        try:
            name, counts = voc_counts(xml, image_id, known)
        except InvalidAnnotation as exc:
            dropped["invalid_annotation"].append({"id": image_id, "reason": exc.reason, "detail": exc.detail})
            continue
        if require_image and not (root / "JPEGImages" / name).is_file():
            dropped["missing_image"].append(image_id)
            continue
        records[image_id] = {"filename": name, "counts": dict(counts), "xml_sha256": sha256(xml)}
    dropped = {"missing_xml": sorted(dropped["missing_xml"]), "missing_image": sorted(dropped["missing_image"]),
               "invalid_annotation": sorted(dropped["invalid_annotation"], key=lambda d: d["id"])}
    invalid = dropped["invalid_annotation"]
    if len(invalid) > INVALID_ANNOTATION_CAP * len(ids):
        reasons = dict(sorted(Counter(d["reason"] for d in invalid).items()))
        raise ValueError(f"Invalid VOC annotations exceed the {INVALID_ANNOTATION_CAP:.0%} cap under {root}: "
                         f"{len(invalid)}/{len(ids)} ids, reasons {reasons}, first ids {[d['id'] for d in invalid[:20]]}")
    return records, dropped


def dropped_summary(dropped, requested, kept):
    by_reason = {"missing_xml": len(dropped["missing_xml"]), "missing_image": len(dropped["missing_image"]),
                 **Counter(d["reason"] for d in dropped["invalid_annotation"])}
    return {**dropped, "n_dropped": sum(map(len, dropped.values())), "n_requested": requested, "n_kept": kept,
            "n_dropped_by_reason": dict(sorted(by_reason.items()))}


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
    archive_hash = verify_archive(args.archive)
    source = {"kind": "third_party_mirror", "url": MIRROR_URL, "zip_sha256": archive_hash,
              "zip_sha256_pinned": True, "zip_bytes": args.archive.stat().st_size}
    root = dataset_root(args.data_root, args.archive)
    splits = read_splits(root)
    train, train_dropped = annotations(root, splits["train"], require_image=False)  # baseline needs XML only
    val, val_dropped = annotations(root, splits["val"])
    dropped = {"train": dropped_summary(train_dropped, len(splits["train"]), len(train)),
               "val": dropped_summary(val_dropped, len(splits["val"]), len(val))}
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
              "dropped_before_sampling": dropped, "seed": SEED, "sample_size": args.sample_size, "positive_zero_ratio": "1:1 per class",
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
            "split_sizes": {name: len(ids) for name, ids in splits.items()},
            "dropped_before_sampling": dropped, "count_config": asdict(settings),
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
