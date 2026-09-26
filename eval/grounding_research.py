"""Research-only local (CPU or Apple MPS) harness for zero-shot Grounding DINO on DIOR-RSVG.

NOT a production path: the provider stays CUDA-only. The official Swin-T
checkpoint is loaded on CPU or Apple MPS and injected into an unmodified
GroundingDINOModel, so the provider's evidence construction and the suite's
scoring primitives run unchanged. A forward hook keeps the raw decoder
outputs (900 query scores + boxes) of that same forward pass, so thresholds
can be swept offline; `sweep` proves offline == online before reporting.
"""

import argparse
import hashlib
import itertools
import json
import math
import re
import shlex
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import registry  # noqa: E402
from eval.suites import grounding_dior_rsvg as suite  # noqa: E402
from models.grounding_dino import GroundingDINOModel  # noqa: E402

BASELINE_BOX_THRESHOLD = 0.35
BASELINE_TEXT_THRESHOLD = 0.25
IOU_LEVELS = (0.25, 0.5, 0.75)
SWEEP_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
PACKAGES = ("torch", "torchvision", "transformers", "groundingdino-py", "timm", "numpy", "pillow")
READABLE_CATEGORY = {
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "expressway_service_area": "expressway service area",
    "expressway_toll_station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
}
# First spatial/relational cue; everything from it onward is dropped by `head`.
_RELATION = re.compile(
    r"\s+(?:in|on|at|near|next to|beside|between|under|above|below|behind|"
    r"which|that|is|are|located|lying|close to|to the)\b.*$",
    re.IGNORECASE,
)


def head_phrase(expression: str) -> str:
    """Expression truncated before its first relational clause (never empty)."""
    return _RELATION.sub("", expression.strip()) or expression.strip()


PROMPTS = {
    "expression": lambda row: row["expression"],
    "head": lambda row: head_phrase(row["expression"]),
    "aerial": lambda row: f"aerial image of {row['expression']}",
    # Diagnostic only: uses the annotated class, which a real user query does not carry.
    "category": lambda row: READABLE_CATEGORY.get(row["category"], row["category"]),
}
DEPLOYABLE_PROMPTS = ("expression", "head", "aerial")


def wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- inference

def _raw(outputs: dict) -> dict:
    # Same tensors and ops, in the same order (.cpu() before sigmoid), as
    # groundingdino.util.inference.predict, so replay is exact on any device.
    return {
        "scores": outputs["pred_logits"].cpu().sigmoid()[0].max(dim=1).values.clone(),
        "boxes": outputs["pred_boxes"].cpu()[0].clone(),
    }


def local_provider(
    checkpoint: Path,
    *,
    device: str = "cpu",
    box_threshold: float = BASELINE_BOX_THRESHOLD,
    text_threshold: float = BASELINE_TEXT_THRESHOLD,
    sink: list | None = None,
) -> GroundingDINOModel:
    from groundingdino.util.inference import load_image, load_model, predict

    provider = GroundingDINOModel(box_threshold=box_threshold, text_threshold=text_threshold)
    model = load_model(str(provider._resolve_config()), str(checkpoint), device=device)
    if sink is not None:
        model.register_forward_hook(lambda _module, _args, outputs: sink.append(_raw(outputs)))
    # ponytail: pre-seeding the lazy-load slots skips only _load()'s CUDA gate;
    # the device override is the only change to the call into predict().
    # infer(), its evidence validation and box conversion are the production code.
    provider._model, provider._load_image = model, load_image
    provider._predict_fn = lambda **kwargs: predict(**{**kwargs, "device": device})
    return provider


def _materialize(image_path: Path, dataset_root: Path, archive: Path | None) -> bool:
    if image_path.is_file() or archive is None:
        return False
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract(f"JPEGImages/{image_path.name}", dataset_root)
    return True


def run_samples(
    provider: GroundingDINOModel,
    samples: list[dict],
    prompt: str,
    dataset_root: Path,
    sink: list,
    *,
    archive: Path | None = None,
    ephemeral: bool = False,
) -> list[dict]:
    """Mirror suite.evaluate's per-row logic for any split, sample and prompt."""
    rows: list[dict] = []
    for image_path, group in itertools.groupby(samples, key=lambda row: row["image_path"]):
        extracted = _materialize(image_path, dataset_root, archive)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            for sample in group:
                gold = suite.normalize_xyxy(sample["ground_truth_pixel_xyxy"], width, height)
                query = PROMPTS[prompt](sample)
                before = len(sink)
                prediction = provider.infer([str(image_path)], query)
                if len(sink) != before + 1:
                    raise RuntimeError("expected exactly one forward pass per inference")
                top = suite._top_prediction(prediction["evidence"])
                rows.append(
                    {
                        "test_index": sample["test_index"],
                        "image_id": sample["image_id"],
                        "category": sample["category"],
                        "expression": sample["expression"],
                        "query": query,
                        "image_size": [width, height],
                        "gold": gold,
                        "evidence": prediction["evidence"],
                        "selected": top,
                        "n_detections": len(prediction["evidence"]),
                        "iou": suite.iou_xyxy(top["coordinates"], gold) if top else 0.0,
                    }
                )
                if len(rows) % 50 == 0:
                    print(f"progress={len(rows)}/{len(samples)}", flush=True)
        finally:
            if extracted and ephemeral:
                image_path.unlink()
    return rows


# --------------------------------------------------------------------------- offline replay

def _xyxy(box: list[float]) -> list[float]:
    # Mirrors GroundingDINOModel.infer's cxcywh -> clipped normalized xyxy.
    cx, cy, w, h = map(float, box)
    clip = lambda value: max(0.0, min(1.0, value))  # noqa: E731
    return [clip(cx - w / 2), clip(cy - h / 2), clip(cx + w / 2), clip(cy + h / 2)]


def replay(raw: dict, box_threshold: float) -> tuple[list[float] | None, float | None, int]:
    """Offline equivalent of provider.infer + suite._top_prediction (top-1 rule)."""
    kept = (raw["scores"] > box_threshold).nonzero(as_tuple=True)[0]
    if len(kept) == 0:
        return None, None, 0
    top = int(kept[raw["scores"][kept].argmax()])
    return _xyxy(raw["boxes"][top].tolist()), float(raw["scores"][top]), len(kept)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = {level: sum(row["iou"] >= level for row in rows) for level in IOU_LEVELS}
    detections = [row["n_detections"] for row in rows]
    answered = sum(count > 0 for count in detections)
    return {
        "n": n,
        **{f"pr_at_{level}": hits[level] / n for level in IOU_LEVELS},
        "hits_at_0.5": hits[0.5],
        "pr_at_0.5_wilson95": wilson(hits[0.5], n),
        "mean_iou": sum(row["iou"] for row in rows) / n,
        "no_detection_rate": 1 - answered / n,
        "multi_detection_rate": sum(count > 1 for count in detections) / n,
        "mean_detections": mean(detections),
        "median_detections": median(detections),
        # Pr@0.5 among answered expressions: what the threshold buys as abstention.
        "selective_pr_at_0.5": hits[0.5] / answered if answered else None,
    }


def sweep(rows: list[dict], raws: list[dict], thresholds=SWEEP_THRESHOLDS) -> dict:
    table = {}
    for threshold in thresholds:
        replayed = []
        for row, raw in zip(rows, raws, strict=True):
            box, _, count = replay(raw, threshold)
            iou = suite.iou_xyxy(box, row["gold"]) if box else 0.0
            replayed.append({"iou": iou, "n_detections": count})
        table[str(threshold)] = summarize(replayed)
    return table


def parity(rows: list[dict], raws: list[dict], box_threshold: float) -> dict:
    """Count rows where the offline replay differs from the online provider run."""
    mismatches = []
    for row, raw in zip(rows, raws, strict=True):
        box, score, count = replay(raw, box_threshold)
        online = row["selected"]
        same = count == row["n_detections"] and (
            (online is None and box is None)
            or (online is not None and box == online["coordinates"] and score == online["confidence"])
        )
        if not same:
            mismatches.append(row["test_index"])
    return {"n": len(rows), "mismatches": len(mismatches), "mismatched_indices": mismatches}


def rows_from_suite_report(report: dict) -> list[dict]:
    """Convert a canonical suite report (e.g. the original T4 run) to harness rows."""
    return [
        {
            "test_index": r["test_index"], "image_id": r["image_id"], "category": r["category"],
            "expression": r["referring_expression"], "query": r["referring_expression"],
            "image_size": [r["image_size"]["width"], r["image_size"]["height"]],
            "gold": r["ground_truth"]["coordinates"], "evidence": r["prediction"]["evidence"],
            "selected": r["selected_prediction"], "n_detections": len(r["prediction"]["evidence"]),
            "iou": r["iou"],
        }
        for r in report["results"]
    ]


def compare_rows(reference: list[dict], candidate: list[dict]) -> dict:
    """Per-example agreement between two runs over the same expressions."""
    pairs = list(zip(reference, candidate, strict=True))
    if any(a["test_index"] != b["test_index"] for a, b in pairs):
        raise ValueError("runs are not aligned on test_index")
    hit = lambda row: row["iou"] >= 0.5  # noqa: E731
    confidence = lambda row: row["selected"]["confidence"] if row["selected"] else None  # noqa: E731
    iou_deltas = [abs(a["iou"] - b["iou"]) for a, b in pairs]
    confidence_deltas = [
        abs(confidence(a) - confidence(b))
        for a, b in pairs
        if confidence(a) is not None and confidence(b) is not None
    ]
    return {
        "n": len(pairs),
        "hits_reference": sum(map(hit, reference)),
        "hits_candidate": sum(map(hit, candidate)),
        "hit_flips": [b["test_index"] for a, b in pairs if hit(a) != hit(b)],
        "detection_presence_flips": [
            b["test_index"] for a, b in pairs if (a["selected"] is None) != (b["selected"] is None)
        ],
        "n_detections_differ": sum(a["n_detections"] != b["n_detections"] for a, b in pairs),
        "bit_identical_selected": sum(a["selected"] == b["selected"] for a, b in pairs),
        "max_abs_iou_delta": max(iou_deltas),
        "mean_abs_iou_delta": mean(iou_deltas),
        "max_abs_confidence_delta": max(confidence_deltas, default=0.0),
        "mean_iou_reference": mean(row["iou"] for row in reference),
        "mean_iou_candidate": mean(row["iou"] for row in candidate),
    }


# --------------------------------------------------------------------------- failure analysis

def image_objects(dataset_root: Path, image_id: str) -> list[dict]:
    annotation = ET.parse(dataset_root / "Annotations" / f"{image_id}.xml").getroot()
    return [
        {
            "category": suite._category(member.findtext("name")),
            "pixel_xyxy": [float(member.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")],
        }
        for member in annotation.findall("object")
    ]


def classify_failure(row: dict, objects: list[dict]) -> str:
    """Deterministic failure type for a row with IoU < 0.5 (objects in normalized xyxy)."""
    if row["selected"] is None:
        return "no_detection"
    box = row["selected"]["coordinates"]
    others = [obj for obj in objects if obj["normalized"] != row["gold"]]
    best = max(others, key=lambda obj: suite.iou_xyxy(box, obj["normalized"]), default=None)
    if best is not None and suite.iou_xyxy(box, best["normalized"]) >= 0.5:
        return "same_class_other_instance" if best["category"] == row["category"] else "other_class_object"
    if row["iou"] >= 0.1:
        return "poor_localization_of_target"
    return "background_or_unannotated"


def size_bucket(pixel_box: list[float]) -> str:
    area = (pixel_box[2] - pixel_box[0]) * (pixel_box[3] - pixel_box[1])
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def _group(rows: list[dict], key) -> dict:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {name: {**summarize(items)} for name, items in sorted(grouped.items())}


def analyze(rows: list[dict], dataset_root: Path) -> dict:
    enriched = []
    for row in rows:
        width, height = row["image_size"]
        objects = [
            {**obj, "normalized": suite.normalize_xyxy(obj["pixel_xyxy"], width, height)}
            for obj in image_objects(dataset_root, row["image_id"])
        ]
        pixel_gold = [row["gold"][0] * width, row["gold"][1] * height, row["gold"][2] * width, row["gold"][3] * height]
        same_class = sum(obj["category"] == row["category"] for obj in objects)
        enriched.append(
            {
                **row,
                "size": size_bucket(pixel_gold),
                "same_class_annotated": "1" if same_class == 1 else "2-3" if same_class <= 3 else ">=4",
                "relational": head_phrase(row["expression"]) != row["expression"].strip(),
                "failure": None if row["iou"] >= 0.5 else classify_failure(row, objects),
            }
        )
    failures = [row for row in enriched if row["failure"]]
    failure_counts: dict[str, int] = defaultdict(int)
    per_class_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in failures:
        failure_counts[row["failure"]] += 1
        per_class_failures[row["category"]][row["failure"]] += 1
    return {
        "overall": summarize(enriched),
        "per_category": _group(enriched, lambda row: row["category"]),
        "per_size": _group(enriched, lambda row: row["size"]),
        "per_same_class_annotated_count": _group(enriched, lambda row: row["same_class_annotated"]),
        "per_relational": _group(enriched, lambda row: "relational" if row["relational"] else "non_relational"),
        "failure_types": dict(sorted(failure_counts.items())),
        "failure_types_per_category": {k: dict(v) for k, v in sorted(per_class_failures.items())},
        "rows": [
            {key: row[key] for key in ("test_index", "category", "size", "same_class_annotated", "relational", "failure", "iou")}
            for row in enriched
        ],
    }


def render_examples(rows: list[dict], analysis: dict, dataset_root: Path, archive: Path | None, out: Path, per_type: int = 2) -> list[str]:
    """Draw gold (green) and top-1 (red) for the first `per_type` rows of each outcome by test_index."""
    by_index = {row["test_index"]: row for row in rows}
    chosen: dict[str, list[int]] = defaultdict(list)
    for item in sorted(analysis["rows"], key=lambda item: item["test_index"]):
        outcome = item["failure"] or "hit"
        if len(chosen[outcome]) < per_type:
            chosen[outcome].append(item["test_index"])
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for outcome, indices in sorted(chosen.items()):
        for index in indices:
            row = by_index[index]
            image_path = dataset_root / "JPEGImages" / f"{row['image_id']}.jpg"
            extracted = _materialize(image_path, dataset_root, archive)
            with Image.open(image_path) as image:
                canvas = image.convert("RGB")
            if extracted:
                image_path.unlink()
            width, height = canvas.size
            draw = ImageDraw.Draw(canvas)
            scale = lambda box: [box[0] * width, box[1] * height, box[2] * width, box[3] * height]  # noqa: E731
            draw.rectangle(scale(row["gold"]), outline=(0, 255, 0), width=3)
            if row["selected"]:
                draw.rectangle(scale(row["selected"]["coordinates"]), outline=(255, 0, 0), width=3)
            draw.text((6, 6), f"{outcome} | {row['query']} | IoU={row['iou']:.2f}", fill=(255, 255, 0))
            target = out / f"{outcome}__{index:05d}.png"
            canvas.save(target)
            written.append(str(target))
    return written


# --------------------------------------------------------------------------- selections and I/O

def select(data_root: Path, selection: str, seed: int) -> tuple[Path, list[dict], str]:
    dataset_root, test_records = suite.load_test_records(data_root)
    if selection == "tquick":
        return dataset_root, suite.stratified_sample(test_records, suite.DEFAULT_SAMPLE_SIZE, seed), "test"
    if selection == "tfull":
        return dataset_root, sorted(test_records, key=lambda row: row["test_index"]), "test"
    if selection == "val-select":
        # Official splits share images; keep validation images that never occur in test.
        _, validation = suite.load_test_records(data_root, expected_split_size=3829, split_file="val.txt")
        test_images = {row["image_id"] for row in test_records}
        eligible = [row for row in validation if row["image_id"] not in test_images]
        return dataset_root, suite.stratified_sample(eligible, suite.DEFAULT_SAMPLE_SIZE, seed), "val (images not in test)"
    raise ValueError(f"unknown selection {selection}")


def _save_run(out_dir: Path, summary: dict, rows: list[dict], raws: list[dict]) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    torch.save(raws, out_dir / "raw.pt")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    import torch

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "rows.json").read_text(encoding="utf-8"))
    return summary, rows, torch.load(run_dir / "raw.pt")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_run(args: argparse.Namespace) -> None:
    started, clock, repository = _now(), time.perf_counter(), registry.repository_state()
    dataset_root, samples, split = select(args.data_root, args.selection, args.seed)
    sink: list = []
    provider = local_provider(args.checkpoint, device=args.device, box_threshold=args.box_threshold, text_threshold=args.text_threshold, sink=sink)
    if args.harness == "suite":
        # Exact canonical harness: evaluate() constructs GroundingDINOModel() itself.
        if args.selection != "tquick" or args.prompt != "expression":
            raise ValueError("--harness suite only reproduces the locked T-quick expression run")
        original, suite.GroundingDINOModel = suite.GroundingDINOModel, lambda: provider
        try:
            report = suite.evaluate(args.data_root, seed=args.seed)
        finally:
            suite.GroundingDINOModel = original
        rows = rows_from_suite_report(report)
    else:
        rows = run_samples(provider, samples, args.prompt, dataset_root, sink, archive=args.archive, ephemeral=args.ephemeral_images)
    summary = {
        "selection": args.selection,
        "split": split,
        "seed": args.seed,
        "prompt": args.prompt,
        "harness": args.harness,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "selection_rule": "top-1 highest-confidence returned box; no box = IoU 0",
        "device": args.device,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "command": shlex.join(sys.argv),
        "started_at": started,
        "ended_at": _now(),
        "runtime_seconds": time.perf_counter() - clock,
        "metrics": summarize(rows),
        "per_category": _group(rows, lambda row: row["category"]),
        "environment": registry.environment(PACKAGES),
        "repository": repository,
    }
    _save_run(args.out_dir, summary, rows, sink)
    print(json.dumps(summary["metrics"], indent=2))


def _rows(path: Path) -> list[dict]:
    if path.is_dir():
        return json.loads((path / "rows.json").read_text(encoding="utf-8"))
    return rows_from_suite_report(json.loads(path.read_text(encoding="utf-8")))


def command_compare(args: argparse.Namespace) -> None:
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        **compare_rows(_rows(args.reference), _rows(args.candidate)),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def command_sweep(args: argparse.Namespace) -> None:
    summary, rows, raws = _load_run(args.run_dir)
    result = {
        "run": str(args.run_dir),
        "parity_at_run_threshold": parity(rows, raws, summary["box_threshold"]),
        "sweep": sweep(rows, raws),
    }
    (args.run_dir / "sweep.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["parity_at_run_threshold"]))
    for threshold, metrics in result["sweep"].items():
        print(threshold, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})


def command_analyze(args: argparse.Namespace) -> None:
    _, rows, _ = _load_run(args.run_dir)
    dataset_root = suite._dataset_root(args.data_root)
    analysis = analyze(rows, dataset_root)
    if args.render_dir:
        analysis["rendered_examples"] = render_examples(rows, analysis, dataset_root, args.archive, args.render_dir)
    (args.run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in analysis.items() if k not in ("rows", "failure_types_per_category")}, indent=1)[:6000])


def command_audit(args: argparse.Namespace) -> None:
    dataset_root = suite._dataset_root(args.data_root)
    object_image: list[str] = []
    for annotation in sorted((dataset_root / "Annotations").rglob("*.xml")):
        root = ET.parse(annotation).getroot()
        object_image.extend(root.findtext("filename") for _ in root.findall("object"))
    splits = {}
    for name in ("train", "val", "test"):
        indices = [int(line) for line in (dataset_root / f"{name}.txt").read_text().split()]
        splits[name] = {"expressions": len(indices), "images": sorted({object_image[i] for i in indices})}
    overlaps = {
        f"{a}&{b}": len(set(splits[a]["images"]) & set(splits[b]["images"]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in args.files}
    audit = {
        "dataset": "DIOR-RSVG",
        "source": "official Google Drive folder 1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_ (as used by kaggle/run_grounding_dior_rsvg.py)",
        "files": files,
        "total_expressions": len(object_image),
        "total_images": len(set(object_image)),
        "splits": {name: {"expressions": s["expressions"], "images": len(s["images"])} for name, s in splits.items()},
        "image_overlap": overlaps,
        "audited_at": _now(),
    }
    args.out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


def command_register(args: argparse.Namespace) -> None:
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8")) if args.run_dir else {}
    extra = json.loads(args.metrics_json.read_text(encoding="utf-8")) if args.metrics_json else {}
    record = {
        "experiment_id": args.experiment_id,
        "title": args.title,
        "dataset": "DIOR-RSVG",
        "dataset_version": "official Google Drive release (Annotations.zip 2023-04-07, JPEGImages.zip 2022-11-12)",
        "dataset_checksum": json.loads(args.dataset_audit.read_text(encoding="utf-8"))["files"],
        "split": summary.get("split", args.split),
        "model_id": "ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth (IDEA-Research groundingdino-py 0.4.0)",
        "model_revision": args.model_revision,
        "adapter_revision": None,
        "seed": summary.get("seed", args.seed),
        "hyperparameters": {
            key: summary[key]
            for key in ("selection", "prompt", "harness", "box_threshold", "text_threshold", "selection_rule", "device")
            if key in summary
        },
        "command": summary.get("command", args.command),
        "started_at": summary.get("started_at", args.started_at),
        "ended_at": summary.get("ended_at", args.ended_at or _now()),
        "metrics": {**({"summary": summary["metrics"]} if "metrics" in summary else {}), **extra},
        "artifact_paths": [str(path) for path in args.artifact],
        "status": args.status,
        "notes": args.notes,
        "environment": summary.get("environment") or registry.environment(PACKAGES),
        # Explicit override for runs that predate per-run SHA capture; None ->
        # captured at registration.
        "repository": json.loads(args.repository_json) if args.repository_json else summary.get("repository"),
    }
    print(registry.write_record(record))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--selection", choices=("tquick", "tfull", "val-select"), required=True)
    run.add_argument("--prompt", choices=sorted(PROMPTS), default="expression")
    run.add_argument("--harness", choices=("suite", "loop"), default="loop")
    run.add_argument("--box-threshold", type=float, default=BASELINE_BOX_THRESHOLD)
    run.add_argument("--text-threshold", type=float, default=BASELINE_TEXT_THRESHOLD)
    run.add_argument("--seed", type=int, default=suite.DEFAULT_SEED)
    run.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    run.add_argument("--archive", type=Path, help="JPEGImages.zip to extract missing images from")
    run.add_argument("--ephemeral-images", action="store_true", help="delete images extracted for this run")
    run.set_defaults(func=command_run)

    sweep_parser = commands.add_parser("sweep")
    sweep_parser.add_argument("--run-dir", type=Path, required=True)
    sweep_parser.set_defaults(func=command_sweep)

    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True, help="run dir or suite report JSON")
    compare.add_argument("--candidate", type=Path, required=True, help="run dir or suite report JSON")
    compare.add_argument("--out", type=Path, required=True)
    compare.set_defaults(func=command_compare)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--run-dir", type=Path, required=True)
    analyze_parser.add_argument("--data-root", type=Path, required=True)
    analyze_parser.add_argument("--archive", type=Path)
    analyze_parser.add_argument("--render-dir", type=Path)
    analyze_parser.set_defaults(func=command_analyze)

    audit = commands.add_parser("audit")
    audit.add_argument("--data-root", type=Path, required=True)
    audit.add_argument("--files", type=Path, nargs="+", required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    register = commands.add_parser("register")
    register.add_argument("--experiment-id", required=True)
    register.add_argument("--title", required=True)
    register.add_argument("--status", choices=sorted(registry.STATUSES), required=True)
    register.add_argument("--notes", required=True)
    register.add_argument("--dataset-audit", type=Path, required=True)
    register.add_argument("--model-revision", required=True)
    register.add_argument("--run-dir", type=Path)
    register.add_argument("--metrics-json", type=Path)
    register.add_argument("--artifact", type=Path, nargs="*", default=[])
    register.add_argument("--split")
    register.add_argument("--seed", type=int)
    register.add_argument("--command")
    register.add_argument("--started-at")
    register.add_argument("--ended-at")
    register.add_argument("--repository-json", help='e.g. {"sha": "...", "dirty": false}')
    register.set_defaults(func=command_register)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
