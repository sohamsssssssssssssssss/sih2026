"""Zero-shot Grounding DINO evaluation on the official DIOR-RSVG test split."""

import argparse
import json
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grounding_dino import GroundingDINOModel  # noqa: E402

OFFICIAL_TEST_SIZE = 7500
DEFAULT_SAMPLE_SIZE = 400
DEFAULT_SEED = 26167
DIOR_CATEGORIES = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway_service_area",
    "expressway_toll_station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)


def _category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _dataset_root(root: Path) -> Path:
    candidates = [root, *(path.parent for path in root.rglob("test.txt"))]
    matches = [
        path
        for path in candidates
        if (path / "test.txt").is_file()
        and (path / "Annotations").is_dir()
        and (path / "JPEGImages").is_dir()
    ]
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected one DIOR-RSVG root under {root}, found {len(unique)}"
        )
    return unique[0]


def load_test_records(
    root: Path,
    *,
    expected_split_size: int = OFFICIAL_TEST_SIZE,
    split_file: str = "test.txt",
) -> tuple[Path, list[dict[str, Any]]]:
    """Reproduce the official loader's sorted-XML, global-object indexing.

    `split_file` selects another official index file (e.g. val.txt); the
    `test_index` key then holds that split's global expression index.
    """
    dataset_root = _dataset_root(root)
    indices = [
        int(line)
        for line in (dataset_root / split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(indices) != expected_split_size or len(set(indices)) != len(indices):
        raise ValueError(
            f"Expected {expected_split_size} unique official {split_file} indices, got {len(indices)}"
        )
    selected_indices = set(indices)
    records: list[dict[str, Any]] = []
    expression_index = 0
    for annotation_path in sorted((dataset_root / "Annotations").rglob("*.xml")):
        annotation = ET.parse(annotation_path).getroot()
        filename = annotation.findtext("filename")
        if not filename:
            raise ValueError(f"Missing filename in {annotation_path}")
        for member in annotation.findall("object"):
            if expression_index in selected_indices:
                box = member.find("bndbox")
                phrase = member.findtext("description") or member.findtext("phrase")
                category = member.findtext("name")
                if box is None or phrase is None or category is None:
                    raise ValueError(f"Invalid object record in {annotation_path}")
                records.append(
                    {
                        "test_index": expression_index,
                        "image_id": Path(filename).stem,
                        "image_path": dataset_root / "JPEGImages" / filename,
                        "expression": phrase.strip(),
                        "category": _category(category),
                        "ground_truth_pixel_xyxy": [
                            float(box.findtext(name, "nan"))
                            for name in ("xmin", "ymin", "xmax", "ymax")
                        ],
                    }
                )
            expression_index += 1
    if len(records) != len(indices):
        raise ValueError(
            f"Resolved {len(records)} of {len(indices)} official test expressions"
        )
    return dataset_root, records


def stratified_sample(
    records: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Select an approximately equal deterministic sample across all 20 classes."""
    if not 1 <= sample_size <= len(records):
        raise ValueError("sample_size must be within the official test split")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    if set(grouped) != set(DIOR_CATEGORIES):
        raise ValueError(
            f"Expected DIOR's 20 categories, got {sorted(grouped)}"
        )

    base, remainder = divmod(sample_size, len(DIOR_CATEGORIES))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for index, category in enumerate(DIOR_CATEGORIES):
        rows = sorted(grouped[category], key=lambda row: row["test_index"])
        random.Random(f"{seed}:{category}").shuffle(rows)
        quota = base + int(index < remainder)
        selected.extend(rows[:quota])
        leftovers.extend(rows[quota:])
    if len(selected) < sample_size:
        random.Random(seed).shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])
    if len(selected) != sample_size:
        raise ValueError(f"Could select only {len(selected)} of {sample_size} samples")
    return sorted(selected, key=lambda row: row["test_index"])


def normalize_xyxy(box: list[float], width: int, height: int) -> list[float]:
    """Convert DIOR-RSVG pixel xyxy ground truth to normalized image-space xyxy."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def iou_xyxy(left: list[float], right: list[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _top_prediction(evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    boxes = [
        item
        for item in evidence
        if item.get("type") == "bounding_box"
        and item.get("coordinate_space") == "normalized_xyxy"
        and isinstance(item.get("coordinates"), list)
        and len(item["coordinates"]) == 4
        and isinstance(item.get("confidence"), (int, float))
    ]
    return max(boxes, key=lambda item: float(item["confidence"]), default=None)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def evaluate(
    data_root: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    git_sha: str | None = None,
    working_tree_sha256: str | None = None,
) -> dict[str, Any]:
    dataset_root, test_records = load_test_records(data_root)
    samples = stratified_sample(test_records, sample_size, seed)
    model = GroundingDINOModel()
    print(
        f"dataset=DIOR-RSVG; split=official test.txt ({len(test_records)} expressions); "
        f"sample={len(samples)}; seed={seed}",
        flush=True,
    )
    print(
        f"model={model.name}; checkpoint={model.version}; "
        f"box_threshold={model.box_threshold}; text_threshold={model.text_threshold}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"DIOR-RSVG image is missing: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        # DIOR-RSVG stores pixels; the frozen provider returns normalized boxes.
        # Convert gold once so every IoU operand uses normalized_xyxy coordinates.
        gold = normalize_xyxy(sample["ground_truth_pixel_xyxy"], width, height)
        prediction = model.infer([str(image_path)], sample["expression"])
        top = _top_prediction(prediction.get("evidence", []))
        overlap = iou_xyxy(top["coordinates"], gold) if top else 0.0
        results.append(
            {
                "test_index": sample["test_index"],
                "image_id": sample["image_id"],
                "image_path": str(image_path.relative_to(dataset_root)),
                "category": sample["category"],
                "referring_expression": sample["expression"],
                "image_size": {"width": width, "height": height},
                "ground_truth": {
                    "coordinates": gold,
                    "coordinate_space": "normalized_xyxy",
                },
                "prediction": prediction,
                "selected_prediction": top,
                "iou": overlap,
                "hit_at_0_5": overlap >= 0.5,
            }
        )
        if index % 25 == 0 or index == len(samples):
            hits = sum(result["hit_at_0_5"] for result in results)
            print(f"progress={index}/{len(samples)}; hits={hits}", flush=True)

    per_category: dict[str, dict[str, Any]] = {}
    for category in DIOR_CATEGORIES:
        rows = [result for result in results if result["category"] == category]
        hits = sum(result["hit_at_0_5"] for result in rows)
        per_category[category] = {
            "n": len(rows),
            "hits": hits,
            "pr_at_0_5": hits / len(rows) if rows else 0.0,
            "mean_iou": sum(result["iou"] for result in rows) / len(rows)
            if rows
            else 0.0,
        }
    hits = sum(result["hit_at_0_5"] for result in results)
    timestamp = datetime.now(timezone.utc).isoformat()
    stamp = timestamp.replace("-", "").replace(":", "").split(".")[0] + "Z"
    report = {
        "run_id": f"grounding-dino-swint__dior-rsvg__{stamp}",
        "git_sha": git_sha or _git_sha(),
        "working_tree_sha256": working_tree_sha256,
        "dataset": {
            "name": "DIOR-RSVG",
            "version": "official test.txt split",
            "official_test_size": len(test_records),
            "n_sampled": len(results),
            "sampling_method": (
                "deterministic category-stratified sampling across the 20 DIOR "
                "classes; equal quota per class with deterministic deficit fill"
            ),
            "sampling_seed": seed,
        },
        "model": {
            "name": model.name,
            "version": model.version,
            "checkpoint": "groundingdino_swint_ogc.pth",
        },
        "configuration": {
            "box_threshold": model.box_threshold,
            "text_threshold": model.text_threshold,
            "prediction_selection": "highest confidence returned bounding box",
            "coordinate_space": "normalized_xyxy",
            "iou_threshold": 0.5,
        },
        "timestamp": timestamp,
        "pr_at_0_5": hits / len(results),
        "mean_iou": sum(result["iou"] for result in results) / len(results),
        "per_category": per_category,
        "results": results,
    }
    print(
        f"Pr@0.5={report['pr_at_0_5']:.6f}; "
        f"mean_IoU={report['mean_iou']:.6f}; n={len(results)}",
        flush=True,
    )
    for category, metrics in per_category.items():
        print(
            f"  {category}: Pr@0.5={metrics['pr_at_0_5']:.6f}; "
            f"mean_IoU={metrics['mean_iou']:.6f}; "
            f"hits={metrics['hits']}/{metrics['n']}",
            flush=True,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--git-sha")
    parser.add_argument("--working-tree-sha256")
    args = parser.parse_args()
    report = evaluate(
        args.data_root,
        sample_size=args.sample_size,
        seed=args.seed,
        git_sha=args.git_sha,
        working_tree_sha256=args.working_tree_sha256,
    )
    output = args.out or ROOT / "results" / f"{report['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
