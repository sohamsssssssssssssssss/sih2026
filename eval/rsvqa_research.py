"""RSVQA-LR research utilities for the VLM adaptation program.

Experimental research code, not a production capability claim. It provides:

- question metadata keyed by the training manifest's ``sample_id``
  (``"<split>-<question_id>"``, see scripts/prepare_rsvqa_training_manifest.py);
- deterministic question-type-stratified evaluation subsets;
- a spatial leakage audit of the official tile-based split;
- scoring with the explicit metric definitions in ``METRICS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from eval.eval import DEGENERATE_NO_RATE, DEGENERATE_YES_RATE, answer_matches
from training.remote_sensing import sha256_file

FILE_SPLITS = {"train": "train", "validation": "val", "test": "test"}
PATCH_METRES = 256 * 10  # 256 px Sentinel-2 RGB patches at 10 m
EARTH_RADIUS_M = 6378137.0  # EPSG:3857 sphere
COUNT_BINS = ((0, 0), (1, 10), (11, 100), (101, 1000), (1001, math.inf))
CLOSED_ANSWERS = {
    "presence": {"yes", "no"},
    "comp": {"yes", "no"},
    "rural_urban": {"rural", "urban"},
}
NUMBER_WORDS = {
    word: index
    for index, word in enumerate(
        "zero one two three four five six seven eight nine ten".split()
    )
}
METRICS = {
    "strict_accuracy": "prediction.strip().casefold() == gold.strip().casefold(); "
    "identical to the metric of scripts/evaluate_remote_sensing_adapter.py before this change.",
    "lenient_accuracy": "eval.eval.answer_matches(prediction, gold): word-boundary containment, "
    "'1' counts as 'yes', number words zero-ten map to digits. This is the scorer behind the "
    "historical 0.514194 base result; reported only for comparability with it.",
    "invalid_rate": "share of predictions outside the question type's answer space after "
    "normalisation (strip, casefold, trailing '.!' removed): presence/comp must be yes|no, "
    "rural_urban must be rural|urban, count must be a non-negative integer or a number word "
    "zero-ten. Empty output is invalid.",
    "count_bin_accuracy": "count questions only: prediction and gold both parse as integers and "
    "fall in the same bin of 0 | 1-10 | 11-100 | 101-1000 | >1000 (the counting quantisation "
    "described for RSVQA-LR by Lobry et al. 2020). Unparseable predictions are wrong.",
    "type_weighted_strict_accuracy": "sum over question types of strict accuracy on that type "
    "weighted by the type's share of the FULL split; corrects the stratified subset's floor "
    "oversampling of small types.",
    "wilson_95": "Wilson score 95% interval for a proportion.",
    "mcnemar_exact_p": "two-sided exact binomial test on discordant paired outcomes "
    "(base wrong/adapter right vs base right/adapter wrong) over the same sample IDs.",
}


def _metadata(root: Path, split: str, kind: str) -> list[dict]:
    path = root / f"LR_split_{FILE_SPLITS[split]}_{kind}.json"
    return json.loads(path.read_text(encoding="utf-8"))[kind]


def questions(root: Path, split: str) -> list[dict]:
    """Active answered questions on active images, in official file order."""
    answers = {
        int(item["question_id"]): str(item["answer"])
        for item in _metadata(root, split, "answers")
        if item.get("active")
    }
    images = {int(item["id"]) for item in _metadata(root, split, "images") if item.get("active")}
    return [
        {
            "sample_id": f"{split}-{int(item['id'])}",
            "image_id": int(item["img_id"]),
            "type": str(item["type"]),
            "question": str(item["question"]),
            "answer": answers[int(item["id"])],
        }
        for item in _metadata(root, split, "questions")
        if item.get("active") and int(item["id"]) in answers and int(item["img_id"]) in images
    ]


def _rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()


def allocate(type_counts: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Types with at most ``floor`` questions are taken whole; the remaining budget is
    split over the other types in proportion to their size (largest remainder)."""
    if total >= sum(type_counts.values()):
        return dict(type_counts)
    small = {kind: count for kind, count in type_counts.items() if count <= floor}
    large = {kind: count for kind, count in type_counts.items() if count > floor}
    budget = total - sum(small.values())
    if budget < len(large):
        raise ValueError("total is too small for the requested floor")
    quotas = {kind: budget * count / sum(large.values()) for kind, count in large.items()}
    allocation = {kind: int(quota) for kind, quota in quotas.items()}
    by_remainder = sorted(large, key=lambda kind: (allocation[kind] - quotas[kind], kind))
    for kind in by_remainder[: budget - sum(allocation.values())]:
        allocation[kind] += 1
    return {**small, **allocation}


def stratified_subset(
    root: Path,
    split: str,
    total: int | None,
    floor: int,
    seed: int,
    exclude_image_ids: frozenset[int] = frozenset(),
    flagged_image_ids: frozenset[int] = frozenset(),
) -> dict:
    """Deterministic subset; ``total=None`` keeps the whole split (the locked test run).

    ``exclude_image_ids`` drops images before selection; ``flagged_image_ids`` keeps them
    but records them so evaluation can also report a footprint-disjoint number.
    """
    all_rows = questions(root, split)
    full_counts = dict(sorted(Counter(row["type"] for row in all_rows).items()))
    rows = [row for row in all_rows if row["image_id"] not in exclude_image_ids]
    eligible_counts = dict(sorted(Counter(row["type"] for row in rows).items()))
    allocation = eligible_counts if total is None else allocate(eligible_counts, total, floor)
    ranked = sorted(rows, key=lambda row: _rank(seed, row["sample_id"]))
    selected = [
        row
        for kind in sorted(allocation)
        for row in [item for item in ranked if item["type"] == kind][: allocation[kind]]
    ]
    selected.sort(key=lambda row: _rank(seed, row["sample_id"]))
    return {
        "schema_version": 1,
        "dataset": "RSVQA-LR",
        "source": "Zenodo record 6344334",
        "split": split,
        "seed": seed,
        "selection": (
            "whole split"
            if total is None
            else f"total={total}; types with <= {floor} questions taken whole; remaining "
            "budget proportional to type size (largest remainder); within a type, "
            "ascending sha256(f'{seed}\\0{sample_id}')"
        ),
        "full_split_type_counts": full_counts,
        "excluded_image_ids": sorted(exclude_image_ids),
        "leakage_flagged_image_ids": sorted(flagged_image_ids),
        "type_counts": dict(sorted(Counter(row["type"] for row in selected).items())),
        "samples": [
            {"sample_id": row["sample_id"], "type": row["type"], "image_id": row["image_id"]}
            for row in selected
        ],
    }


# --- spatial leakage audit -------------------------------------------------

_NAME = re.compile(
    r"(?P<platform>S2[AB])_MSI(?P<level>L1C|L2A)_(?P<date>\d{8})T\d+_N\d+_R\d+_"
    r"T(?P<tile>\d{2}[A-Z]{3})_\d+T\d+_(?P<row>\d+)-(?P<col>\d+)\.tif$"
)


def mercator_scale(y: float) -> float:
    """sec(latitude) at an EPSG:3857 northing: 3857 units per ground metre."""
    latitude = 2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2
    return 1 / math.cos(latitude)


def footprints(root: Path) -> list[dict]:
    """Every active image's footprint as an axis-aligned EPSG:3857 box.

    ``upperleft_map_x/y`` are EPSG:3857 coordinates, not UTM: within each source tile
    they are exactly linear in the file name's ``<col>-<row>`` pixel offsets with a
    per-tile scale of 10 m x sec(latitude) (docs/research/rsvqa-lr-audit.md).
    """
    rows = []
    for split in FILE_SPLITS:
        for image in _metadata(root, split, "images"):
            if not image.get("active"):
                continue
            match = _NAME.match(image["original_name"])
            if match is None:
                raise ValueError(f"Unparsed RSVQA-LR source name: {image['original_name']}")
            x, y = float(image["upperleft_map_x"]), float(image["upperleft_map_y"])
            extent = PATCH_METRES * mercator_scale(y)
            rows.append(
                {
                    "split": split,
                    "image_id": int(image["id"]),
                    "tile": match["tile"],
                    "date": match["date"],
                    "level": match["level"],
                    "box": (x, y - extent, x + extent, y),
                }
            )
    return rows


def _overlap_fraction(a: tuple, b: tuple) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / ((a[2] - a[0]) * (a[3] - a[1]))


def _centre_distance(a: tuple, b: tuple) -> float:
    """Approximate ground metres: 3857 distance divided by the mean-northing scale."""
    ya, yb = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
    planar = math.dist(((a[0] + a[2]) / 2, ya), ((b[0] + b[2]) / 2, yb))
    return planar / mercator_scale((ya + yb) / 2)


def pixel_agreement(root: Path, held: dict, reference: dict, radius: int = 15) -> dict:
    """Grey-level correlation of the predicted overlap region, searched over +/- ``radius``
    px because the footprint model carries a few pixels of reprojection error. A clear
    peak far above the window median means the two patches image the same ground."""
    import numpy as np
    from PIL import Image

    def grey(image_id: int):
        with Image.open(root / "Images_LR" / f"{image_id}.tif") as image:
            return np.asarray(image.convert("L"), dtype=float)

    a, b = grey(held["image_id"]), grey(reference["image_id"])
    pixel = PATCH_METRES * mercator_scale(held["box"][3]) / 256
    base_dx = round((reference["box"][0] - held["box"][0]) / pixel)
    base_dy = round((held["box"][3] - reference["box"][3]) / pixel)
    scores = []
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            dx, dy = base_dx + shift_x, base_dy + shift_y
            y0, x0, y1, x1 = max(0, dy), max(0, dx), min(256, 256 + dy), min(256, 256 + dx)
            if y1 - y0 < 16 or x1 - x0 < 16:
                continue
            p, q = a[y0:y1, x0:x1].ravel(), b[y0 - dy : y1 - dy, x0 - dx : x1 - dx].ravel()
            scores.append((float(np.corrcoef(p, q)[0, 1]), shift_x, shift_y, p.size))
    if not scores:
        return {"peak_correlation": None}
    peak = max(scores)
    return {
        "peak_correlation": round(peak[0], 4),
        "peak_shift_px": [peak[1], peak[2]],
        "peak_region_pixels": peak[3],
        "window_median_correlation": round(float(np.median([item[0] for item in scores])), 4),
    }


def spatial_leakage_audit(root: Path, verify_pixels: bool = True) -> dict:
    # ponytail: O(n^2) over 772 boxes; use an R-tree if this is reused on BigEarthNet-scale data.
    rows = footprints(root)
    by_split = {split: [row for row in rows if row["split"] == split] for split in FILE_SPLITS}
    questions_per_image = {
        split: Counter(row["image_id"] for row in questions(root, split)) for split in FILE_SPLITS
    }

    def against(held: list[dict], reference: list[dict]) -> dict:
        overlaps, distances, pairs = [], [], []
        for row in held:
            fractions = [(_overlap_fraction(row["box"], ref["box"]), ref) for ref in reference]
            overlaps.append(max(fraction for fraction, _ in fractions))
            distances.append(min(_centre_distance(row["box"], ref["box"]) for ref in reference))
            pairs += [
                {
                    "held_image_id": row["image_id"],
                    "reference_image_id": ref["image_id"],
                    "reference_tile_date": f"{ref['tile']}@{ref['date']}",
                    "overlap_fraction": round(fraction, 4),
                    **(pixel_agreement(root, row, ref) if verify_pixels else {}),
                }
                for fraction, ref in fractions
                if fraction > 0
            ]
        distances.sort()
        overlapping = sorted({pair["held_image_id"] for pair in pairs})
        split = held[0]["split"]
        return {
            "overlapping_pairs": pairs,
            "overlapping_image_ids": overlapping,
            "questions_on_overlapping_images": sum(
                questions_per_image[split][image_id] for image_id in overlapping
            ),
            "questions_in_split": sum(questions_per_image[split].values()),
            "images": len(held),
            "images_with_any_footprint_overlap": len(overlapping),
            "images_with_overlap_ge_50pct": sum(value >= 0.5 for value in overlaps),
            "max_overlap_fraction": round(max(overlaps), 4),
            "nearest_centre_distance_m": {
                "min": round(distances[0], 1),
                "median": round(distances[len(distances) // 2], 1),
                "max": round(distances[-1], 1),
            },
            "images_with_reference_within_5km": sum(value <= 5000 for value in distances),
            "images_with_reference_within_10km": sum(value <= 10000 for value in distances),
        }

    train_by_location = Counter(row["box"] for row in by_split["train"])
    return {
        "schema_version": 1,
        "method": (
            "Each active image's footprint is the EPSG:3857 box from upperleft_map_x/y with "
            "side 2560 m x sec(latitude) (256 px at 10 m). Overlap fraction is relative to the "
            "held-out patch area; centre distances are approximate ground metres (3857 distance "
            "/ sec(mean latitude)). Every overlapping pair is checked in pixels with "
            "pixel_agreement(): grey-level correlation over the predicted overlap, searched "
            "+/-15 px."
        ),
        "tiles_by_split": {
            split: dict(sorted(Counter(f"{row['tile']}@{row['date']}/{row['level']}" for row in items).items()))
            for split, items in by_split.items()
        },
        "validation_vs_train": against(by_split["validation"], by_split["train"]),
        "test_vs_train": against(by_split["test"], by_split["train"]),
        "test_vs_validation": against(by_split["test"], by_split["validation"]),
        "train_locations_repeated_across_dates": sum(
            count > 1 for count in train_by_location.values()
        ),
    }


# --- scoring ---------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold()).rstrip(".!").strip()


def parse_count(text: str) -> int | None:
    value = _normalise(text)
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _count_bin(value: int) -> int:
    return next(index for index, (low, high) in enumerate(COUNT_BINS) if low <= value <= high)


def score(prediction: str, gold: str, question_type: str) -> dict:
    normalised = _normalise(prediction)
    if question_type == "count":
        predicted = parse_count(prediction)
        valid = predicted is not None
        count_bin = valid and _count_bin(predicted) == _count_bin(int(gold))
    else:
        valid = normalised in CLOSED_ANSWERS[question_type]
        count_bin = None
    return {
        "strict": prediction.strip().casefold() == gold.strip().casefold(),
        "lenient": answer_matches(prediction, gold),
        "valid": valid,
        "count_bin": count_bin,
    }


def wilson_95(successes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 6), round(centre + half, 6)]


def _rate(rows: list[dict], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def summarise(rows: list[dict], full_split_type_counts: dict[str, int] | None = None) -> dict:
    """``rows`` carry ``type``, ``prediction`` and the ``score()`` fields."""
    per_type = {}
    for kind in sorted({row["type"] for row in rows}):
        typed = [row for row in rows if row["type"] == kind]
        per_type[kind] = {
            "n": len(typed),
            "strict_accuracy": _rate(typed, "strict"),
            "strict_wilson_95": wilson_95(sum(row["strict"] for row in typed), len(typed)),
            "lenient_accuracy": _rate(typed, "lenient"),
            "invalid_rate": 1 - _rate(typed, "valid"),
        }
        if kind == "count":
            per_type[kind]["count_bin_accuracy"] = _rate(typed, "count_bin")
    binary = [row for row in rows if row["type"] in ("presence", "comp")]
    yes_rate = (
        sum(_normalise(row["prediction"]) == "yes" for row in binary) / len(binary)
        if binary
        else None
    )
    summary = {
        "n": len(rows),
        "strict_accuracy": _rate(rows, "strict"),
        "strict_wilson_95": wilson_95(sum(row["strict"] for row in rows), len(rows)),
        "lenient_accuracy": _rate(rows, "lenient"),
        "invalid_rate": 1 - _rate(rows, "valid") if rows else None,
        "pred_yes_rate_on_binary": yes_rate,
        "degenerate_binary": yes_rate is not None
        and not DEGENERATE_NO_RATE <= yes_rate <= DEGENERATE_YES_RATE,
        "per_type": per_type,
    }
    if full_split_type_counts:
        total = sum(full_split_type_counts.values())
        summary["type_weighted_strict_accuracy"] = sum(
            per_type[kind]["strict_accuracy"] * count / total
            for kind, count in full_split_type_counts.items()
            if kind in per_type
        )
    return summary


def paired_comparison(base: list[dict], adapted: list[dict]) -> dict:
    """Paired strict-accuracy comparison over identical sample IDs."""
    base_by_id = {row["sample_id"]: row["strict"] for row in base}
    adapted_by_id = {row["sample_id"]: row["strict"] for row in adapted}
    if base_by_id.keys() != adapted_by_id.keys():
        raise ValueError("paired comparison needs identical sample IDs")
    gained = sum(adapted_by_id[key] and not base_by_id[key] for key in base_by_id)
    lost = sum(base_by_id[key] and not adapted_by_id[key] for key in base_by_id)
    discordant = gained + lost
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / 2**discordant
    return {
        "n": len(base_by_id),
        "strict_accuracy_delta": (gained - lost) / len(base_by_id),
        "base_wrong_adapter_right": gained,
        "base_right_adapter_wrong": lost,
        "mcnemar_exact_p": min(1.0, 2 * tail) if discordant else 1.0,
    }


def rescore_prior(root: Path, prior: Path) -> dict:
    """Re-score the stored historical test predictions (no inference) under ``METRICS``.

    The historical run evaluated every active test question in official file order,
    so row i is question i; the question text is checked for every row.
    """
    report = json.loads(prior.read_text(encoding="utf-8"))
    rows = questions(root, "test")
    if len(rows) != len(report["results"]):
        raise ValueError("historical result does not cover the full test split")
    scored = []
    for row, result in zip(rows, report["results"]):
        if row["question"] != result["question"] or row["answer"] != result["expected_answer"]:
            raise ValueError(f"historical result misaligned at {row['sample_id']}")
        prediction = result["prediction"]["answer"]
        scored.append(
            {"sample_id": row["sample_id"], "type": row["type"], "prediction": prediction}
            | score(prediction, row["answer"], row["type"])
        )
    counts = dict(Counter(row["type"] for row in rows))
    return {
        "source_result": prior.name,
        "source_git_sha": report.get("git_sha"),
        "source_gpu": report.get("gpu"),
        "metrics": METRICS,
        "summary": summarise(scored, counts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    subset = commands.add_parser("subset")
    subset.add_argument("--split", choices=tuple(FILE_SPLITS), required=True)
    subset.add_argument("--total", type=int, help="omit to keep the whole split")
    subset.add_argument("--floor", type=int, default=100)
    subset.add_argument("--seed", type=int, default=26167)
    subset.add_argument("--audit", type=Path, help="spatial audit JSON from the audit command")
    subset.add_argument(
        "--leakage", choices=("exclude", "flag"), default="flag",
        help="exclude train-overlapping images, or keep them flagged for a sensitivity number",
    )
    audit = commands.add_parser("audit")
    rescore = commands.add_parser("rescore-prior")
    rescore.add_argument("--prior", type=Path, required=True)
    for command in (subset, audit, rescore):
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "subset":
        overlapping = frozenset()
        if args.audit and args.split != "train":
            audit_report = json.loads(args.audit.read_text(encoding="utf-8"))
            overlapping = frozenset(audit_report[f"{args.split}_vs_train"]["overlapping_image_ids"])
        payload = stratified_subset(
            args.dataset_root, args.split, args.total, args.floor, args.seed,
            exclude_image_ids=overlapping if args.leakage == "exclude" else frozenset(),
            flagged_image_ids=overlapping if args.leakage == "flag" else frozenset(),
        )
        if args.audit:
            payload["audit_sha256"] = sha256_file(args.audit)
    elif args.command == "audit":
        payload = spatial_leakage_audit(args.dataset_root)
    else:
        payload = rescore_prior(args.dataset_root, args.prior)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; research artifacts are never overwritten")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out} sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
