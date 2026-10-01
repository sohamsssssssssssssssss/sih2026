"""Zero-shot threshold sweep for Grounding DINO on DIOR-RSVG (no training).

Captures raw Grounding DINO queries (boxes + full token-logit rows) once per
image and re-scores every (box_threshold, text_threshold) combination offline,
so a 12-combination sweep costs exactly one forward pass per expression
(400 forwards total) instead of one per combination (4,800).

Score reconstruction is bit-faithful to the Step 3 suite: for each combination
the kept-query mask and per-query confidence are derived exactly like the
official ``groundingdino.util.inference.predict`` (max token logit vs
box_threshold), and labels are re-derived per combination exactly like the
official phrase construction (logit > text_threshold via
``get_phrases_from_posmap``). text_threshold affects only the label strings,
never which boxes are kept or their confidence, so the selected
highest-confidence prediction (coordinates + confidence) is identical across
text_threshold values at a fixed box_threshold and the metrics coincide.

Run with the official checkpoint on a CUDA GPU (Kaggle T4) using
``kaggle/run_grounding_threshold_sweep.py``.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites.grounding_dior_rsvg import (  # noqa: E402
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    DIOR_CATEGORIES,
    iou_xyxy,
    load_test_records,
    normalize_xyxy,
    stratified_sample,
)

WEAK_CATEGORIES = (
    "dam",
    "tenniscourt",
    "windmill",
    "baseballfield",
    "expressway_toll_station",
    "harbor",
    "overpass",
)
BOX_GRID = (0.20, 0.25, 0.30, 0.35)
TEXT_GRID = (0.15, 0.20, 0.25)

RAW_LOGITS_LIMIT = 256


def _shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def capture_raw_queries(
    model: Any, image: Any, caption: str, *, device: str = "cuda"
) -> dict[str, Any]:
    """Run the backbone once and return raw per-query data before thresholding.

    Reproduces the official ``predict`` preprocessing/forward exactly, but
    returns every query so all threshold combinations can be re-scored
    offline.
    """
    import torch

    from groundingdino.util.inference import preprocess_caption

    original_query = caption.strip()
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    tokenized = model.tokenizer(caption)
    rows: list[dict[str, Any]] = []
    for logits, box in zip(
        prediction_logits.tolist(), prediction_boxes.tolist(), strict=True
    ):
        rows.append(
            {
                "confidence": max(logits),  # official per-query score
                "box_cxcywh": [float(value) for value in box],
                "logits": logits,
            }
        )
    return {
        "caption": caption,
        "query": original_query,
        "tokenized": tokenized,
        "rows": rows,
    }


def _default_phrase_fn() -> Any:
    """Official phrase construction: posmap decode via get_phrases_from_posmap."""
    import torch
    from groundingdino.util.utils import get_phrases_from_posmap

    def phrase_fn(
        logit_values: list[float],
        tokenized: dict[str, Any],
        tokenizer: Any,
        text_threshold: float,
    ) -> str:
        posmap = [value > text_threshold for value in logit_values]
        posmap_tensor = torch.BoolTensor(posmap)
        return get_phrases_from_posmap(
            posmap_tensor, tokenized, tokenizer
        ).replace(".", "")

    return phrase_fn


def rescore_combination(
    raw: dict[str, Any],
    tokenizer: Any,
    box_threshold: float,
    text_threshold: float,
    phrase_fn: Any | None = None,
) -> dict[str, Any]:
    """Re-score one raw capture at one threshold combination, official-style.

    ``phrase_fn`` defaults to the official label construction; tests inject a
    CPU-only reference implementation with identical semantics.
    """
    if phrase_fn is None:
        phrase_fn = _default_phrase_fn()
    rows = raw["rows"]
    # Official keep-mask: max token logit must exceed box_threshold.
    kept = [
        index
        for index, row in enumerate(rows)
        if row["confidence"] > box_threshold
    ]
    phrases = [
        phrase_fn(rows[index]["logits"], raw["tokenized"], tokenizer, text_threshold)
        for index in kept
    ]
    evidence = []
    for index, phrase in zip(kept, phrases, strict=True):
        row = rows[index]
        center_x, center_y, width, height = row["box_cxcywh"]
        coordinates = [
            max(0.0, min(1.0, center_x - width / 2)),
            max(0.0, min(1.0, center_y - height / 2)),
            max(0.0, min(1.0, center_x + width / 2)),
            max(0.0, min(1.0, center_y + height / 2)),
        ]
        evidence.append(
            {
                "type": "bounding_box",
                "label": phrase,
                "coordinates": coordinates,
                "coordinate_space": "normalized_xyxy",
                "confidence": row["confidence"],
                "source_scene_id": None,
            }
        )
    count = len(evidence)
    query = raw.get("query") or raw["caption"].rstrip(".").strip()
    answer = (
        f"No match found for '{query}'."
        if count == 0
        else f"Found {count} match{'es' if count != 1 else ''} for '{query}'."
    )
    return {"answer": answer, "evidence": evidence}


def evaluate(data_root: Path, *, sample_size: int, seed: int) -> dict[str, Any]:
    """Capture raw queries once per expression, then re-score every combination."""
    from groundingdino.util.inference import load_image, load_model
    from huggingface_hub import hf_hub_download

    from models.grounding_dino.model import GroundingDINOModel

    dataset_root, test_records = load_test_records(data_root)
    samples = stratified_sample(test_records, sample_size, seed)
    provider = GroundingDINOModel()
    print(
        f"dataset=DIOR-RSVG; split=official test.txt ({len(test_records)} expressions); "
        f"sample={len(samples)}; seed={seed}",
        flush=True,
    )
    print(
        f"model={provider.name}; checkpoint={provider.version}; "
        f"box_grid={list(BOX_GRID)}; text_grid={list(TEXT_GRID)}",
        flush=True,
    )

    config_path = (
        Path(__import__("groundingdino").__file__).resolve().parent
        / "config"
        / "GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_path = hf_hub_download(
        repo_id="ShilongLiu/GroundingDINO",
        filename="groundingdino_swint_ogc.pth",
    )
    backbone = load_model(str(config_path), checkpoint_path, device="cuda")

    captures: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"DIOR-RSVG image is missing: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        gold = normalize_xyxy(sample["ground_truth_pixel_xyxy"], width, height)
        _, image_tensor = load_image(str(image_path))
        raw = capture_raw_queries(
            backbone, image_tensor, sample["expression"], device="cuda"
        )
        captures.append(
            {
                "sample": sample,
                "image_size": {"width": width, "height": height},
                "gold": gold,
                "raw": raw,
            }
        )
        if index % 25 == 0 or index == len(samples):
            print(f"capture progress={index}/{len(samples)}", flush=True)

    tokenizer = backbone.tokenizer
    combinations: list[dict[str, Any]] = []
    for box_threshold in BOX_GRID:
        for text_threshold in TEXT_GRID:
            results: list[dict[str, Any]] = []
            for capture in captures:
                prediction = rescore_combination(
                    capture["raw"], tokenizer, box_threshold, text_threshold
                )
                top = _top_prediction(prediction["evidence"])
                overlap = iou_xyxy(top["coordinates"], capture["gold"]) if top else 0.0
                results.append(
                    {
                        "test_index": capture["sample"]["test_index"],
                        "image_id": capture["sample"]["image_id"],
                        "category": capture["sample"]["category"],
                        "iou": overlap,
                        "hit_at_0_5": overlap >= 0.5,
                        "n_boxes": len(prediction["evidence"]),
                        "top_label": top["label"] if top else None,
                    }
                )
            per_category: dict[str, dict[str, Any]] = {}
            for category in DIOR_CATEGORIES:
                rows = [row for row in results if row["category"] == category]
                hits = sum(row["hit_at_0_5"] for row in rows)
                per_category[category] = {
                    "n": len(rows),
                    "hits": hits,
                    "pr_at_0_5": hits / len(rows) if rows else 0.0,
                    "mean_iou": sum(row["iou"] for row in rows) / len(rows)
                    if rows
                    else 0.0,
                }
            hits = sum(row["hit_at_0_5"] for row in results)
            combinations.append(
                {
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "pr_at_0_5": hits / len(results),
                    "mean_iou": sum(row["iou"] for row in results) / len(results),
                    "n": len(results),
                    "per_category": per_category,
                }
            )
            print(
                f"combo box={box_threshold:.2f} text={text_threshold:.2f} -> "
                f"Pr@0.5={combinations[-1]['pr_at_0_5']:.4f} "
                f"mean_IoU={combinations[-1]['mean_iou']:.4f}",
                flush=True,
            )

    weak = {
        category: [
            {
                "box_threshold": combo["box_threshold"],
                "text_threshold": combo["text_threshold"],
                "pr_at_0_5": combo["per_category"][category]["pr_at_0_5"],
                "mean_iou": combo["per_category"][category]["mean_iou"],
            }
            for combo in combinations
        ]
        for category in WEAK_CATEGORIES
    }
    return {
        "combinations": combinations,
        "weak_categories": weak,
        "captures_preview": [
            {
                "test_index": capture["sample"]["test_index"],
                "category": capture["sample"]["category"],
                "referring_expression": _shorten(capture["sample"]["expression"]),
                "caption_sent": _shorten(capture["raw"]["caption"]),
                "n_queries": len(capture["raw"]["rows"]),
                "max_confidence": max(
                    (row["confidence"] for row in capture["raw"]["rows"]), default=0.0
                ),
                "logits_preview": [
                    [round(value, 4) for value in row["logits"][:RAW_LOGITS_LIMIT]]
                    for row in capture["raw"]["rows"][:2]
                ],
                "box_cxcywh_preview": [
                    [round(value, 6) for value in row["box_cxcywh"]]
                    for row in capture["raw"]["rows"][:2]
                ],
                "label_note": "labels are re-derived per combination in re-scoring",
            }
            for capture in captures
        ],
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--git-sha")
    parser.add_argument("--working-tree-sha256")
    args = parser.parse_args()

    sweep = evaluate(args.data_root, sample_size=args.sample_size, seed=args.seed)
    timestamp = datetime.now(timezone.utc).isoformat()
    stamp = timestamp.replace("-", "").replace(":", "").split(".")[0] + "Z"
    report = {
        "run_id": f"grounding-dino-swint__dior-rsvg__threshold-sweep__{stamp}",
        "git_sha": args.git_sha or _git_sha(),
        "working_tree_sha256": args.working_tree_sha256,
        "dataset": {
            "name": "DIOR-RSVG",
            "version": "official test.txt split",
            "official_test_size": 7500,
            "n_sampled": args.sample_size,
            "sampling_method": (
                "deterministic category-stratified sampling across the 20 DIOR "
                "classes; equal quota per class with deterministic deficit fill; "
                "identical sample list as the Step 3 baseline run"
            ),
            "sampling_seed": args.seed,
        },
        "model": {
            "name": "grounding-dino-swint",
            "version": "ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth",
            "checkpoint": "groundingdino_swint_ogc.pth",
        },
        "sweep": {
            "kind": "zero-shot threshold sweep (post-processing only, no training)",
            "box_grid": list(BOX_GRID),
            "text_grid": list(TEXT_GRID),
            "baseline_combination": {"box_threshold": 0.35, "text_threshold": 0.25},
            "efficiency": (
                "raw queries captured once per image (400 forwards); all "
                "combinations re-scored offline from the same captures"
            ),
        },
        "configuration": {
            "prediction_selection": "highest confidence returned bounding box",
            "coordinate_space": "normalized_xyxy",
            "iou_threshold": 0.5,
        },
        "timestamp": timestamp,
        **sweep,
    }
    output = (
        args.out
        or ROOT
        / "results"
        / f"{report['run_id']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
