"""CPU-only counting policy; the detector callable owns inference/device selection."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "configs/count_grounding.json"
Detector = Callable[[list[str], str], dict]


@dataclass(frozen=True)
class CountThresholds:
    box_threshold: float
    text_threshold: float
    nms_iou: float

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Count thresholds must be finite numbers between zero and one")


def load_thresholds(path: Path = CONFIG) -> CountThresholds:
    settings = json.loads(path.read_text())
    return CountThresholds(**{key: settings[key] for key in CountThresholds.__dataclass_fields__})


def iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def count_objects(image_paths: list[str], noun: str, detector: Detector,
                  thresholds: CountThresholds) -> dict:
    """Count validated, scored boxes after score filtering and stable greedy NMS."""
    if len(image_paths) != 1 or not noun.strip():
        raise ValueError("Detection counts require one image and an object prompt")
    result = detector(image_paths, noun)
    if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
        raise ValueError("Detector must return an evidence list")
    boxes = []
    for item in result["evidence"]:
        if not isinstance(item, dict) or item.get("type") != "bounding_box" or item.get("coordinate_space") != "normalized_xyxy":
            raise ValueError("Count evidence must use normalized_xyxy bounding boxes")
        coordinates, score = item.get("coordinates"), item.get("confidence")
        if (not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates)
                or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Malformed count box coordinates or score")
        x1, y1, x2, y2 = coordinates
        if x2 < x1 or y2 < y1:
            raise ValueError("Inverted count box coordinates")
        if x2 > x1 and y2 > y1 and score > thresholds.box_threshold:
            boxes.append(item)
    kept = []
    for box in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        if all(iou(box["coordinates"], previous["coordinates"]) <= thresholds.nms_iou for previous in kept):
            kept.append(box)
    settings = asdict(thresholds)
    summary = {"type": "detection_count", "object": noun, "count": len(kept), "thresholds": settings}
    return {**result, "answer": str(len(kept)), "count": len(kept),
            "evidence": [*kept, summary], "count_thresholds": settings,
            "answer_source": "grounding_dino_count"}
