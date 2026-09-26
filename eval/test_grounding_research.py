from pathlib import Path

import pytest
import torch

from eval.grounding_research import (
    classify_failure,
    compare_rows,
    head_phrase,
    replay,
    summarize,
    wilson,
)
from eval.suites.grounding_dior_rsvg import _top_prediction
from models.grounding_dino import GroundingDINOModel


@pytest.mark.parametrize(
    ("expression", "head"),
    [
        ("The tennis court in the middle", "The tennis court"),
        ("The dam is on the lower right of the slender gray bridge", "The dam"),
        ("The white rounded storage tank", "The white rounded storage tank"),
        ("A storage tank on the right", "A storage tank"),
        ("in the middle", "in the middle"),  # never truncates to empty
    ],
)
def test_head_phrase_drops_first_relational_clause(expression, head):
    assert head_phrase(expression) == head


def test_wilson_matches_protocol_baseline_interval():
    low, high = wilson(65, 400)

    assert round(low, 3) == 0.130 and round(high, 3) == 0.202


@pytest.mark.parametrize("box_threshold", [0.0, 0.3, 0.35, 0.9])
def test_offline_replay_matches_production_provider(tmp_path: Path, box_threshold):
    scores = torch.tensor([0.2, 0.5, 0.36, 0.1])
    boxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.1, 0.9, 0.4, 0.4], [0.5, 0.5, 1.2, 0.3], [0.3, 0.3, 0.1, 0.1]]
    )
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"x")
    provider = GroundingDINOModel(box_threshold=box_threshold)

    def fake_predict(**kwargs):
        mask = scores > kwargs["box_threshold"]
        return boxes[mask], scores[mask], ["label"] * int(mask.sum())

    provider._model, provider._load_image = object(), lambda path: (None, None)
    provider._predict_fn = fake_predict

    online = _top_prediction(provider.infer([str(image)], "a ship")["evidence"])
    box, score, count = replay({"scores": scores, "boxes": boxes}, box_threshold)

    assert count == int((scores > box_threshold).sum())
    assert (online is None and box is None) or (
        online["coordinates"] == box and online["confidence"] == score
    )


def test_summarize_counts_empty_predictions_as_failures():
    rows = [
        {"iou": 0.8, "n_detections": 2},
        {"iou": 0.0, "n_detections": 0},
        {"iou": 0.3, "n_detections": 1},
        {"iou": 0.55, "n_detections": 1},
    ]

    metrics = summarize(rows)

    assert metrics["pr_at_0.5"] == 0.5
    assert metrics["pr_at_0.25"] == 0.75
    assert metrics["no_detection_rate"] == 0.25
    assert metrics["multi_detection_rate"] == 0.25
    assert metrics["selective_pr_at_0.5"] == pytest.approx(2 / 3)


def test_classify_failure_types():
    target = {"category": "ship", "normalized": [0.0, 0.0, 0.2, 0.2]}
    twin = {"category": "ship", "normalized": [0.5, 0.5, 0.7, 0.7]}
    tank = {"category": "storagetank", "normalized": [0.8, 0.0, 1.0, 0.2]}
    objects = [target, twin, tank]

    def row(box, iou):
        selected = None if box is None else {"coordinates": box}
        return {"gold": target["normalized"], "category": "ship", "selected": selected, "iou": iou}

    assert classify_failure(row(None, 0.0), objects) == "no_detection"
    assert classify_failure(row([0.5, 0.5, 0.7, 0.7], 0.0), objects) == "same_class_other_instance"
    assert classify_failure(row([0.8, 0.0, 1.0, 0.2], 0.0), objects) == "other_class_object"
    assert classify_failure(row([0.0, 0.0, 0.4, 0.4], 0.25), objects) == "poor_localization_of_target"
    assert classify_failure(row([0.3, 0.8, 0.4, 0.9], 0.0), objects) == "background_or_unannotated"


def test_compare_rows_reports_hit_flips_and_deltas():
    box = {"coordinates": [0, 0, 1, 1], "confidence": 0.5}
    reference = [
        {"test_index": 1, "iou": 0.6, "selected": box, "n_detections": 1},
        {"test_index": 2, "iou": 0.0, "selected": None, "n_detections": 0},
    ]
    candidate = [
        {"test_index": 1, "iou": 0.4, "selected": {**box, "confidence": 0.4}, "n_detections": 1},
        {"test_index": 2, "iou": 0.0, "selected": None, "n_detections": 0},
    ]

    result = compare_rows(reference, candidate)

    assert result["hit_flips"] == [1]
    assert result["bit_identical_selected"] == 1
    assert result["max_abs_iou_delta"] == pytest.approx(0.2)
    assert result["max_abs_confidence_delta"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        compare_rows(reference, candidate[::-1])
