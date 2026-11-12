"""Counting policy tests use CPU callables; no model or accelerator is loaded."""
from dataclasses import asdict
import json

import pytest

from models.grounding_dino.count import CountThresholds, count_objects, load_thresholds
from models.qwen_vl.b13 import count_target
from models.qwen_vl.model import QwenVLModel


@pytest.mark.parametrize("question,noun", [
    ("How many buildings are there?", "buildings"),
    ("What is the number of ships in the image?", "ships"),
    ("What is the amount of small buildings?", "small buildings"),
    ("number of roads", "roads"), ("amount of grass areas", "grass areas"),
    ("Count the houses!", "houses"),
    ("HOW MANY planes are visible in this scene?", "planes"),
    ("How many roads in this picture?", "roads"),
    ("Are there more buildings than roads?", None), ("How many ?", None),
])
def test_count_noun(question, noun):
    assert count_target(question) == noun


def box(coords, confidence):
    return {"type": "bounding_box", "coordinates": coords,
            "coordinate_space": "normalized_xyxy", "confidence": confidence, "label": "buildings"}


def test_threshold_then_nms_and_evidence_with_injected_detector():
    proposals = [box([0, 0, 0.4, 0.4], 0.8), box([0, 0, 0.4, 0.4], 0.9),
                 box([0.6, 0.6, 1, 1], 0.7), box([0.4, 0, 0.5, 0.1], 0.35),
                 box([0, 0, 0.1, 0.1], 0.2), box([0, 0, 0, 0.2], 0.95)]
    calls = []
    def detector(paths, noun):
        calls.append((paths, noun))
        return {"evidence": proposals}
    settings = CountThresholds(0.35, 0.25, 0.5)
    result = count_objects(["cpu-stub.png"], "buildings", detector, settings)
    assert calls == [(["cpu-stub.png"], "buildings")]
    assert result["answer"] == "2" and result["count"] == 2
    assert result["evidence"][:2] == [proposals[1], proposals[2]]
    assert result["evidence"][-1] == {"type": "detection_count", "object": "buildings",
                                       "count": 2, "thresholds": asdict(settings)}
    assert result["count_thresholds"] == asdict(settings)
    assert result["answer_source"] == "grounding_dino_count"


def test_nms_iou_is_configurable_and_boundary_is_retained():
    proposals = [box([0, 0, 0.75, 1], 0.9), box([0.25, 0, 1, 1], 0.8)]  # IoU exactly 0.5
    detector = lambda *args: {"evidence": proposals}
    assert count_objects(["image"], "roads", detector, CountThresholds(0.35, 0.25, 0.5))["count"] == 2
    assert count_objects(["image"], "roads", detector, CountThresholds(0.35, 0.25, 0.49))["count"] == 1


def test_zero_detections_is_a_verified_zero():
    result = count_objects(["image"], "roads", lambda *args: {"evidence": []}, load_thresholds())
    assert result["count"] == 0 and result["answer"] == "0"
    assert "unverified_count" not in result


@pytest.mark.parametrize("proposal", [
    box([0, 0, 1, 1], float("nan")), box([0, 0, 1, 1], None),
    box([0, 0, 1, 1], 2), box([1, 0, 0, 1], 0.9), box([0, 0, 2, 1], 0.9),
    {"type": "not_a_box"},
])
def test_bad_detector_evidence_is_rejected(proposal):
    with pytest.raises(ValueError):
        count_objects(["image"], "roads", lambda *args: {"evidence": [proposal]}, load_thresholds())


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_thresholds_rejected(value):
    with pytest.raises(ValueError, match="thresholds"):
        CountThresholds(0.35, 0.25, value)


def test_threshold_file_controls_detector_defaults(tmp_path):
    from models.grounding_dino.model import GroundingDINOModel
    path = tmp_path / "count.json"
    path.write_text(json.dumps({"box_threshold": 0.3, "text_threshold": 0.2, "nms_iou": 0.5}))
    assert load_thresholds(path) == CountThresholds(0.3, 0.2, 0.5)
    model = GroundingDINOModel()
    settings = load_thresholds()
    assert (model.box_threshold, model.text_threshold) == (settings.box_threshold, settings.text_threshold)


def test_direct_adapter_count_also_carries_unverified_label(tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_VQA_ADAPTER", str(tmp_path))
    image = tmp_path / "image.png"
    image.touch()
    model = QwenVLModel()
    monkeypatch.setattr(model, "_generate_answer", lambda *args: "12")
    result = model.infer([str(image)], "What is the amount of buildings?")
    assert result["answer"] == "12"
    assert result["unverified_count"] == "detector_unavailable"
