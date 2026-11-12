"""Synthetic data/scoring tests only; never call the CUDA evaluator entry point."""
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from kaggle import eval_count_detection as evaluation
from kaggle.run_grounding_dior_rsvg import _find_dataset_root


def test_count_metrics():
    assert evaluation.metrics([{"expected": 2, "prediction": 2}, {"expected": 4, "prediction": 1}]) == {
        "n": 2, "accuracy": 0.5, "mean_absolute_error": 1.5, "within_one": 0.5}
    with pytest.raises(ValueError):
        evaluation.metrics([])


def test_manifest_hash_is_required_before_lr_rows_are_used(tmp_path):
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text('{"question_type":"count"}')
    with pytest.raises(ValueError, match="frozen"):
        evaluation.count_rows(manifest)


def dataset(tmp_path):
    (tmp_path / "ImageSets/Main").mkdir(parents=True)
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "JPEGImages").mkdir()
    for split, ids in {"train": ["tr1", "tr2", "tr3"], "val": ["v1", "v2", "v3", "v4"], "test": ["no_test_annotations"]}.items():
        (tmp_path / "ImageSets/Main" / f"{split}.txt").write_text("\n".join(ids))
    for image_id, classes in {"tr1": ["ship", "ship"], "tr2": ["ship", "ship"], "tr3": [],
                              "v1": ["ship"], "v2": ["vehicle"], "v3": ["ship"], "v4": ["vehicle"]}.items():
        xml = ET.Element("annotation")
        ET.SubElement(xml, "filename").text = f"{image_id}.jpg"
        size = ET.SubElement(xml, "size")
        ET.SubElement(size, "width").text = "800"
        ET.SubElement(size, "height").text = "800"
        for cls in classes:
            obj = ET.SubElement(xml, "object")
            ET.SubElement(obj, "name").text = cls
            ET.SubElement(obj, "difficult").text = "1"  # Count every annotated instance.
            box = ET.SubElement(obj, "bndbox")
            for name, value in zip(("xmin", "ymin", "xmax", "ymax"), (1, 1, 20, 20)):
                ET.SubElement(box, name).text = str(value)
        ET.ElementTree(xml).write(tmp_path / "Annotations" / f"{image_id}.xml")
        (tmp_path / "JPEGImages" / f"{image_id}.jpg").touch()
    return tmp_path


def test_dior_layout_disjoint_splits_counts_baselines_and_frozen_balancing(tmp_path):
    root = dataset(tmp_path)
    assert _find_dataset_root(root, "ImageSets/Main") == root
    splits = evaluation.read_splits(root)
    train = evaluation.annotations(root, splits["train"])
    val = evaluation.annotations(root, splits["val"])
    assert evaluation.majority_counts(train, ["ship", "vehicle"]) == {"ship": 2, "vehicle": 0}
    rows = evaluation.sample_pairs(val, ["ship", "vehicle"], 8)
    assert rows == evaluation.sample_pairs(val, ["ship", "vehicle"], 8)
    assert len({(r["image_id"], r["class"]) for r in rows}) == 8
    assert Counter((r["class"], r["expected"] == 0) for r in rows) == {
        ("ship", True): 2, ("ship", False): 2, ("vehicle", True): 2, ("vehicle", False): 2}
    assert all(r["image_id"] in splits["val"] for r in rows)
    manifest = tmp_path / "frozen.json"
    content = {"seed": 17, "pairs": rows}
    digest = evaluation.freeze_manifest(manifest, content)
    assert digest == evaluation.freeze_manifest(manifest, content)
    with pytest.raises(ValueError, match="Frozen"):
        evaluation.freeze_manifest(manifest, {"seed": 18, "pairs": rows})
    (root / "ImageSets/Main/train.txt").write_text("v1")
    with pytest.raises(ValueError, match="leakage"):
        evaluation.read_splits(root)


def test_bad_layout_missing_annotations_and_insufficient_pairs_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="Found tree"):
        evaluation.read_splits(tmp_path)
    root = dataset(tmp_path)
    (root / "Annotations/v1.xml").unlink()
    with pytest.raises(ValueError, match="Missing exhaustive"):
        evaluation.annotations(root, ["v1"])
    with pytest.raises(ValueError, match="Insufficient"):
        evaluation.sample_pairs({"a": {"counts": {"ship": 1}, "filename": "a.jpg", "xml_sha256": "synthetic"}}, ["ship"], 2)


def test_lr_negative_control_checks_gate_only(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(evaluation, "count_rows", lambda _: [
        {"sample_id": "synthetic-zero", "question": "How many buildings?", "answer": "0"},
        {"sample_id": "synthetic-many", "question": "What is the number of ships?", "answer": "20"}])
    result = evaluation.lr_negative_control(SimpleNamespace(manifest=Path("unused"), config=evaluation.CONFIG))
    assert result["n"] == 2 and result["ground_truth_zero_count"] == 1
    assert all(row["count_status"] == "not_resolvable" for row in result["gates"])
    assert "metrics" not in result
