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


def dataset(base):
    tmp_path = base / evaluation.DATASET_SUBDIR
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
    train, _ = evaluation.annotations(root, splits["train"])
    val, _ = evaluation.annotations(root, splits["val"])
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
    (root / "Annotations/v1.xml").write_text("<annotation>")  # present but malformed: still a hard stop
    with pytest.raises(ValueError, match="Invalid VOC annotation"):
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


def test_split_files_tolerate_crlf_blanks_and_missing_final_newline(tmp_path):
    root = dataset(tmp_path)
    (root / "ImageSets/Main/train.txt").write_text("tr1\r\n\n  tr2  \n\ntr3")
    (root / "ImageSets/Main/val.txt").write_text("v1\nv2\nv3\nv4\n\n")
    splits = evaluation.read_splits(root)
    assert splits["train"] == ["tr1", "tr2", "tr3"] and splits["val"] == ["v1", "v2", "v3", "v4"]


def test_missing_jpeg_or_xml_val_ids_are_dropped_before_sampling_and_recorded(tmp_path):
    root = dataset(tmp_path)
    (root / "JPEGImages/v1.jpg").unlink()
    (root / "Annotations/v2.xml").unlink()
    (root / "JPEGImages/tr3.jpg").unlink()
    val, dropped = evaluation.annotations(root, ["v1", "v2", "v3", "v4"])
    assert sorted(val) == ["v3", "v4"]
    assert dropped == {"missing_xml": ["v2"], "missing_image": ["v1"]}
    assert evaluation.dropped_summary(dropped, 4) == {"missing_xml": ["v2"], "missing_image": ["v1"],
                                                       "n_dropped": 2, "n_requested": 4}
    train, train_dropped = evaluation.annotations(root, ["tr1", "tr2", "tr3"], require_image=False)
    assert sorted(train) == ["tr1", "tr2", "tr3"] and train_dropped == {"missing_xml": [], "missing_image": []}
    rows = evaluation.sample_pairs(val, ["ship", "vehicle"], 4)
    assert {r["image_id"] for r in rows} <= {"v3", "v4"}


def test_expected_voc2007_layout_or_loud_failure(tmp_path):
    root = dataset(tmp_path / "ok")
    assert evaluation.dataset_root(tmp_path / "ok") == root
    wrong = tmp_path / "wrong"
    (wrong / "VOC2007/ImageSets/Main").mkdir(parents=True)
    with pytest.raises(ValueError, match="VOC2007.*Found tree"):
        evaluation.dataset_root(wrong)


def test_archive_hash_and_size_are_pinned(tmp_path):
    archive = tmp_path / "DIOR-VOC.zip"
    archive.write_bytes(b"not the mirror")
    with pytest.raises(ValueError, match="size"):
        evaluation.verify_archive(archive)
    with pytest.raises(ValueError, match="sha256"):
        evaluation.verify_archive(archive, expected_bytes=archive.stat().st_size)
    good = evaluation.sha256(archive)
    assert evaluation.verify_archive(archive, good, archive.stat().st_size) == good
    assert evaluation.MIRROR_SHA256 == "7a386a56ba815333627d89d3b8b372d1af2a07b78955c017530ec62b7f1e0540"
    for cell in ("fetch_dior_cpu.sh", "run_count_dior_gpu.txt"):
        text = (Path(evaluation.__file__).parent / cell).read_text()
        assert evaluation.MIRROR_SHA256 in text and str(evaluation.MIRROR_BYTES) in text
