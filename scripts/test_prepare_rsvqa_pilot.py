"""Synthetic RGB fixtures only; no models, training, or network access."""
from collections import Counter
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_rsvqa_pilot import build, digest, read_split, select_rows, TYPES
from training.remote_sensing import load_manifest, sha256_file


@pytest.fixture
def source(tmp_path):
    metadata = tmp_path / "metadata"
    images = tmp_path / "Images_LR"
    metadata.mkdir()
    images.mkdir()
    for split, ids in (("train", [1, 2, 3]), ("val", [10, 11])):
        questions, answers = [], []
        for i in ids:
            Image.new("RGB", (4, 4), (i, 20, 30)).save(images / f"{i}.tif")
            for n, kind in enumerate(TYPES):
                qid = i * 10 + n
                questions.append({"id": qid, "img_id": i, "type": kind,
                                  "question": f"Synthetic {kind} {i}?", "active": True})
                answers.append({"question_id": qid, "answer": "yes", "active": True})
        questions.append({"id": 999, "active": False})
        for kind, rows in (("images", [{"id": i, "active": True} for i in ids]),
                           ("questions", questions), ("answers", answers)):
            (metadata / f"LR_split_{split}_{kind}.json").write_text(json.dumps({kind: rows}))
    # Test metadata is hashed but test questions must never be parsed/selected.
    for kind in ("images", "questions", "answers"):
        (metadata / f"LR_split_test_{kind}.json").write_text("test must not be parsed")
    policy = {"policy_id": "synthetic-v1", "excluded_image_ids": {
        "train": [3], "validation": [11], "test": []},
        "expected_remaining": {"train": {"images": 2, "qa": 8}, "validation": {"images": 1, "qa": 4}},
        "test_selection_permitted": False, "seed": 17,
        "train_qa_limit": 4, "validation_per_type_limit": 2}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return metadata, images, policy_path


def test_exclusions_reproducibility_types_hashes_and_loader(source, tmp_path):
    metadata, images, policy = source
    before = {p: p.read_bytes() for p in metadata.iterdir()}
    first = build(metadata, images, policy, tmp_path / "one")
    second = build(metadata, images, policy, tmp_path / "two")
    assert first == second
    for name in ("train.jsonl", "validation.jsonl", "pilot.jsonl", "provenance.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    rows = [json.loads(line) for line in (tmp_path / "one/pilot.jsonl").read_text().splitlines()]
    assert len({r["question_id"] for r in rows}) == len(rows)
    assert {r["image_id"] for r in rows} == {1, 2, 10}
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    assert len({r["image_id"] for r in train[:2]}) == 2
    assert {r["image_id"] for r in train}.isdisjoint(r["image_id"] for r in val)
    originals = read_split(metadata, "train", []) + read_split(metadata, "validation", [])
    by_id = {r["question_id"]: r for r in originals}
    for r in rows:
        assert r == by_id[r["question_id"]]
        assert r["question"] == r["instruction"] and r["answer"] == r["response"]
        assert not Path(r["image"]).is_absolute()
    assert first["selected"]["validation"]["per_type"] == dict.fromkeys(TYPES, 1)
    assert first["test_selected_qa"] == 0
    assert len(load_manifest(tmp_path / "one/pilot.jsonl", images)) == 8
    for name, h in first["selected_image_sha256"].items():
        assert sha256_file(images / name) == h
    identity = first.pop("manifest_identity")
    assert digest(first) == identity
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        build(metadata, images, policy, tmp_path / "one")


def test_selection_independent_of_input_order_seed_and_global_image_balance(source):
    metadata, _, _ = source
    rows = read_split(metadata, "train", [3])
    chosen = select_rows(rows, "train", 17, 7)
    assert chosen == select_rows(list(reversed(rows)), "train", 17, 7)
    assert chosen != select_rows(rows, "train", 18, 7)
    for n in range(1, len(chosen) + 1):
        counts = Counter(r["image_id"] for r in chosen[:n])
        assert abs(counts[1] - counts[2]) <= 1
    with pytest.raises(ValueError, match="only"):
        select_rows(rows, "test", 17, 1)


def test_exact_type_quotas_preserve_legacy_selection(source):
    metadata, _, _ = source
    rows = read_split(metadata, "train", [3])
    legacy = select_rows(rows, "train", 17, 4)
    quotas = {"presence": 2, "comp": 1, "count": 1, "rural_urban": 0}
    chosen = select_rows(rows, "train", 17, 4, quotas=quotas)
    assert chosen == select_rows(list(reversed(rows)), "train", 17, 4, quotas=quotas)
    selected_types = Counter(row["question_type"] for row in chosen)
    assert {kind: selected_types[kind] for kind in TYPES} == quotas
    assert Counter(row["image_id"] for row in chosen) == {1: 2, 2: 2}
    assert select_rows(rows, "train", 17, 4) == legacy
    with pytest.raises(ValueError, match="quotas"):
        select_rows(rows, "train", 17, 4, quotas={**quotas, "presence": 3})
    with pytest.raises(ValueError, match="quotas"):
        select_rows(rows, "train", 17, 4, per_type=2, quotas=quotas)


def test_duplicate_qa_ids_fail(source):
    metadata, images, policy = source
    p = metadata / "LR_split_train_questions.json"
    data = json.loads(p.read_text())
    data["questions"].append(data["questions"][0])
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate active question"):
        read_split(metadata, "train", [3])


def test_wrong_remaining_counts_fail_closed(source, tmp_path):
    metadata, images, policy = source
    data = json.loads(policy.read_text())
    data["expected_remaining"]["train"]["qa"] = 999
    policy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Remaining counts"):
        build(metadata, images, policy, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
