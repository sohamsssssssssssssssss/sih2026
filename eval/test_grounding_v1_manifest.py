from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.grounding_v1_manifest import (
    canonical_parents, choose, validate_disjoint, verify_images, vrs_parent,
)


def row(parent, index):
    return {"parent": parent, "index": index, "source": "fixture", "image": f"{index}.jpg",
            "upstream_split": "test"}


def test_shared_dior_parent_from_vrsbench_cannot_cross_splits():
    dior = row("dior:00001", 1)
    vrs = row(vrs_parent("00001_0003.png"), 2)
    validate_disjoint({"eval": [dior, vrs]})
    with pytest.raises(ValueError, match="dior:00001 assigned to train and eval"):
        validate_disjoint({"train": [dior], "eval": [vrs]})


def test_selection_uses_distinct_parents_and_is_independent_of_source_order():
    rows = [row("dior:00001", 1), row("dior:00001", 2), row("dior:00002", 3),
            row(vrs_parent("P0321_0003.png"), 4)]
    selected = choose(rows, 2, "fixture", {"dior:00002"})
    assert selected == choose(reversed(rows), 2, "fixture", {"dior:00002"})
    assert {item["parent"] for item in selected} == {"dior:00001", "dota-v2:P0321"}


def test_archive_fingerprint_detects_an_image_alias_across_splits():
    images = {("dior-rsvg", "all"): {name: SimpleNamespace(CRC=123, file_size=300) for name in ("1.jpg", "2.jpg")}}
    original = row("dior:00001", 1)
    alias = {**original, "parent": "dior:00002", "image": "2.jpg"}
    with pytest.raises(ValueError, match="fingerprint shared across splits"):
        verify_images({"dior-rsvg": {"train": [original], "eval": [alias]}}, images)


def test_frozen_manifests_match_hashes_and_keep_image_parents_disjoint():
    root = Path(__file__).resolve().parents[1] / "results" / "grounding"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    splits = {"train": [], "select": [], "eval": []}
    for filename, record in manifest["files"].items():
        content = (root / filename).read_bytes()
        rows = [json.loads(line) for line in content.splitlines()]
        assert sha256(content).hexdigest() == record["sha256"]
        assert len(rows) == record["items"]
        splits[Path(filename).stem.rsplit("-", 1)[-1]].extend(rows)

    validate_disjoint(splits)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 10_000, "select": 1_000, "eval": 3_000,
    }


def test_archive_aliases_include_cross_dataset_crops_of_the_aliased_parent():
    copy = SimpleNamespace(CRC=123, file_size=300)
    crop = SimpleNamespace(CRC=124, file_size=200)
    images = {
        ("dior-rsvg", "all"): {"00001.jpg": copy, "00002.jpg": copy},
        ("vrsbench", "test"): {"00002_0000.png": crop},
    }
    aliases = canonical_parents(images)
    assert aliases == {"dior:00001": "dior:00001", "dior:00002": "dior:00001"}
