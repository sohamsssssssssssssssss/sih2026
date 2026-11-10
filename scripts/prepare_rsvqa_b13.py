"""Freeze the approved 6,000-question B13 RSVQA-LR training subset offline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_rsvqa_pilot import TYPES, digest, read_split, select_rows, summary
from training.remote_sensing import load_manifest, sha256_file

POLICY = ROOT / "data/manifests/rsvqa/b13-6k-policy.v1.json"
QUOTAS = {"comp": 2362, "count": 1793, "presence": 1785, "rural_urban": 60}


def build(metadata, images, output, policy_path=POLICY):
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (policy.get("policy_id") != "rsvqa-lr-b13-6k-v1"
            or policy.get("selection_algorithm") != "image-balanced-sha256-v1-exact-quotas"
            or policy.get("seed") != 17 or policy.get("train_qa_limit") != 6000
            or policy.get("train_type_quotas") != QUOTAS
            or policy.get("expected_selected_images") != 558
            or policy.get("max_selected_qa_per_image") != 11
            or policy.get("test_selection_permitted") is not False
            or policy.get("validation_manifest_unchanged") is not True
            or policy.get("parent_exclusion_policy") != "data/manifests/rsvqa/pilot-exclusion-policy.v1.json"):
        raise ValueError("B13 approved selection policy differs")
    parent_path = ROOT / policy["parent_exclusion_policy"]
    if sha256_file(parent_path) != policy["parent_exclusion_policy_sha256"]:
        raise ValueError("Authoritative exclusion policy changed")
    frozen = policy["b12_files_sha256"]
    if any(sha256_file(ROOT / name) != expected for name, expected in frozen.items()):
        raise ValueError("Frozen B12 manifest or provenance changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["test_selection_permitted"] or parent["excluded_image_ids"]["test"]:
        raise ValueError("Test selection must remain forbidden")
    b12 = json.loads((ROOT / "data/manifests/rsvqa/pilot.v1/provenance.json").read_text())
    metadata_hashes = b12["source_metadata_sha256"]
    if len(metadata_hashes) != 9 or any(
            sha256_file(metadata / name) != expected for name, expected in metadata_hashes.items()):
        raise ValueError("Official source metadata differs from frozen B12 provenance")

    eligible = read_split(metadata, "train", parent["excluded_image_ids"]["train"])
    pool = summary(eligible)
    if (any(pool[k] != value for k, value in policy["expected_eligible_train"].items())
            or pool["per_type"] != b12["eligible"]["train"]["per_type"]):
        raise ValueError("Eligible training pool differs")
    selected = select_rows(eligible, "train", policy["seed"], 6000, quotas=QUOTAS)
    counts = Counter(row["question_type"] for row in selected)
    per_image = Counter(row["image_id"] for row in selected)
    question_ids = [row["question_id"] for row in selected]
    if (len(selected) != 6000 or counts != QUOTAS or len(set(question_ids)) != 6000
            or len(per_image) != 558 or max(per_image.values()) > 11
            or any(row["split"] != "train" for row in selected)):
        raise ValueError("B13 selection count, quota, or coverage failed")

    # Protected split IDs are read only after selection; none influence selection.
    protected = {}
    for split in ("val", "test"):
        protected[split] = {}
        for kind in ("images", "questions"):
            source = json.loads((metadata / f"LR_split_{split}_{kind}.json").read_text())
            protected[split][kind] = {int(row["id"]) for row in source[kind] if row.get("active")}
    selected_images = set(per_image)
    if any(selected_images & protected[s]["images"] or set(question_ids) & protected[s]["questions"]
           for s in protected):
        raise ValueError("B13 selection overlaps an official protected split")
    b12_validation = [json.loads(line) for line in
                      (ROOT / "data/manifests/rsvqa/pilot.v1/validation.jsonl").read_text().splitlines()]
    if (selected_images & {row["image_id"] for row in b12_validation}
            or set(question_ids) & {row["question_id"] for row in b12_validation}):
        raise ValueError("B13 selection overlaps frozen B12 validation")

    image_hashes = {}
    for image_id in sorted(selected_images):
        name = f"{image_id}.tif"
        path = images / name
        if not path.is_file():
            raise FileNotFoundError(f"Selected image missing: {name}")
        with Image.open(path) as image:
            image.load()
        measured = sha256_file(path)
        if measured != b12["selected_image_sha256"].get(name):
            raise ValueError(f"Selected image differs from B12 provenance: {name}")
        image_hashes[name] = measured

    manifest_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode()
    provenance = {
        "schema_version": 1, "policy_id": policy["policy_id"],
        "policy_sha256": sha256_file(policy_path),
        "parent_exclusion_policy_sha256": sha256_file(parent_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "selection_algorithm": policy["selection_algorithm"], "seed": policy["seed"],
        "source_metadata_sha256": metadata_hashes,
        "b12_files_sha256": frozen,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_question_ids_in_manifest_order": question_ids,
        "selected_image_ids": sorted(selected_images),
        "selected_image_sha256": image_hashes,
        "image_path_base": "Explicit image_root pointing to official Images_LR",
    }
    provenance["manifest_identity"] = digest(provenance)
    validation = {
        "status": "PASS", "manifest_identity": provenance["manifest_identity"],
        "train_qa": len(selected), "train_images": len(per_image),
        "per_type": {kind: counts[kind] for kind in TYPES},
        "min_qa_per_image": min(per_image.values()),
        "median_qa_per_image": median(per_image.values()),
        "max_qa_per_image": max(per_image.values()),
        "decoded_and_hashed_images": len(image_hashes),
        "official_val_image_and_question_overlap": 0,
        "official_test_image_and_question_overlap": 0,
        "frozen_b12_validation_overlap": 0,
        "frozen_b12_files_unchanged": True,
        "official_metadata_unchanged": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.jsonl").write_bytes(manifest_bytes)
    if len(load_manifest(output / "train.jsonl", images)) != 6000:
        raise ValueError("Written B13 manifest failed the production loader")
    if (any(sha256_file(images / name) != expected for name, expected in image_hashes.items())
            or any(sha256_file(metadata / name) != expected for name, expected in metadata_hashes.items())
            or any(sha256_file(ROOT / name) != expected for name, expected in frozen.items())
            or sha256_file(parent_path) != policy["parent_exclusion_policy_sha256"]):
        raise ValueError("Source or frozen artifact changed during B13 preparation")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return provenance, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    provenance, validation = build(args.metadata_root, args.image_root, args.output_dir)
    print(json.dumps({"manifest_identity": provenance["manifest_identity"], **validation}))


if __name__ == "__main__":
    main()
