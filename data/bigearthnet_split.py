"""Deterministic MGRS-blocked split of BigEarthNet caption patches."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from data.bigearthnet import iter_index

ALGORITHM = "sha256-tile-rank-closest-prefix-v1"
SEED = 26167
EVAL_FRACTION = 0.10
EXPECTED_CAPTIONS = 463_932
PATCH_ID = re.compile(
    r"S2[AB]_MSIL2A_[0-9]{8}T[0-9]{6}_N[0-9]{4}_R[0-9]{3}_"
    r"(T(?:0[1-9]|[1-5][0-9]|60)[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V])_"
    r"[0-9]+_[0-9]+"
)


def mgrs_tile(patch_id: str) -> str:
    """Extract a valid Sentinel-2 MGRS tile or reject the entire patch ID."""
    match = PATCH_ID.fullmatch(patch_id) if isinstance(patch_id, str) else None
    if match is None:
        raise ValueError(f"Malformed BigEarthNet patch_id: {patch_id!r}")
    return match.group(1)


def build_split(
    metadata_path: str | Path,
    annotations_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_captions: int = EXPECTED_CAPTIONS,
) -> dict:
    """Write a sorted, reproducible gzip CSV and its JSON provenance sidecar."""
    rows = []
    for batch in iter_index(metadata_path, annotations_path, ("captioning",), include_text=False):
        if (batch["_merge"] != "both").any():
            raise AssertionError("Caption patch_id missing from metadata")
        for patch_id, official_split, country in batch[["patch_id", "split", "country"]].itertuples(index=False, name=None):
            rows.append((patch_id, mgrs_tile(patch_id), official_split, country))

    patch_ids = [row[0] for row in rows]
    if len(rows) != expected_captions or len(set(patch_ids)) != len(rows):
        raise AssertionError("Caption count mismatch or duplicate patch assignments")
    if any(not isinstance(row[2], str) or not row[2] for row in rows):
        raise AssertionError("Missing official metadata split")

    tile_counts = Counter(row[1] for row in rows)
    if len(tile_counts) < 2:
        raise AssertionError("At least two MGRS tiles are required")
    ranked_tiles = sorted(tile_counts, key=lambda tile: (hashlib.sha256(f"{SEED}:{tile}".encode()).hexdigest(), tile))
    target = math.ceil(len(rows) * EVAL_FRACTION)
    eval_tiles = set()
    eval_count = 0
    for tile in ranked_tiles:
        next_count = eval_count + tile_counts[tile]
        if eval_tiles and next_count >= target and target - eval_count < next_count - target:
            break
        eval_tiles.add(tile)
        eval_count = next_count
        if eval_count >= target:
            break

    rows.sort(key=lambda row: row[0])
    train_ids = {row[0] for row in rows if row[1] not in eval_tiles}
    eval_ids = {row[0] for row in rows if row[1] in eval_tiles}
    train_tiles = set(tile_counts) - eval_tiles
    if train_tiles & eval_tiles or train_ids & eval_ids:
        raise AssertionError("Geographic or patch overlap")
    if not train_tiles or not eval_tiles or train_ids | eval_ids != set(patch_ids):
        raise AssertionError("Incomplete caption assignment")
    if len(train_ids) + len(eval_ids) != expected_captions:
        raise AssertionError("Assigned caption count mismatch")

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(("patch_id", "mgrs_tile", "geo_split", "official_split", "country"))
                writer.writerows(
                    (patch_id, tile, "eval" if tile in eval_tiles else "train", split, country)
                    for patch_id, tile, split, country in rows
                )
    os.replace(temporary, manifest_path)

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "algorithm": ALGORITHM,
        "seed": SEED,
        "eval_fraction": EVAL_FRACTION,
        "source_annotation_type": "captioning",
        "source_metadata": str(Path(metadata_path).expanduser().resolve()),
        "source_annotations": str(Path(annotations_path).expanduser().resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "total_patches": len(rows),
        "train_patches": len(train_ids),
        "eval_patches": len(eval_ids),
        "train_mgrs_tiles": len(train_tiles),
        "eval_mgrs_tiles": len(eval_tiles),
        "source_official_split_counts": dict(sorted(Counter(row[2] for row in rows).items())),
    }
    sidecar = manifest_path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BigEarthNet caption MGRS-blocked split")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_split(args.metadata, args.annotations, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
