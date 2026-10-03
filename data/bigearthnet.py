"""Batch-wise BigEarthNet.txt annotation index; no imagery access."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ANNOTATION_TYPES = ("captioning", "binary", "mcq", "bounding box")
METADATA_COLUMNS = {"patch_id", "split", "country", "s1_name", "s2v1_name"}
ANNOTATION_COLUMNS = {"patch_id", "type", "input", "output"}


def iter_index(
    metadata_path: str | Path,
    annotations_path: str | Path,
    annotation_types: tuple[str, ...] | None = None,
    *,
    batch_size: int = 65_536,
    include_text: bool = True,
):
    """Yield joined DataFrames, retaining unmatched annotations with null metadata."""
    selected = None if annotation_types is None else set(annotation_types)
    if selected is not None and (not selected or selected - set(ANNOTATION_TYPES)):
        raise ValueError(f"Unknown annotation type(s): {sorted(selected - set(ANNOTATION_TYPES))}")

    metadata_schema = pq.read_schema(metadata_path)
    annotation_file = pq.ParquetFile(annotations_path)
    for name, actual, required in (
        ("metadata", set(metadata_schema.names), METADATA_COLUMNS),
        ("annotations", set(annotation_file.schema_arrow.names), ANNOTATION_COLUMNS),
    ):
        missing = required - actual
        if missing:
            raise ValueError(f"{name} parquet missing columns: {', '.join(sorted(missing))}")

    metadata = pd.read_parquet(metadata_path)
    if metadata.patch_id.isna().any() or metadata.patch_id.duplicated().any():
        raise ValueError("metadata patch_id must be non-null and unique")

    # The annotation parquet also carries split/country/S1 fields. The official
    # metadata values win, so these duplicate annotation fields are omitted.
    columns = ["patch_id", "type"]
    if include_text:
        columns += [
            name for name in annotation_file.schema_arrow.names
            if name not in {"patch_id", "type", "split", "country", "s1_name"}
        ]
    for batch in annotation_file.iter_batches(batch_size=batch_size, columns=columns):
        annotations = batch.to_pandas()
        if selected is not None:
            annotations = annotations[annotations.type.isin(selected)]
        if not annotations.empty:
            yield annotations.merge(metadata, on="patch_id", how="left", validate="many_to_one", indicator=True)


def summarize(metadata_path: str | Path, annotations_path: str | Path) -> dict:
    """Count all annotation rows and distinct patch IDs with bounded batch reads."""
    types: Counter = Counter()
    splits: Counter = Counter()
    countries: Counter = Counter()
    patch_ids: set[str] = set()
    unmatched_ids: set[str] = set()
    total = 0
    for batch in iter_index(metadata_path, annotations_path, include_text=False):
        total += len(batch)
        patch_ids.update(batch.patch_id.dropna())
        unmatched_ids.update(batch.loc[batch._merge == "left_only", "patch_id"].dropna())
        types.update(batch.type.value_counts().to_dict())
        splits.update(batch.split.fillna("<unmatched>").value_counts().to_dict())
        countries.update(batch.country.fillna("<unmatched>").value_counts().to_dict())
    return {
        "total_annotation_rows": total,
        "unique_patch_ids": len(patch_ids),
        "by_annotation_type": dict(sorted(types.items())),
        "by_metadata_split": dict(sorted(splits.items())),
        "by_country": dict(sorted(countries.items())),
        "unmatched_patch_ids": len(unmatched_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BigEarthNet.txt against official metadata")
    parser.add_argument("metadata", type=Path, help="Path to metadata.parquet")
    parser.add_argument("annotations", type=Path, help="Path to BigEarthNet.txt.parquet")
    args = parser.parse_args()
    print(json.dumps(summarize(args.metadata, args.annotations), indent=2))


if __name__ == "__main__":
    main()
