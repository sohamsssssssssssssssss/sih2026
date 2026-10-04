"""Small parquet fixtures for the BigEarthNet text index."""

import csv
import gzip
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.bigearthnet import ANNOTATION_TYPES, iter_index, summarize
from data.bigearthnet_split import build_split, mgrs_tile


class BigEarthNetIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.metadata = root / "metadata.parquet"
        self.annotations = root / "BigEarthNet.txt.parquet"
        pd.DataFrame(
            [
                {"patch_id": "a", "split": "train", "country": "Austria", "s1_name": "s1-a", "s2v1_name": "s2-a"},
                {"patch_id": "b", "split": "test", "country": "Belgium", "s1_name": "s1-b", "s2v1_name": "s2-b"},
            ]
        ).to_parquet(self.metadata)
        pd.DataFrame(
            [
                {"patch_id": "a", "type": "captioning", "input": "Describe", "output": "A field", "split": "wrong", "country": "wrong"},
                {"patch_id": "a", "type": "binary", "input": "Field?", "output": "yes", "split": "wrong", "country": "wrong"},
                {"patch_id": "b", "type": "mcq", "input": "Which?", "output": "a", "split": "wrong", "country": "wrong"},
                {"patch_id": "missing", "type": "bounding box", "input": "Where?", "output": "[0.0 0.0, 1.0 1.0]", "split": "wrong", "country": "wrong"},
            ]
        ).to_parquet(self.annotations)

    def test_filter_join_and_summary(self) -> None:
        for annotation_type in ANNOTATION_TYPES:
            batches = list(iter_index(self.metadata, self.annotations, (annotation_type,), batch_size=2))
            self.assertEqual(sum(map(len, batches)), 1)
            self.assertEqual(batches[0].iloc[0]["type"], annotation_type)

        rows = pd.concat(iter_index(self.metadata, self.annotations, batch_size=2))
        self.assertEqual(rows.loc[rows.patch_id == "a", "split"].tolist(), ["train", "train"])
        self.assertEqual(rows.loc[rows.patch_id == "a", "country"].tolist(), ["Austria", "Austria"])
        self.assertEqual(rows.loc[rows.patch_id == "a", "s2v1_name"].tolist(), ["s2-a", "s2-a"])
        self.assertEqual(rows.loc[rows.patch_id == "missing", "_merge"].iloc[0], "left_only")

        self.assertEqual(summarize(self.metadata, self.annotations), {
            "total_annotation_rows": 4,
            "unique_patch_ids": 3,
            "by_annotation_type": {"binary": 1, "bounding box": 1, "captioning": 1, "mcq": 1},
            "by_metadata_split": {"<unmatched>": 1, "test": 1, "train": 2},
            "by_country": {"<unmatched>": 1, "Austria": 2, "Belgium": 1},
            "unmatched_patch_ids": 1,
        })

    def test_required_columns_and_unique_metadata_ids(self) -> None:
        pd.DataFrame({"patch_id": ["a"], "split": ["train"]}).to_parquet(self.metadata)
        with self.assertRaisesRegex(ValueError, "metadata parquet missing columns"):
            list(iter_index(self.metadata, self.annotations))
        pd.DataFrame({"patch_id": ["a", "a"], "split": ["train", "test"], "country": ["A", "B"], "s1_name": ["1", "2"], "s2v1_name": ["1", "2"]}).to_parquet(self.metadata)
        with self.assertRaisesRegex(ValueError, "metadata patch_id"):
            list(iter_index(self.metadata, self.annotations))
        pd.DataFrame({"patch_id": ["a"], "type": ["captioning"]}).to_parquet(self.annotations)
        with self.assertRaisesRegex(ValueError, "annotations parquet missing columns"):
            list(iter_index(self.metadata, self.annotations))

    def test_cli(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "data.bigearthnet", str(self.metadata), str(self.annotations)],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(result.stdout)["total_annotation_rows"], 4)


class BigEarthNetGeoSplitTest(unittest.TestCase):
    @staticmethod
    def patch(tile: str, row: int) -> str:
        return f"S2A_MSIL2A_20170613T101031_N9999_R022_{tile}_{row}_57"

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.metadata = root / "metadata.parquet"
        self.annotations = root / "annotations.parquet"
        self.manifest = root / "split.csv.gz"
        self.patches = [self.patch(tile, i) for tile in ("T33UUP", "T34UCF", "T35UMA") for i in range(2)]
        pd.DataFrame([
            {"patch_id": patch, "split": "train" if i % 2 else "test", "country": "Austria", "s1_name": f"s1-{i}", "s2v1_name": f"s2-{i}"}
            for i, patch in enumerate(self.patches)
        ]).to_parquet(self.metadata)
        pd.DataFrame([
            {"patch_id": patch, "type": "captioning", "input": "Describe", "output": "A field"}
            for patch in self.patches
        ]).to_parquet(self.annotations)

    def test_mgrs_parser_rejects_malformed_ids(self) -> None:
        self.assertEqual(mgrs_tile(self.patches[0]), "T33UUP")
        for bad in ("T33UUP", self.patches[0] + "_extra", self.patch("T00UUP", 1), self.patch("T61UUP", 1), self.patch("T33IUP", 1), self.patch("T33UUI", 1), None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Malformed"):
                mgrs_tile(bad)

    def test_deterministic_complete_disjoint_manifest(self) -> None:
        first = build_split(self.metadata, self.annotations, self.manifest, expected_captions=6)
        copy = self.manifest.with_name("copy.csv.gz")
        second = build_split(self.metadata, self.annotations, copy, expected_captions=6)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(self.manifest.read_bytes(), copy.read_bytes())
        with gzip.open(self.manifest, "rt", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        train = [row for row in rows if row["geo_split"] == "train"]
        evaluation = [row for row in rows if row["geo_split"] == "eval"]
        self.assertFalse({row["mgrs_tile"] for row in train} & {row["mgrs_tile"] for row in evaluation})
        self.assertFalse({row["patch_id"] for row in train} & {row["patch_id"] for row in evaluation})
        self.assertEqual({row["patch_id"] for row in rows}, set(self.patches))
        self.assertEqual(len(rows), len(self.patches))
        self.assertEqual(first["train_patches"] + first["eval_patches"], 6)
        self.assertEqual(first["source_official_split_counts"], {"test": 3, "train": 3})
        self.assertEqual(json.loads(self.manifest.with_suffix(".json").read_text())["manifest_sha256"], first["manifest_sha256"])

    def test_fails_closed_for_bad_ids_missing_or_duplicate_captions(self) -> None:
        captions = pd.read_parquet(self.annotations)
        captions.loc[0, "patch_id"] = "bad"
        captions.to_parquet(self.annotations)
        with self.assertRaisesRegex(AssertionError, "missing from metadata"):
            build_split(self.metadata, self.annotations, self.manifest, expected_captions=6)
        metadata = pd.read_parquet(self.metadata)
        metadata.loc[0, "patch_id"] = "bad"
        metadata.to_parquet(self.metadata)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            build_split(self.metadata, self.annotations, self.manifest, expected_captions=6)
        metadata.loc[0, "patch_id"] = self.patches[0]
        metadata.to_parquet(self.metadata)
        captions.loc[0, "patch_id"] = self.patches[1]
        captions.to_parquet(self.annotations)
        with self.assertRaisesRegex(AssertionError, "duplicate patch"):
            build_split(self.metadata, self.annotations, self.manifest, expected_captions=6)
        captions = captions.iloc[:-1]
        captions.to_parquet(self.annotations)
        with self.assertRaisesRegex(AssertionError, "count mismatch"):
            build_split(self.metadata, self.annotations, self.manifest, expected_captions=6)


if __name__ == "__main__":
    unittest.main()
