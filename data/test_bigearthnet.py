"""Small parquet fixtures for the BigEarthNet text index."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.bigearthnet import ANNOTATION_TYPES, iter_index, summarize


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


if __name__ == "__main__":
    unittest.main()
