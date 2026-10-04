import csv
import gzip
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from data.bigearthnet_s2 import BANDS
from scripts.validate_bigearthnet_s2 import NATIVE_GRID, band_stats, select_patches, validate


def test_selection_is_deterministic_across_manifest_order(tmp_path):
    rows = [
        {"patch_id": f"patch-{side}-{tile}-{index}", "mgrs_tile": f"tile-{tile}", "geo_split": side}
        for side in ("train", "eval") for tile in range(5) for index in range(6)
    ]
    results = []
    for name, ordered in (("forward", rows), ("reverse", list(reversed(rows)))):
        path = tmp_path / f"{name}.csv.gz"
        with gzip.open(path, "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(ordered)
        results.append(select_patches(path))
    assert results[0] == results[1]
    assert len(results[0]) == 32
    for side in ("train", "eval"):
        assert len({row["mgrs_tile"] for row in results[0] if row["geo_split"] == side}) == 4


def test_statistics_and_extracted_patch_validation(tmp_path):
    assert band_stats(np.array([0, 0, 10, 10]))["zero_fraction"] == 0.5
    patch_id = "S2A_tile_01_02"
    patch = tmp_path / "S2A_tile" / patch_id
    patch.mkdir(parents=True)
    for index, band in enumerate(BANDS, 1):
        size, metres = NATIVE_GRID[band]
        with rasterio.open(patch / f"{patch_id}_{band}.tif", "w", driver="GTiff",
                           width=size, height=size, count=1, dtype="uint16",
                           transform=from_origin(0, 1200, metres, metres)) as dst:
            dst.write(np.full((size, size), index, dtype="uint16"), 1)
    report = validate(tmp_path, [{"patch_id": patch_id, "mgrs_tile": "tile", "geo_split": "train"}])
    assert (report["successful_patches"], report["failed_patches"]) == (1, 0)
    assert report["observations"][0]["output"] == {"dtype": "float32", "shape": [12, 120, 120], "finite": True}
    assert report["statistics"]["train"]["B12"]["mean"] == 12
    assert report["statistics"]["combined"]["B01"]["zero_fraction"] == 0
    with rasterio.open(patch / f"{patch_id}_B01.tif", "w", driver="GTiff",
                       width=2, height=2, count=1, dtype="uint16",
                       transform=from_origin(0, 2, 1, 1)) as dst:
        dst.write(np.ones((2, 2), dtype="uint16"), 1)
    failed = validate(tmp_path, [{"patch_id": patch_id, "mgrs_tile": "tile", "geo_split": "train"}])
    assert failed["failed_patches"] == 1
    assert "Unexpected native grid for B01" in failed["observations"][0]["error"]
