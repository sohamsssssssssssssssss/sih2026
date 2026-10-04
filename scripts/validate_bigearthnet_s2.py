"""Validate a small, already extracted BigEarthNet S2 sample; never opens archives."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch

from data.bigearthnet_s2 import BANDS, load_s2_patch

NATIVE_GRID = {
    "B01": (20, 60), "B02": (120, 10), "B03": (120, 10), "B04": (120, 10),
    "B05": (60, 20), "B06": (60, 20), "B07": (60, 20), "B08": (120, 10),
    "B8A": (60, 20), "B09": (20, 60), "B11": (60, 20), "B12": (60, 20),
}


def select_patches(manifest: Path) -> list[dict[str, str]]:
    """Four hash-ranked tiles and four hash-ranked patches/tile on each geo side."""
    by_side = defaultdict(lambda: defaultdict(list))
    with gzip.open(manifest, "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            by_side[row["geo_split"]][row["mgrs_tile"]].append(row)

    def rank(value: str) -> str:
        return hashlib.sha256(f"a2-real-v1:{value}".encode()).hexdigest()

    selected = []
    for side in ("train", "eval"):
        tiles = sorted(by_side[side], key=lambda tile: (rank(tile), tile))[:4]
        if len(tiles) != 4:
            raise ValueError(f"Insufficient {side} MGRS tiles")
        for tile in tiles:
            rows = sorted(by_side[side][tile], key=lambda row: (rank(row["patch_id"]), row["patch_id"]))[:4]
            if len(rows) != 4:
                raise ValueError(f"Insufficient patches in {side} tile {tile}")
            selected.extend({key: row[key] for key in ("patch_id", "mgrs_tile", "geo_split")} for row in rows)
    return selected


def band_stats(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64, copy=False).ravel()
    p1, p5, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "p1": float(p1), "p5": float(p5), "p50": float(p50),
            "p95": float(p95), "p99": float(p99),
            "zero_fraction": float(np.count_nonzero(values == 0) / values.size)}


def validate(root: Path, selected: list[dict[str, str]]) -> dict:
    observations = []
    pixels = defaultdict(lambda: defaultdict(list))
    for row in selected:
        patch_id = row["patch_id"]
        patch_dir = root / patch_id.rsplit("_", 2)[0] / patch_id
        record = dict(row)
        try:
            source = {}
            for band in BANDS:
                path = patch_dir / f"{patch_id}_{band}.tif"
                with rasterio.open(path) as image:
                    size, metres = NATIVE_GRID[band]
                    if (image.height, image.width) != (size, size) or not np.allclose(image.res, (metres, metres)):
                        raise ValueError(f"Unexpected native grid for {band}: {image.height}x{image.width}, {image.res}")
                    source_pixels = image.read(1)
                    source[band] = {"dtype": image.dtypes[0], "dimensions": [image.height, image.width],
                                    "resolution": list(image.res), "nodata": image.nodata,
                                    "mask_flags": [flag.name for flag in image.mask_flag_enums[0]],
                                    "source_zero_fraction": float(np.count_nonzero(source_pixels == 0) / source_pixels.size)}
            tensor = load_s2_patch(patch_dir)
            if tensor.shape != (12, 120, 120) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("Output tensor has invalid shape, dtype, or nonfinite values")
            for index, band in enumerate(BANDS):
                pixels[row["geo_split"]][band].append(tensor[index].numpy())
            record.update(status="ok", source=source, output={"dtype": "float32", "shape": [12, 120, 120], "finite": True})
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        observations.append(record)
    statistics = {}
    for side in ("train", "eval", "combined"):
        statistics[side] = {}
        for band in BANDS:
            chunks = (pixels["train"][band] + pixels["eval"][band]) if side == "combined" else pixels[side][band]
            if chunks:
                statistics[side][band] = band_stats(np.stack(chunks))
    return {"selection_method": "SHA256(a2-real-v1:<tile or patch_id>), four tiles per geo side, four patches per tile",
            "bands": list(BANDS), "observations": observations, "statistics": statistics,
            "successful_patches": sum(row["status"] == "ok" for row in observations),
            "failed_patches": sum(row["status"] == "failed" for row in observations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Root containing selected extracted S2 tiles")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.root, select_patches(args.manifest))
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['successful_patches']} successful; {report['failed_patches']} failed")


if __name__ == "__main__":
    main()
