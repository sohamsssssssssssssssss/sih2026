"""Study an explicitly extracted train-only S2 sample; never opens archives."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from data.bigearthnet_s2 import BANDS, load_s2_patch


PERCENTILES = (0.1, 1, 5, 50, 95, 99, 99.9)


def stats(values: np.ndarray) -> dict:
    flat = np.asarray(values).ravel()
    percentiles = np.percentile(flat, PERCENTILES)
    result = {f"p{p}": float(value) for p, value in zip(PERCENTILES, percentiles)}
    result.update(dtype=str(flat.dtype), min=float(flat.min()), max=float(flat.max()),
                  mean=float(flat.mean(dtype=np.float64)), std=float(flat.std(dtype=np.float64)),
                  frac_eq_0=float(np.mean(flat == 0)), frac_eq_1=float(np.mean(flat == 1)),
                  frac_le_5=float(np.mean(flat <= 5)), frac_ge_10000=float(np.mean(flat >= 10000)),
                  frac_ge_dtype_max=float(np.mean(flat >= 65535)))
    return result


def quicklook(tensor: np.ndarray, path: Path) -> None:
    def display(indices):
        channels = tensor[indices].astype(np.float32)
        lo, hi = np.percentile(channels, [2, 98], axis=(1, 2))
        return np.moveaxis(np.uint8(np.clip((channels - lo[:, None, None]) /
                                               np.maximum(hi - lo, 1)[:, None, None], 0, 1) * 255), 0, -1)

    rgb = display([3, 2, 1])
    nir = display([7])[..., 0]
    near_one = np.any(tensor[[3, 4, 5, 6, 7, 8, 9]] == 1, axis=0)
    all_one = np.all(tensor == 1, axis=0)
    panel = np.concatenate((rgb, np.repeat(nir[..., None], 3, axis=2),
                            np.stack((near_one * 255, all_one * 255, np.zeros_like(nir)), axis=2)), axis=1)
    Image.fromarray(panel.astype(np.uint8)).save(path)


def run(root: Path, id_list: Path, scratch: Path, quicklook_dir: Path) -> dict:
    selection = json.loads(id_list.read_text())
    if selection["split_name"] != "caption-geo-split.v1/train":
        raise ValueError("Study requires the frozen train split")
    ids = selection["patch_ids"]
    n = len(ids)
    scratch.mkdir(parents=True, exist_ok=True)
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.memmap(scratch / "resampled-float32.dat", mode="w+", dtype="float32", shape=(n, 12, 120, 120))
    hist = {band: np.zeros(51, dtype=np.int64) for band in BANDS}
    raw_counts = {band: Counter() for band in BANDS}
    per_patch = []
    for index, patch_id in enumerate(ids):
        patch_dir = root / patch_id.rsplit("_", 2)[0] / patch_id
        raw_ones = {}
        for band in BANDS:
            with rasterio.open(patch_dir / f"{patch_id}_{band}.tif") as source:
                if source.count != 1 or source.dtypes[0] != "uint16" or source.nodata not in (None, 0):
                    raise ValueError(f"Unexpected source dtype/nodata: {patch_id} {band}")
                raw = source.read(1)
                hist[band] += np.bincount(raw[raw <= 50].ravel(), minlength=51)[:51]
                raw_counts[band]["pixels"] += raw.size
                raw_counts[band]["eq_0"] += np.count_nonzero(raw == 0)
                raw_counts[band]["eq_1"] += np.count_nonzero(raw == 1)
                raw_counts[band]["ge_10000"] += np.count_nonzero(raw >= 10000)
                raw_counts[band]["ge_65535"] += np.count_nonzero(raw == 65535)
                raw_ones[band] = float(np.mean(raw == 1))
        tensor = load_s2_patch(patch_dir).numpy()
        if not np.isfinite(tensor).all():
            raise ValueError(f"Nonfinite tensor: {patch_id}")
        pixels[index] = tensor
        ones = tensor == 1
        per_patch.append({"patch_id": patch_id, "frac_eq_1_by_band": dict(zip(BANDS, np.mean(ones, axis=(1, 2)).tolist())),
                          "source_frac_eq_1_by_band": raw_ones,
                          "frac_all_12_eq_1": float(np.mean(np.all(ones, axis=0))),
                          "frac_any_b04_to_b09_eq_1": float(np.mean(np.any(ones[[3, 4, 5, 6, 7, 8, 9]], axis=0)))})
        if index % 500 == 0:
            print(f"processed {index + 1}/{n}", flush=True)
    pixels.flush()

    by_band = {}
    for band_index, band in enumerate(BANDS):
        values = pixels[:, band_index]
        by_band[band] = stats(values)
        by_band[band]["source_dtype"] = "uint16"
        by_band[band]["source_fractions"] = {key: count / raw_counts[band]["pixels"] for key, count in raw_counts[band].items() if key != "pixels"}
        by_band[band]["source_pixels"] = raw_counts[band]["pixels"]
        by_band[band]["low_50_source_dn_histogram"] = {str(value): int(count) for value, count in enumerate(hist[band])}
        by_band[band]["resampled_noninteger_fraction"] = float(np.mean(values != np.floor(values)))

    remaining = np.arange(n)
    subsets = {}
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(remaining, size=1000, replace=False))
        remaining = np.setdiff1d(remaining, take)
        subsets[str(seed)] = {}
        for index, band in enumerate(BANDS):
            measured = stats(pixels[take, index])
            subsets[str(seed)][band] = {key: measured[key] for key in ("mean", "std", "p99")}
    stability = {}
    for band in BANDS:
        stability[band] = {key: float((max(subsets[str(seed)][band][key] for seed in (0, 1, 2)) -
                                       min(subsets[str(seed)][band][key] for seed in (0, 1, 2))) /
                                      abs(by_band[band][key])) for key in ("mean", "std", "p99")}

    # Show three distinct failure signatures instead of three near-identical B09 patches.
    all_band = max(range(n), key=lambda i: (per_patch[i]["frac_all_12_eq_1"], ids[i]))
    broad = max(range(n), key=lambda i: (sum(per_patch[i]["frac_eq_1_by_band"].values()), ids[i]))
    b09_only = max(range(n), key=lambda i: (per_patch[i]["frac_eq_1_by_band"]["B09"] -
                                            sum(value for band, value in per_patch[i]["frac_eq_1_by_band"].items() if band != "B09") / 11,
                                            ids[i]))
    worst = list(dict.fromkeys((all_band, broad, b09_only)))
    worst_patches = []
    for index in worst:
        path = quicklook_dir / f"{ids[index]}.png"
        quicklook(pixels[index], path)
        worst_patches.append({**per_patch[index], "per_band_stats": {band: stats(pixels[index, j]) for j, band in enumerate(BANDS)},
                              "quicklook": path.as_posix()})

    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "split_name": selection["split_name"],
            "n": n, "seed": selection["seed"], "id_list_sha256": hashlib.sha256(id_list.read_bytes()).hexdigest(),
            "extracted_tiff_count": n * len(BANDS),
            "extracted_tiff_bytes": sum(path.stat().st_size for path in root.rglob("*.tif")),
            "statistics_basis": "bilinear-resampled float32 raw DN, all pixels, population std",
            "bands": by_band, "per_patch": per_patch, "worst_patches": worst_patches,
            "one_diagnostics": {
                "patches_with_any_all_12_eq_1": sum(item["frac_all_12_eq_1"] > 0 for item in per_patch),
                "max_frac_all_12_eq_1": max(item["frac_all_12_eq_1"] for item in per_patch),
                "patches_with_b09_entirely_eq_1": sum(item["frac_eq_1_by_band"]["B09"] == 1 for item in per_patch),
            },
            "disjoint_1000_patch_subsets": subsets, "max_relative_diff": stability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--quicklooks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.root, args.ids, args.scratch, args.quicklooks), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
