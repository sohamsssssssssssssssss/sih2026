"""Fit and compare Stage-1 normalization candidates on saved train pixels only."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.bigearthnet_s2 import BANDS


def fit(study_path: Path, ids_path: Path, pixels_path: Path) -> tuple[dict, dict]:
    study = json.loads(study_path.read_text())
    selection = json.loads(ids_path.read_text())
    if study["split_name"] != "caption-geo-split.v1/train" or selection["split_name"] != study["split_name"]:
        raise ValueError("Normalization fit requires only the frozen train split")
    if study["id_list_sha256"] != hashlib.sha256(ids_path.read_bytes()).hexdigest():
        raise ValueError("Study and ID list disagree")
    n = study["n"]
    pixels = np.memmap(pixels_path, mode="r", dtype="float32", shape=(n, 12, 120, 120))
    histogram = np.zeros(65536, dtype=np.int64)
    flat = pixels.reshape(-1)
    for start in range(0, flat.size, 5_000_000):
        block = np.asarray(flat[start:start + 5_000_000])
        if not np.isfinite(block).all() or np.any((block < 0) | (block >= 65536)):
            raise ValueError("Unexpected resampled DN outside uint16 range")
        histogram += np.bincount(np.floor(block).astype(np.int32), minlength=65536)
    rank = int(np.ceil(0.999 * flat.size))
    lower_bin = int(np.searchsorted(np.cumsum(histogram), rank))
    clip_dn = lower_bin + 1  # Upper edge of the pooled p99.9 DN bin.
    clip = clip_dn / 10000

    means, stds, clipped, near_one = {}, {}, {}, {}
    option_a_subsets, option_b_subsets = {}, {}
    remaining = np.arange(n)
    subset_indices = {}
    for seed in (0, 1, 2):
        take = np.sort(np.random.default_rng(seed).choice(remaining, size=1000, replace=False))
        remaining = np.setdiff1d(remaining, take)
        subset_indices[str(seed)] = take
        option_a_subsets[str(seed)] = {}
        option_b_subsets[str(seed)] = {}
    for index, band in enumerate(BANDS):
        raw = pixels[:, index]
        clipped[band] = float(np.mean(raw > clip_dn))
        near_one[band] = float(np.mean(raw / 10000 >= 0.99))
        scaled = np.clip(raw / 10000, 0, clip)
        means[band] = float(scaled.mean(dtype=np.float64))
        stds[band] = float(scaled.std(dtype=np.float64))
        if stds[band] <= 0:
            raise ValueError(f"Zero fit std for {band}")
        for seed, take in subset_indices.items():
            subset = scaled[take]
            option_b_subsets[seed][band] = {"mean": float(subset.mean()), "std": float(subset.std()),
                                            "p99": float(np.percentile(subset, 99))}
            standardized = (subset - means[band]) / stds[band]
            option_a_subsets[seed][band] = {"mean": float(standardized.mean()), "std": float(standardized.std()),
                                            "p99": float(np.percentile(standardized, 99))}

    contract = {"version": "1.0", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "split_name": study["split_name"],
                "n": n, "seed": study["seed"], "id_list_sha256": study["id_list_sha256"],
                "manifest_sha256": selection["manifest_sha256"], "bands": list(BANDS),
                "input_dtype": "uint16", "output_dtype": "float32", "target_size": [120, 120],
                "resampling": "rasterio bilinear", "divisor": 10000.0, "clip_min": 0.0,
                "clip_max": clip, "mean": means, "std": stds,
                "nodata": {"source_value": 0, "action": "reject patch with any raw zero when normalization is requested"},
                "raw_one_handling": "set pixels where all 12 resampled bands equal 1 to zero after standardization; retain band-specific ones"}

    def max_relative_spread(subsets: dict, metric: str) -> float:
        return max((max(subsets[str(seed)][band][metric] for seed in (0, 1, 2)) -
                    min(subsets[str(seed)][band][metric] for seed in (0, 1, 2))) /
                   abs(sum(subsets[str(seed)][band][metric] for seed in (0, 1, 2)) / 3)
                   for band in BANDS)

    report = {"git_sha": contract["git_sha"], "timestamp_utc": contract["timestamp_utc"],
              "split_name": contract["split_name"], "n": n, "seed": contract["seed"],
              "id_list_sha256": contract["id_list_sha256"], "pooled_p999_dn_bin_lower": lower_bin,
              "all_12_one_pixel_fraction": float(np.mean([row["frac_all_12_eq_1"] for row in study["per_patch"]])),
              "clip_upper_dn": clip_dn, "clip_upper_scaled": clip,
              "clip_choice": "upper edge of one-DN histogram bin containing pooled train p99.9",
              "frac_clipped_by_band": clipped, "pooled_frac_clipped": float(np.mean(list(clipped.values()))),
              "frac_ge_0_99_after_divisor_before_clip_by_band": near_one,
              "option_A": {"definition": "clip(x/10000,0,c), then per-band train-fit (x-mean)/std",
                           "subset_stats": option_a_subsets,
                           "max_subset_absolute_mean": max(abs(option_a_subsets[str(seed)][band]["mean"])
                                                           for seed in (0, 1, 2) for band in BANDS),
                           "max_subset_absolute_std_delta_from_1": max(abs(option_a_subsets[str(seed)][band]["std"] - 1)
                                                                       for seed in (0, 1, 2) for band in BANDS),
                           "max_relative_p99_spread": max_relative_spread(option_a_subsets, "p99")},
              "option_B": {"definition": "clip(x/10000,0,c), one global scale, no per-band fit",
                           "subset_stats": option_b_subsets,
                           "max_relative_mean_spread": max_relative_spread(option_b_subsets, "mean"),
                           "max_relative_std_spread": max_relative_spread(option_b_subsets, "std"),
                           "max_relative_p99_spread": max_relative_spread(option_b_subsets, "p99")},
              "chosen": "A", "chosen_reason": "Per-band train fit equalizes observed band scales; subset spread remains and is an open limitation"}
    return contract, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract, report = fit(args.study, args.ids, args.pixels)
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
