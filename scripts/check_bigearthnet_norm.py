"""Check a frozen normalization contract through the real loader on train IDs."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.bigearthnet_s2 import BANDS, load_s2_patch


def check(root: Path, ids_path: Path, contract_path: Path) -> dict:
    selection = json.loads(ids_path.read_text())
    contract = json.loads(contract_path.read_text())
    if (selection["split_name"] != "caption-geo-split.v1/train" or
            contract["id_list_sha256"] != hashlib.sha256(ids_path.read_bytes()).hexdigest()):
        raise ValueError("Contract and train ID list disagree")
    sums = np.zeros(12, dtype=np.float64)
    squares = np.zeros(12, dtype=np.float64)
    ids = selection["patch_ids"]
    for patch_id in ids:
        directory = root / patch_id.rsplit("_", 2)[0] / patch_id
        pixels = load_s2_patch(directory, norm_contract=contract_path).numpy()
        if pixels.shape != (12, 120, 120) or not np.isfinite(pixels).all():
            raise ValueError(f"Invalid normalized tensor: {patch_id}")
        sums += pixels.sum(axis=(1, 2), dtype=np.float64)
        squares += np.square(pixels.astype(np.float64)).sum(axis=(1, 2))
    count = len(ids) * 120 * 120
    means = sums / count
    stds = np.sqrt(squares / count - means ** 2)
    tolerance = 0.01
    passed = bool(np.all(np.abs(means) < tolerance) and np.all(np.abs(stds - 1) < tolerance))
    result = {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "split_name": selection["split_name"],
              "n": len(ids), "seed": selection["seed"], "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
              "tolerance_absolute": tolerance, "per_band_mean": dict(zip(BANDS, means.tolist())),
              "per_band_std": dict(zip(BANDS, stds.tolist())), "all_finite": True, "passed": passed}
    if not passed:
        raise AssertionError("Normalized train-sample mean/std exceeded tolerance")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.root, args.ids, args.contract), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
