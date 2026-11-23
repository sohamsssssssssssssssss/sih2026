"""Assign the registered attempt-3 groups using only manifest-pinned labels."""
import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ben.adapt import COMMON
from ben.transfer import verify_wc_manifest

OLD = (("delhi", "rajasthan", "sundarbans"), ("assam", "kerala", "punjab"))
GROUPS = "ABCDE"


def assign(old_counts, new_counts):
    """Exact prevalence ranking prevents floating-point differences from hiding ties."""
    names = tuple(sorted(new_counts))
    if len(names) != 10 or set(names) & old_counts.keys() or set(old_counts) != set(sum(OLD, ())):
        raise ValueError("expected the six fixed and ten new regions")
    counts = old_counts | new_counts
    total = [sum(v[col] for v in counts.values()) for col in range(len(COMMON))]
    pooled = [Fraction(n, 300 * len(counts)) for n in total]

    def evaluate(group):
        positives = tuple(sum(counts[name][col] for name in group) for col in range(len(COMMON)))
        gaps = tuple(abs(Fraction(n, 300 * len(group)) - ref) for n, ref in zip(positives, pooled))
        return positives, gaps

    if any(min(evaluate(group)[0]) < 50 for group in OLD):
        return {"status": "not run", "reason": "fixed group below minimum positives"}
    best = None
    for four in combinations(names, 4):
        remainder = tuple(name for name in names if name not in four)
        for three in combinations(remainder[1:], 2):
            first = (remainder[0], *three)
            second = tuple(name for name in remainder if name not in first)
            new_groups = tuple(sorted((four, first, second)))
            groups = OLD + new_groups
            info = [evaluate(group) for group in groups]
            if any(min(positives) < 50 for positives, _ in info):
                continue
            all_gaps = [gap for _, gaps in info for gap in gaps]
            score = (max(all_gaps), sum(gap**2 for gap in all_gaps), new_groups)
            if best is None or score < best[0]:
                best = (score, groups, info)
    if best is None:
        return {"status": "not run", "reason": "no split clears minimum positives"}
    (worst, squared, _), groups, info = best
    return {"status": "assigned", "classes": COMMON, "worst_prevalence_gap": float(worst),
            "sum_squared_gaps": float(squared), "pooled_positives": dict(zip(COMMON, total)),
            "groups": {
                letter: {"regions": group, "patches": 300 * len(group),
                         "positives": dict(zip(COMMON, positives)),
                         "absolute_prevalence_gaps": dict(zip(COMMON, map(float, gaps)))}
                for letter, group, (positives, gaps) in zip(GROUPS, groups, info)
            }}


def load_counts(pack):
    meta = pd.read_parquet(pack / "meta.parquet")
    info = json.loads((pack / "classes.json").read_text())
    cols = [info["coarse_classes"].index(c) for c in COMMON]
    yy = np.load(pack / "labels_coarse.npy", allow_pickle=False)[:, cols]
    counts = {}
    for region in sorted(set(meta.loc[meta.group == "india", "region"])):
        region_labels = yy[meta.region == region]
        if region_labels.shape != (300, len(COMMON)) or not np.isin(region_labels, [0, 1]).all():
            raise ValueError(f"expected 300 binary label rows for {region}")
        counts[region] = region_labels.sum(axis=0).astype(int).tolist()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    hashes = {"old": verify_wc_manifest(args.old), "new": verify_wc_manifest(args.new)}
    if hashes["old"] != "5cc82c713ad76a3286afd4ad3eb66f33da3fb7ff365ab23a6371dc3ba999c40d":
        raise ValueError("registered old WorldCover manifest hash differs")
    result = assign(load_counts(args.old), load_counts(args.new))
    result["manifest_sha256"] = hashes
    with args.result.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["status"] != "assigned":
        raise SystemExit(result["reason"])


if __name__ == "__main__":
    main()
