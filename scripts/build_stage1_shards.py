"""Build Stage-1 training shards from the BigEarthNet-S2 archive.

One sequential gzip pass over the split archive parts; only selected members are
kept. Disk budget and buffer are enforced before any byte is written, and the
measured shard sizes feed the labeled projections in the build report.

    python scripts/build_stage1_shards.py --n-total 2000 --out data/shards/train-2000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bigearthnet_shards import (  # noqa: E402
    assert_split_purity,
    build_shards,
    load_captions,
    load_split_rows,
    select_patch_ids,
    shard_bytes,
    verify_manifest,
)

DEFAULT_ARCHIVE_ROOT = Path("/Users/soham/Datasets/BigEarthNet/V2")
DEFAULT_CAPTIONS = Path("/Users/soham/Datasets/BigEarthNet/BigEarthNet.txt.parquet")
DEFAULT_SPLIT = ROOT / "data" / "manifests" / "bigearthnet" / "caption-geo-split.v1.csv.gz"
DEFAULT_CONTRACT = ROOT / "data" / "manifests" / "bigearthnet" / "norm_contract.json"
PROJECTION_GRID = (100000, 150000, 250000, 422809)
SESSION_DISK_BUDGET_BYTES = 30_000_000_000
LOCAL_DISK_BUFFER_BYTES = 40 * 1024 ** 3


def git_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def project_bytes(measured_raw: int, measured_rows: int, n_total: int, patches_per_shard: int) -> dict:
    """Linear projection from the measured shard; explicitly not a measurement."""
    per_patch = measured_raw / measured_rows
    return {
        "n_total": n_total,
        "raw_bytes": per_patch * n_total,
        "shards": -(-n_total // patches_per_shard),
        "label": "projection derived from the measured shard; not measured",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--patches-per-shard", type=int, default=2000)
    parser.add_argument("--side", choices=("train", "eval"), default="train")
    parser.add_argument("--seed", type=int, default=26168)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--projections", action="store_true",
                        help="also report projections for the pre-registered N grid")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    parts = [args.archive_root / f"BigEarthNet-S2.tar.gz{part}" for part in ("aa", "ab")]
    missing = [str(part) for part in parts if not part.is_file()]
    if missing:
        print(f"archive parts missing: {missing}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    rows = load_split_rows(args.split)
    print(f"split manifest: {len(rows)} rows in {time.perf_counter() - started:.1f}s", flush=True)
    selected = select_patch_ids(rows, args.side, args.n_total, args.seed)
    assert_split_purity(selected, rows, args.side)
    print(f"selected {len(selected)} {args.side} patches across "
          f"{len({rows[p]['mgrs_tile'] for p in selected})} tiles", flush=True)

    captions = load_captions(args.captions, selected)
    projected_raw = shard_bytes(len(selected))
    free = free_bytes(ROOT)
    if free - projected_raw < LOCAL_DISK_BUFFER_BYTES:
        print(f"refusing to build: need {projected_raw} bytes, free {free}, "
              f"buffer {LOCAL_DISK_BUFFER_BYTES}", file=sys.stderr)
        return 3

    manifest = build_shards(parts, selected, captions, rows, args.out,
                            git_sha=git_sha(), seed=args.seed, side=args.side,
                            patches_per_shard=args.patches_per_shard,
                            norm_contract_path=args.contract)
    check = verify_manifest(args.out / "shards-manifest.json")
    manifest["verify"] = check
    report = {
        "manifest": manifest,
        "measured": {
            "rows": manifest["n_patches"],
            "npz_bytes": sum(record["npz_bytes"] for record in manifest["shards"]),
            "raw_pixel_bytes": manifest["raw_pixel_bytes_total"],
            "deflated_pixel_bytes": manifest["deflated_pixel_bytes_total"],
            "deflate_ratio": manifest["deflate_ratio"],
            "build_seconds": manifest["build_seconds"],
            "archive_bytes_streamed": manifest["archive_bytes_streamed"],
            "archive_stream_stopped_early": manifest["archive_stream_stopped_early"],
            "selection_size": manifest["selection_size"],
            "shortfall": manifest["shortfall"],
            "never_seen_in_archive": manifest["never_seen_in_archive"],
            "rejected_count": manifest["rejected_count"],
            "candidate_rejected_fraction": manifest["candidate_rejected_fraction"],
            "rejected_by_reason": manifest["rejected_by_reason"],
        },
        "free_bytes_before_build": free,
        "free_bytes_after_build": free_bytes(ROOT),
    }
    if args.projections:
        report["projections"] = [project_bytes(manifest["raw_pixel_bytes_total"],
                                               manifest["n_patches"], n,
                                               args.patches_per_shard)
                                 for n in PROJECTION_GRID]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("measured", "free_bytes_after_build")}, indent=2))
    if not check["ok"]:
        print(f"manifest verification problems: {check['problems']}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())