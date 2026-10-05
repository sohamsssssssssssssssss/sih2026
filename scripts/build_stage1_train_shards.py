"""Build the loop-9 train shard set: prefix-stratified, resumable, guarded.

One archive pass writes 2,000-patch shards in order. The published manifest is
rewritten after every shard, so an interrupted run leaves a valid prefix, and a
rerun verifies existing shards by SHA256, skips them, and continues.

    caffeinate -i nice -n 10 python scripts/build_stage1_train_shards.py \
        --n-total 150000 --out data/shards/train-150000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bigearthnet_shards import (  # noqa: E402
    INTERLEAVED_SELECTION,
    assert_split_purity,
    build_shard_set,
    load_captions,
    load_split_rows,
    prefix_tile_balance,
    select_patch_ids_interleaved,
    shard_bytes,
    verify_shard_files,
)

DEFAULT_ARCHIVE_ROOT = Path("/Users/soham/Datasets/BigEarthNet/V2")
DEFAULT_CAPTIONS = Path("/Users/soham/Datasets/BigEarthNet/BigEarthNet.txt.parquet")
DEFAULT_SPLIT = ROOT / "data" / "manifests" / "bigearthnet" / "caption-geo-split.v1.csv.gz"
PREFIXES = (10000, 50000, 100000, 150000)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def read_selected_ids(path: Path, n: int, seed: int) -> list[str]:
    record = json.loads(path.read_text())
    ids = record["patch_ids"]
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if record["seed"] != seed or digest != record["id_list_sha256"]:
        raise ValueError("Selection seed or ID-list SHA256 mismatch")
    if not 0 < n <= len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid selection prefix length or duplicate IDs")
    return ids[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-total", type=int, default=150000)
    parser.add_argument("--patches-per-shard", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=26168)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "shards" / "train-150000")
    parser.add_argument("--published-manifest", type=Path, default=None)
    parser.add_argument("--selection-id-list", type=Path, default=None,
                        help="Use an explicit checksummed full ID list; n-total takes its prefix")
    parser.add_argument("--selection-record", type=Path, default=None)
    parser.add_argument("--exclude-shards", type=Path, action="append", default=[],
                        help="shard directory whose patch IDs must not be reused (the eval shard)")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--min-free-bytes", type=float, default=45_000_000_000)
    parser.add_argument("--speed-guard-after-shards", type=int, default=10)
    parser.add_argument("--speed-guard-max-seconds-per-patch", type=float, default=0.288)
    parser.add_argument("--max-rejection-fraction", type=float, default=0.01)
    parser.add_argument("--shard-window", type=int, default=None,
                        help="shards buffered in RAM per archive pass; a prefix-stratified "
                             "selection needs one pass per window. Default: 30%% of RAM")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    published = args.published_manifest or (
        ROOT / "data" / "manifests" / "bigearthnet"
        / f"stage1-shards-train-{args.n_total}.v1.json")
    selection_record = args.selection_record or (
        ROOT / "data" / "manifests" / "bigearthnet"
        / f"stage1-train-selection-{args.n_total}.v1.json")
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [args.archive_root / f"BigEarthNet-S2.tar.gz{part}" for part in ("aa", "ab")]
    missing = [str(part) for part in parts if not part.is_file()]
    if missing:
        print(f"archive parts missing: {missing}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    rows = load_split_rows(args.split)
    selection = (read_selected_ids(args.selection_id_list, args.n_total, args.seed)
                 if args.selection_id_list else
                 select_patch_ids_interleaved(rows, "train", args.n_total, args.seed))
    assert_split_purity(selection, rows, "train")
    balance = prefix_tile_balance(rows, selection, [n for n in PREFIXES if n <= len(selection)])
    print(f"selection {len(selection)} patches, {len({rows[p]['mgrs_tile'] for p in selection})} tiles",
          flush=True)
    for prefix, values in balance.items():
        print(f"  prefix {prefix}: tiles={values['tiles_present']} min={values['min']} "
              f"max={values['max']} spread={values['spread']}", flush=True)

    excluded_ids: set[str] = set()
    excluded_tiles: dict[str, int] = {}
    for shard_dir in args.exclude_shards:
        manifest_path = Path(shard_dir) / "shards-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        ids = set()
        for record in manifest["shards"]:
            sidecar = json.loads((Path(shard_dir) / record["sidecar"]).read_text())
            ids.update(row["patch_id"] for row in sidecar["rows"])
        excluded_ids |= ids
        excluded_tiles[manifest["split_name"]] = len(ids)
        print(f"excluding {len(ids)} patch IDs from {shard_dir}", flush=True)
    overlap = excluded_ids & set(selection)
    if overlap:
        raise ValueError(f"Selection overlaps the excluded shard set: {sorted(overlap)[:3]}")

    record = {
        "kind": "stage-1 train selection record",
        "git_sha": git_sha(), "seed": args.seed, "n_total": args.n_total,
        "split_name": "caption-geo-split.v1/train",
        "algorithm": INTERLEAVED_SELECTION,
        "prefix_tile_balance": {str(key): value for key, value in balance.items()},
        "prefix_balance_gate": "max - min <= 1 per tile for every reported prefix",
        "first_ten_ids": selection[:10],
        "last_ten_ids": selection[-10:],
        "note": ("the full ID list is not stored here; rerun select_patch_ids_interleaved with this "
                 "seed and n_total to reproduce it, and the per-shard sidecars carry the IDs that "
                 "were actually written"),
    }
    selection_record.parent.mkdir(parents=True, exist_ok=True)
    selection_record.write_text(json.dumps(record, indent=2) + "\n")

    captions = load_captions(args.captions, selection)
    projected = shard_bytes(len(selection))
    free_before = shutil.disk_usage(args.out).free
    print(f"captions joined for {len(captions)} patches; projected raw {projected} B; "
          f"free {free_before} B", flush=True)

    existing_bytes = sum(p.stat().st_size for p in args.out.glob("shard-*.npz"))
    projected_remaining = max(0, projected - existing_bytes) + 100_000_000
    if free_before - projected_remaining < args.min_free_bytes:
        raise ValueError("Projected rebuild would violate the free-disk guard; choose another volume")

    result = build_shard_set(
        parts, selection, captions, rows, args.out,
        git_sha=git_sha(), seed=args.seed, side="train",
        patches_per_shard=args.patches_per_shard, published_manifest=published,
        excluded_ids=frozenset(excluded_ids), excluded_tile_counts=excluded_tiles,
        min_free_bytes=int(args.min_free_bytes),
        speed_guard_after_shards=args.speed_guard_after_shards,
        speed_guard_max_seconds_per_patch=args.speed_guard_max_seconds_per_patch,
        max_rejection_fraction=args.max_rejection_fraction,
        shard_window=args.shard_window,
    )
    verified, problems = verify_shard_files(args.out, result["shards"])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report = {
        "kind": "loop 9 train shard build report",
        "git_sha": git_sha(),
        "seed": args.seed,
        "n_total_requested": args.n_total,
        "device": "CPU-only preparation (Apple M5 MacBook Pro, py3.11.15 .venv-stage1)",
        "published_manifest": str(published.relative_to(ROOT)),
        "selection_record": str(selection_record.relative_to(ROOT)),
        "prefix_tile_balance": {str(key): value for key, value in balance.items()},
        "status": result["status"],
        "stopped_reason": result["stopped_reason"],
        "shards": len(result["shards"]),
        "n_patches": result["n_patches"],
        "total_bytes": result["total_bytes"],
        "verified_shards_after_build": len(verified),
        "verification_problems": problems,
        "rejected_this_run": result["rejected_this_run"],
        "rejected_count_total": result["rejected_count_total"],
        "rejection_fraction_this_run": result["rejection_fraction_this_run"],
        "rejected_by_reason": result["rejected_by_reason"],
        "seconds_elapsed": time.perf_counter() - started,
        "seconds_per_patch": result["seconds_per_patch_this_run"],
        "archive_bytes_streamed": result["archive_bytes_streamed"],
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(args.out).free,
        "free_bytes_min": result["free_bytes_min"],
        "extraction_temp_bytes": 0,
        "extraction_temp_note": ("selected members are decoded from rasterio MemoryFile and never "
                                 "written to disk; no temporary extraction directory exists"),
        "peak_process_rss_bytes": peak_rss,
        "guards": {
            "min_free_bytes": int(args.min_free_bytes),
            "speed_guard_after_shards": args.speed_guard_after_shards,
            "speed_guard_max_seconds_per_patch": args.speed_guard_max_seconds_per_patch,
            "max_rejection_fraction": args.max_rejection_fraction,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "shards", "n_patches", "total_bytes", "seconds_per_patch",
        "rejected_this_run", "free_bytes_min")}, indent=2), flush=True)
    if problems:
        print(f"verification problems: {problems[:5]}", file=sys.stderr)
        return 4
    return 0 if result["status"] == "complete" else 5


if __name__ == "__main__":
    raise SystemExit(main())