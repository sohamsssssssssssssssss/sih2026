"""Verify a built Stage-1 shard set against its published manifest (loop 9, step 3).

Everything reported here is measured from the files on disk and from the sidecars,
never projected. The verifier is deliberately independent of the builder: it
re-hashes every .npz, re-reads every sidecar, and re-derives split purity, tile
balance and caption coverage from those sidecars, so a builder bug cannot hide
behind the builder's own bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import random
import tarfile

import numpy as np
from rasterio.io import MemoryFile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bigearthnet_shards import (  # noqa: E402
    BANDS, load_captions, load_split_rows, sha256_file, _ConcatStream,
)

CHECKS = (
    "hashes_match_manifest",
    "sizes_match_manifest",
    "split_pure",
    "no_duplicate_ids",
    "no_overlap_with_excluded_shards",
    "all_bands_present",
    "captions_joined",
    "rows_match_manifest",
    "native_checksums_match",
    "raw_pixels_match",
    "raw_sample_complete",
    "source_captions_match",
)


def audit_native_pixels(shard_dir, sidecars, archive_parts, captions_parquet, seed, problems):
    """Independent decode/checksum path; no builder decode or writer bookkeeping."""
    from data.bigearthnet_s2 import NATIVE_SIZE

    rng = random.Random(seed)
    locations = [(r["npz"], i, row) for r, c in sidecars for i, row in enumerate(c["rows"])]
    sample = []
    for first in range(0, len(sidecars), 10):
        group = [(r["npz"], i, row) for r, c in sidecars[first:first + 10]
                 for i, row in enumerate(c["rows"])]
        sample.extend(rng.sample(group, min(40, len(group))))
    remaining = [item for item in locations if (item[0], item[1]) not in
                 {(x[0], x[1]) for x in sample}]
    # Tiny synthetic fixtures check every row; real sets must meet the registered floor.
    needed = min(len(locations), max(100, 40 * ((len(sidecars) + 9) // 10)))
    sample.extend(rng.sample(remaining, max(0, needed - len(sample))))
    wanted = {row["patch_id"]: (file, i, row) for file, i, row in sample}
    sampled_pixels = {}
    checked_rows = 0
    for record, sidecar in sidecars:
        with np.load(shard_dir / record["npz"], allow_pickle=False) as pixels:
            if set(pixels.files) != set(BANDS):
                problems["all_bands_present"].append(record["npz"])
                continue
            bands = {band: pixels[band] for band in BANDS}
            if any(a.dtype != np.uint16 or a.shape != (len(sidecar["rows"]), NATIVE_SIZE[b], NATIVE_SIZE[b])
                   for b, a in bands.items()):
                problems["all_bands_present"].append(record["npz"] + " shape/dtype")
                continue
            for i, row in enumerate(sidecar["rows"]):
                digest = hashlib.sha256()
                for band in BANDS:
                    size = NATIVE_SIZE[band]
                    digest.update(f"{band}\0uint16\0{size}x{size}\0".encode())
                    digest.update(bands[band][i].astype("<u2", copy=False).tobytes(order="C"))
                checked_rows += 1
                if digest.hexdigest() != row.get("native_sha256"):
                    problems["native_checksums_match"].append(row["patch_id"])
                if row["patch_id"] in wanted:
                    sampled_pixels[row["patch_id"]] = {b: a[i].copy() for b, a in bands.items()}
    if captions_parquet is None:
        problems["source_captions_match"].append("BEN.txt source is required")
    else:
        captions = load_captions(captions_parquet, list(wanted))
        for pid, (_, _, row) in wanted.items():
            if captions[pid] != row["caption"]:
                problems["source_captions_match"].append(pid)
    seen = set()
    mismatches = 0
    if not archive_parts:
        problems["raw_sample_complete"].append("Source archive parts are required")
    else:
        stream = _ConcatStream(archive_parts)
        try:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    path = Path(member.name)
                    pid = path.parent.name
                    prefix = pid + "_"
                    if pid not in wanted or not path.name.startswith(prefix) or path.suffix.lower() != ".tif":
                        continue
                    band = path.stem[len(prefix):]
                    if band not in BANDS:
                        continue
                    key = (pid, band)
                    if key in seen:
                        problems["raw_pixels_match"].append(f"Duplicate source {pid}/{band}")
                    seen.add(key)
                    source = archive.extractfile(member)
                    with MemoryFile(source.read()) as memory:
                        with memory.open() as raster:
                            raw = raster.read(1)
                            if raster.count != 1 or raw.dtype != np.uint16:
                                problems["raw_pixels_match"].append(f"Malformed source {pid}/{band}")
                    actual = sampled_pixels.get(pid, {}).get(band)
                    if actual is None or not np.array_equal(raw, actual):
                        mismatches += 1
                        problems["raw_pixels_match"].append(f"{pid}/{band}")
                    if len(seen) == len(wanted) * len(BANDS):
                        break
        finally:
            stream.close()
    if len(seen) != len(wanted) * len(BANDS) or not wanted:
        problems["raw_sample_complete"].append(f"Read {len(seen)} of {len(wanted) * len(BANDS)} bands")
    return {"checksum_rows_checked": checked_rows, "raw_sample_n": len(wanted),
            "raw_band_checks": len(seen), "raw_band_mismatches": mismatches,
            "sample_seed": seed, "sample_ids": sorted(wanted),
            "native_checksum_scheme": "sha256-native-v1", "mismatches_allowed": 0}


def verify(shard_dir: Path, manifest_path: Path, *, excluded_dirs: list[Path],
           captions_parquet: Path | None = None, split_csv: Path | None = None,
           archive_parts: list[Path] | None = None, sample_seed: int = 26170) -> dict:
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("shards", [])
    side = manifest.get("side") or manifest["split_name"].rsplit("/", 1)[-1]
    split_rows = load_split_rows(split_csv) if split_csv is not None else None
    sidecars: list[tuple[dict, dict]] = []
    problems: dict[str, list[str]] = {name: [] for name in CHECKS}

    for record in records:
        npz = shard_dir / record["npz"]
        sidecar_path = shard_dir / record["sidecar"]
        if not npz.is_file():
            problems["hashes_match_manifest"].append(f"{record['npz']} missing")
            continue
        if npz.stat().st_size != record["npz_bytes"]:
            problems["sizes_match_manifest"].append(
                f"{record['npz']} {npz.stat().st_size} != {record['npz_bytes']}")
        if sha256_file(npz) != record["npz_sha256"]:
            problems["hashes_match_manifest"].append(f"{record['npz']} sha256 mismatch")
        sidecar = json.loads(sidecar_path.read_text())
        sidecars.append((record, sidecar))
        if len(sidecar["rows"]) != record["rows"]:
            problems["rows_match_manifest"].append(f"{record['sidecar']} row count")
        if list(sidecar["bands"]) != list(BANDS):
            problems["all_bands_present"].append(f"{record['sidecar']} band list")
        if sidecar["split_name"] != f"caption-geo-split.v1/{side}":
            problems["split_pure"].append(f"{record['sidecar']} split_name")
        for row in sidecar["rows"]:
            if not row.get("caption"):
                problems["captions_joined"].append(f"{record['sidecar']}:{row['patch_id']}")
                break

    seen: dict[str, int] = {}
    tile_counts: dict[str, int] = defaultdict(int)
    prefix_tile_counts: list[dict[str, int]] = []
    per_shard_tiles: list[int] = []
    patch_ids: list[str] = []
    for index, (record, sidecar) in enumerate(sidecars):
        for row in sidecar["rows"]:
            patch_id = row["patch_id"]
            if patch_id in seen:
                problems["no_duplicate_ids"].append(
                    f"{patch_id} in shards {seen[patch_id]} and {record['index']}")
            seen[patch_id] = record["index"]
            if row.get("geo_split") not in (None, side):
                problems["split_pure"].append(f"{patch_id} geo_split={row.get('geo_split')}")
            if split_rows is not None:
                split_row = split_rows.get(patch_id)
                if split_row is None:
                    problems["split_pure"].append(f"{patch_id} absent from the split CSV")
                elif split_row["geo_split"] != side or split_row["mgrs_tile"] != row["mgrs_tile"]:
                    problems["split_pure"].append(
                        f"{patch_id} split={split_row['geo_split']}/{split_row['mgrs_tile']} "
                        f"sidecar={side}/{row['mgrs_tile']}")
            tile_counts[row["mgrs_tile"]] += 1
            patch_ids.append(patch_id)
        per_shard_tiles.append(len({row["mgrs_tile"] for row in sidecar["rows"]}))
        prefix_tile_counts.append(dict(tile_counts))

    excluded_ids: set[str] = set()
    for directory in excluded_dirs:
        excluded_manifest = directory / "shards-manifest.json"
        if not excluded_manifest.is_file():
            problems["no_overlap_with_excluded_shards"].append(f"Excluded manifest missing: {directory}")
            continue
        for record in json.loads(excluded_manifest.read_text()).get("shards", []):
            sidecar = json.loads((directory / record["sidecar"]).read_text())
            excluded_ids.update(row["patch_id"] for row in sidecar["rows"])
    overlap = sorted(set(seen) & excluded_ids)
    if overlap:
        problems["no_overlap_with_excluded_shards"].extend(overlap[:10])

    caption_coverage = None
    if captions_parquet is not None and captions_parquet.is_file():
        captions = load_captions(captions_parquet, patch_ids)
        caption_coverage = sum(1 for patch_id in patch_ids if captions.get(patch_id)) / len(patch_ids)

    pixel_audit = audit_native_pixels(shard_dir, sidecars, archive_parts, captions_parquet, sample_seed, problems)
    total_rows = sum(record["rows"] for record in records)
    tile_min = min(tile_counts.values()) if tile_counts else 0
    tile_max = max(tile_counts.values()) if tile_counts else 0
    shares = {tile: count / total_rows for tile, count in tile_counts.items()} if total_rows else {}
    shard_tile_min = [record["tile_min"] for record in records if "tile_min" in record]
    shard_tile_max = [record["tile_max"] for record in records if "tile_max" in record]

    report = {
        "kind": "stage-1 shard set verification (loop 9, step 3)",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": f"caption-geo-split.v1/{side}",
        "seed": manifest.get("seed"),
        "device": "CPU (py3.11.15 .venv-stage1)",
        "dtype": manifest.get("dtype"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "shard_dir": str(shard_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": manifest.get("status"),
        "shards_present": len(sidecars),
        "shards_expected": manifest.get("shard_count"),
        "is_complete_prefix": len(sidecars) == len(records) and bool(records),
        "n_patches": total_rows,
        "total_bytes": sum(record["npz_bytes"] for record in records),
        "raw_pixel_bytes": sum(record.get("raw_pixel_bytes") or 0 for record in records),
        "bytes_per_patch": (sum(record["npz_bytes"] for record in records) / total_rows
                            if total_rows else None),
        "seconds_total": sum(record.get("seconds") or 0.0 for record in records),
        "seconds_per_patch_writing": (
            sum(record.get("seconds") or 0.0 for record in records) / total_rows
            if total_rows else None),
        "seconds_elapsed_builder": manifest.get("seconds_elapsed"),
        "seconds_per_patch_builder": manifest.get("seconds_per_patch_overall"),
        "archive_passes": manifest.get("archive_passes"),
        "archive_bytes_streamed": manifest.get("archive_bytes_streamed"),
        "rejected_count_total": manifest.get("rejected_count_total"),
        "rejection_fraction": manifest.get("rejection_fraction_this_run"),
        "rejected_by_reason": manifest.get("rejected_by_reason"),
        "tiles_present": len(tile_counts),
        "tile_counts": dict(sorted(tile_counts.items())),
        "tile_min": tile_min,
        "tile_max": tile_max,
        "per_shard_tile_min": min(shard_tile_min) if shard_tile_min else None,
        "per_shard_tile_max": max(shard_tile_max) if shard_tile_max else None,
        "per_shard_tiles_present_min": min(per_shard_tiles) if per_shard_tiles else None,
        "per_shard_tiles_present_max": max(per_shard_tiles) if per_shard_tiles else None,
        "per_shard_tiles_present_mean": (
            sum(per_shard_tiles) / len(per_shard_tiles) if per_shard_tiles else None),
        "per_shard_tiles_present": per_shard_tiles,
        "prefix_tile_counts": {
            str(len_prefix): {
                "patches": sum(prefix_tile_counts[len_prefix - 1].values()),
                "tiles_present": len(prefix_tile_counts[len_prefix - 1]),
                "min": min(prefix_tile_counts[len_prefix - 1].values()),
                "max": max(prefix_tile_counts[len_prefix - 1].values()),
            } for len_prefix in (1, 5, 10, 25, 50, 75)
            if len_prefix <= len(prefix_tile_counts)},
        "prefix_tile_counts_note": (
            "prefix_tile_counts[k] is the cumulative per-tile count after the first k+1 "
            "shards. A tile drops out of later shards once its available captioning patches "
            "are used, so per-shard tile counts fall from 48 toward 32 while the cumulative "
            "prefix keeps all 48."),
        "caption_coverage": caption_coverage,
        "excluded_shard_ids": len(excluded_ids),
        "pixel_audit": pixel_audit,
        "problem_counts": {name: len(values) for name, values in problems.items()},
        "checks": {name: not values for name, values in problems.items()},
        "problems": {name: values[:20] for name, values in problems.items() if values},
        "all_passed": all(not values for values in problems.values()),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-shards", type=Path, action="append", default=[])
    parser.add_argument("--captions", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--archive-part", type=Path, action="append", required=True)
    parser.add_argument("--sample-seed", type=int, default=26170)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify(args.shards, args.manifest, excluded_dirs=list(args.exclude_shards),
                    captions_parquet=args.captions, split_csv=args.split,
                    archive_parts=args.archive_part, sample_seed=args.sample_seed)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['shards_present']}/{report['shards_expected']} shards, "
          f"{report['n_patches']} patches, {report['total_bytes']} bytes, "
          f"all_passed={report['all_passed']}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())