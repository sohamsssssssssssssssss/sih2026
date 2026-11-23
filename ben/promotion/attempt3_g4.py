"""Collect paired single-date S1 and S2 chips using the read-only Ask acquisition code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

INTEG = Path.home() / "sih2026-integ"
MIN_FREE = 15 * 2**30
TARGET = 100
START, END = "2021-01-01", "2021-04-30"
CHANNELS = {"S1": 2, "S2": 4}


class Skipped(ValueError):
    pass


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align(payload, patch, sensor):
    target = from_origin(patch.utm_x - 600, patch.utm_y + 600, 10, 10)
    pixels = np.full((CHANNELS[sensor], 120, 120), np.nan, dtype=np.float32)
    with MemoryFile(payload) as tile, tile.open() as grid:
        provenance = json.loads(grid.tags()["satquery_provenance"])
        if sensor == "S2":
            cloud = float(grid.tags()["cloud_pct"])
            if not 0 <= cloud < 10:
                raise Skipped(f"S2 cloud cover {cloud:g}% at {provenance['item_id']}")
            provenance["cloud_pct"] = cloud
        if not START <= provenance["acquisition_datetime"][:10] <= END:
            raise Skipped(f"{sensor} acquisition outside the fixed season")
        reproject(grid.read(), pixels, src_crs=grid.crs, src_transform=grid.transform,
                  dst_transform=target, dst_crs=f"EPSG:{patch.utm_epsg}", src_nodata=np.nan,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    if not np.isfinite(pixels).all():
        raise Skipped(f"{sensor} nodata after grid alignment at {provenance['item_id']}")
    provenance["fetched_chip_sha256"] = hashlib.sha256(payload).hexdigest()
    provenance["aligned_pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
    return pixels, provenance


def acquire(client, fresh, patch):
    point = (patch.lon, patch.lat)
    bands, scenes = {}, {}
    for sensor in ("S2", "S1"):
        query = fresh.FreshRequest(lon=point[0], lat=point[1], start_date=START, end_date=END, sensor=sensor)
        try:
            payload = fresh.fetch_chip(client, query, point)
            bands[sensor], scenes[sensor] = align(payload, patch, sensor)
        except (fresh.ProviderError, fresh.SceneUnavailable, Skipped, TimeoutError) as exc:
            return {"patch_id": patch.patch_id, "skipped": f"{sensor}: {exc}"}, None
    return {"patch_id": patch.patch_id, "scenes": scenes, "utm_epsg": int(patch.utm_epsg),
            "utm_x": patch.utm_x, "utm_y": patch.utm_y}, bands


def build_region(rows, labels, directory, region, client, fresh, pool):
    archive, log = directory / f"{region}.npz", directory / f"{region}.json"
    if archive.is_file() and log.is_file():
        recorded = json.loads(log.read_text())
        if sha(archive) != recorded["sha256"]:
            raise ValueError(f"{region} saved archive hash differs")
        return recorded

    if shutil.disk_usage(directory).free < MIN_FREE + 128 * 2**20:
        raise RuntimeError("STOP: disk headroom insufficient for another G4 region")
    accepted, skipped, pixels = [], [], {"S1": [], "S2": []}
    yy = []
    start = 0
    while start < len(rows) and len(accepted) < TARGET:
        candidates = rows[start:start + min(8, TARGET - len(accepted))]
        start += len(candidates)
        results = pool.map(lambda patch: acquire(client, fresh, patch), candidates)
        for patch, (record, arrays) in zip(candidates, results):
            if arrays is None:
                skipped.append(record)
            else:
                accepted.append(record)
                yy.append(labels[patch.idx])
                for sensor in pixels:
                    pixels[sensor].append(arrays[sensor])
    if shutil.disk_usage(directory).free < MIN_FREE + sum(x.nbytes for bands in pixels.values() for x in bands) + 2**20:
        raise RuntimeError("STOP: disk headroom insufficient for G4 output")
    with archive.open("xb") as stream:
        np.savez(stream, patch_id=np.array([row["patch_id"] for row in accepted]),
                 labels=np.asarray(yy, dtype=np.uint8).reshape(len(yy), labels.shape[1]),
                 **{sensor.lower(): np.asarray(values, dtype=np.float32).reshape(-1, CHANNELS[sensor], 120, 120)
                    for sensor, values in pixels.items()})
    record = {"region": region, "file": archive.name, "sha256": sha(archive),
              "bytes": archive.stat().st_size, "fetched": len(accepted), "skipped_count": len(skipped),
              "not_attempted_after_target": len(rows) - len(accepted) - len(skipped),
              "patches": accepted, "skipped": skipped}
    with log.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"{region}: {len(accepted)} paired, {len(skipped)} skipped", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("groups", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    groups = json.loads(args.groups.read_text())
    if groups["status"] != "assigned":
        raise ValueError("STOP: group rule did not produce an assignment")
    sources = {}
    for key, pack in (("old", args.old), ("new", args.new)):
        if sha(pack / "MANIFEST.json") != groups["manifest_sha256"][key]:
            raise ValueError(f"STOP: {key} pack manifest mismatch")
        meta = pd.read_parquet(pack / "meta.parquet")
        labels = np.load(pack / "labels_coarse.npy", allow_pickle=False)
        meta["patch_id"] = [f"{key}:{idx}" for idx in meta.idx]
        sources[key] = meta, labels

    sys.path.insert(0, str(INTEG))
    from backend import fresh_scenes as fresh

    integration_commit = subprocess.check_output(["git", "-C", str(INTEG), "rev-parse", "HEAD"], text=True).strip()
    # The app reads on its native imagery grid. Add four source pixels per side
    # so bilinear alignment onto the pack grid cannot lose an edge to rounding.
    fresh.CHIP_SIZE = 128
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", str(128 * 2**20))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"integration_commit": integration_commit, "season": [START, END],
                "source_chip_pixels": fresh.CHIP_SIZE, "target_chip_pixels": 120,
                "target_per_region": TARGET, "group_sha256": sha(args.groups),
                "label_classes": json.loads((args.old / "classes.json").read_text())["coarse_classes"],
                "regions": []}
    with httpx.Client(timeout=25) as client, ThreadPoolExecutor(max_workers=8) as pool:
        for letter, group in groups["groups"].items():
            for region in group["regions"]:
                key = "old" if letter in "AB" else "new"
                meta, labels = sources[key]
                rows = list(meta[meta.region == region].sort_values("idx").itertuples(index=False))
                if len(rows) != 300:
                    raise ValueError(f"STOP: wrong patch count for {region}")
                record = build_region(rows, labels, args.output, region, client, fresh, pool)
                record = {k: v for k, v in record.items() if k not in ("patches", "skipped")}
                manifest["regions"].append(dict(record, group=letter,
                                                metadata=f"{region}.json",
                                                metadata_sha256=sha(args.output / f"{region}.json")))
    with (args.output / "MANIFEST.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
