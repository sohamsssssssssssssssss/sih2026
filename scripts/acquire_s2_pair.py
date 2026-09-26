"""Acquire bounded Sentinel-2 L2A T1/T2 windows from the public Earth Search STAC.

Research acquisition only: no credentials, no resampling of the analysed bands.
Both observations are read from their own native MGRS tile grid. When T1 and T2
share a tile grid, T2 reuses T1's exact pixel window, so the pair is aligned by
construction rather than by forced co-registration. Observations on different
grids are written as-is and left for the canonical pair gate to reject.

Output per pair (outside git): t1.tif, t2.tif (float32 B02/B03/B04/B08/dataMask
surface reflectance) and t1_scl.tif, t2_scl.tif (uint8 Scene Classification,
nearest-upsampled 20 m -> 10 m for diagnostics only). Provenance JSON is written
to data/manifests/change/.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
STAC_ROOT = "https://earth-search.aws.element84.com/v1"
SPEC_PATH = ROOT / "data" / "manifests" / "change" / "pairs.v1.json"
PROVENANCE_DIR = ROOT / "data" / "manifests" / "change"
DEFAULT_OUTPUT = Path.home() / "satquery-data" / "change"
BANDS = ("B02", "B03", "B04", "B08")
DESCRIPTIONS = (*BANDS, "dataMask")
MAX_WINDOW_PIXELS = 800  # per side: keeps a 5-band float32 pair under the 20 MiB API upload cap
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_item(collection: str, item_id: str) -> dict:
    response = requests.get(f"{STAC_ROOT}/collections/{collection}/items/{item_id}", timeout=90)
    response.raise_for_status()
    return response.json()


def even_window(window: Window, tile_shape: tuple[int, int], limit: int = MAX_WINDOW_PIXELS) -> Window:
    """Snap to even offsets/sizes so the 20 m SCL grid nests exactly in the 10 m grid.

    Fails closed when the AOI is not fully inside the tile: rasterio silently
    clips such reads, and writing the clipped block into the full-size window
    stretches it (the defect that invalidated water-puzhal-lake-2018-2019).
    """
    col = 2 * math.floor(window.col_off / 2)
    row = 2 * math.floor(window.row_off / 2)
    width = 2 * math.ceil((window.col_off + window.width - col) / 2)
    height = 2 * math.ceil((window.row_off + window.height - row) / 2)
    if width > limit or height > limit:
        raise ValueError(f"AOI window {width}x{height} exceeds {limit}x{limit} pixels")
    if col < 0 or row < 0 or col + width > tile_shape[1] or row + height > tile_shape[0]:
        raise ValueError(f"AOI window {(col, row, width, height)} is not inside the {tile_shape} tile; choose a tile that contains the AOI")
    return Window(col, row, width, height)


def reflectance(dn: np.ndarray, scale: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the STAC-declared scale/offset; DN 0 is L2A nodata and is masked, not converted."""
    valid = dn != 0
    values = np.where(valid, dn.astype(np.float64) * scale + offset, 0.0).astype(np.float32)
    return values, valid


def upsample_scl(scl: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)


def _bytes_touched(dataset, window: Window) -> int:
    """Compressed bytes of every native block the window intersects (transfer estimate)."""
    rows, cols = dataset.block_shapes[0]
    total = 0
    for i in range(int(window.row_off) // rows, math.ceil((window.row_off + window.height) / rows)):
        for j in range(int(window.col_off) // cols, math.ceil((window.col_off + window.width) / cols)):
            total += dataset.block_size(1, i, j)
    return total


def _read_asset(href: str, window: Window) -> dict:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as dataset:
        data = dataset.read(1, window=window)
        if data.shape != (int(window.height), int(window.width)):
            raise ValueError(f"{href}: read {data.shape} for window {window}; refusing a clipped read")
        return {
            "data": data,
            "crs": dataset.crs,
            "transform": dataset.window_transform(window),
            "native_transform": dataset.transform,
            "bytes_touched": _bytes_touched(dataset, window),
        }


def _grid_window(item: dict, aoi: list[float]) -> tuple[Window, rasterio.Affine, str]:
    href = item["assets"]["red"]["href"]
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as dataset:
        bounds = transform_bounds("EPSG:4326", dataset.crs, *aoi, densify_pts=21)
        window = even_window(from_bounds(*bounds, transform=dataset.transform), dataset.shape)
        return window, dataset.transform, dataset.crs.to_string()


def acquire_observation(item: dict, window: Window, output: Path, label: str) -> dict:
    assets = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
    scl_window = Window(window.col_off / 2, window.row_off / 2, window.width / 2, window.height / 2)
    jobs = {band: (item["assets"][key]["href"], window) for band, key in assets.items()}
    jobs["SCL"] = (item["assets"]["scl"]["href"], scl_window)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        reads = dict(zip(jobs, pool.map(lambda job: _read_asset(*job), jobs.values())))

    reference = reads["B04"]
    if any(reads[band]["transform"] != reference["transform"] for band in BANDS):
        raise ValueError(f"{item['id']}: 10 m bands disagree on window transform")
    stack, valid = [], np.ones(reads["B04"]["data"].shape, dtype=bool)
    scaling = {}
    for band in BANDS:
        declared = item["assets"][assets[band]]["raster:bands"][0]
        scale, offset = float(declared.get("scale", 1.0)), float(declared.get("offset", 0.0))
        scaling[band] = {"scale": scale, "offset": offset}
        values, band_valid = reflectance(reads[band]["data"], scale, offset)
        stack.append(values)
        valid &= band_valid
    stack.append(valid.astype(np.float32))
    scl = upsample_scl(reads["SCL"]["data"])
    if scl.shape != valid.shape:
        raise ValueError(f"{item['id']}: SCL window {scl.shape} does not nest in {valid.shape}")

    profile = {
        "driver": "GTiff", "width": int(window.width), "height": int(window.height),
        "crs": reference["crs"], "transform": reference["transform"], "tiled": True,
        "blockxsize": 256, "blockysize": 256, "compress": "deflate",
    }
    raster_path, scl_path = output / f"{label}.tif", output / f"{label}_scl.tif"
    # ponytail: no nodata tag; reflectance 0.0 is a valid value, validity lives in dataMask.
    with rasterio.open(raster_path, "w", count=5, dtype="float32", predictor=3, **profile) as dst:
        dst.write(np.stack(stack))
        for index, name in enumerate(DESCRIPTIONS, 1):
            dst.set_band_description(index, name)
    with rasterio.open(scl_path, "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(scl, 1)
        dst.set_band_description(1, "SCL")

    properties = item["properties"]
    red_dn = reads["B04"]["data"][reads["B04"]["data"] > 0]
    classes, counts = np.unique(scl, return_counts=True)
    return {
        "label": label,
        "stac_item_id": item["id"],
        "collection": item["collection"],
        "acquisition_time": properties["datetime"],
        "platform": properties.get("platform"),
        "processing_baseline": properties.get("s2:processing_baseline"),
        "product_uri": properties.get("s2:product_uri"),
        "granule_id": properties.get("s2:granule_id"),
        "mgrs_tile": properties.get("grid:code"),
        "tile_cloud_cover_percent": properties.get("eo:cloud_cover"),
        "sun_elevation_deg": properties.get("view:sun_elevation"),
        "sun_azimuth_deg": properties.get("view:sun_azimuth"),
        "asset_hrefs": {band: jobs[band][0] for band in jobs},
        "asset_file_checksums": {
            band: item["assets"][assets.get(band, "scl")].get("file:checksum") for band in jobs
        },
        "declared_scaling": scaling,
        "observed_red_dn_p01": float(np.percentile(red_dn, 1)) if red_dn.size else None,
        "native_window": _window_record(window),
        "estimated_bytes_transferred": sum(read["bytes_touched"] for read in reads.values()),
        "aoi_scl_histogram": {int(k): int(v) for k, v in zip(classes, counts)},
        "aoi_nodata_fraction": float(1 - valid.mean()),
        "raster": _raster_record(raster_path),
        "scl_raster": {"path": scl_path.name, "sha256": sha256(scl_path),
                       "derivation": "SCL 20 m window, nearest 2x2 replication to 10 m"},
    }


def _raster_record(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        return {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "crs": dataset.crs.to_string(),
            "transform": list(dataset.transform)[:6],
            "bounds": list(dataset.bounds),
            "width": dataset.width,
            "height": dataset.height,
            "resolution_m": list(dataset.res),
            "dtype": dataset.dtypes[0],
            "band_descriptions": list(dataset.descriptions),
            "nodata_tag": dataset.nodata,
            "units": "surface reflectance = DN * scale + offset (STAC raster:bands)",
            "mask_semantics": "dataMask=1 where all four bands have non-zero DN; clouds are NOT masked",
        }


def _window_record(window: Window) -> dict:
    return {"col_off": int(window.col_off), "row_off": int(window.row_off),
            "width": int(window.width), "height": int(window.height)}


def reuse_observation(item_id: str, window: Window, output: Path, label: str) -> dict | None:
    """Copy an identical (item, native window) observation from another acquired pair instead of re-downloading.

    The copy is accepted only if both files still match their recorded SHA-256.
    """
    for path in sorted(PROVENANCE_DIR.glob("*.provenance.json")):
        provenance = json.loads(path.read_text())
        for observation in provenance["observations"]:
            if observation["stac_item_id"] != item_id or observation["native_window"] != _window_record(window):
                continue
            source = output.parent / provenance["pair_id"]
            copies = {
                source / observation["raster"]["path"]: (output / f"{label}.tif", observation["raster"]["sha256"]),
                source / observation["scl_raster"]["path"]: (output / f"{label}_scl.tif", observation["scl_raster"]["sha256"]),
            }
            if not all(src.is_file() and sha256(src) == digest for src, (_, digest) in copies.items()):
                continue
            for src, (dst, _) in copies.items():
                shutil.copyfile(src, dst)
            return {
                **observation,
                "label": label,
                "raster": {**observation["raster"], "path": f"{label}.tif"},
                "scl_raster": {**observation["scl_raster"], "path": f"{label}_scl.tif"},
                "estimated_bytes_transferred": 0,
                "reused_from": {"pair_id": provenance["pair_id"], "label": observation["label"],
                                "original_retrieval": provenance["retrieved_at"]},
            }
    return None


def acquire_pair(spec: dict, output_root: Path) -> dict:
    collection = spec["collection"]
    items = [fetch_item(collection, spec[key]) for key in ("t1_item", "t2_item")]
    windows = [_grid_window(item, spec["aoi_lonlat"]) for item in items]
    same_grid = windows[0][1] == windows[1][1] and windows[0][2] == windows[1][2]
    shift = spec.get("t2_window_shift_px", [0, 0])
    t1_window = windows[0][0]
    t2_window = t1_window if same_grid else windows[1][0]
    if shift != [0, 0]:
        t2_window = Window(t2_window.col_off + shift[0], t2_window.row_off + shift[1],
                           t2_window.width, t2_window.height)
    output = output_root / spec["pair_id"]
    output.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat()
    observations = [
        reuse_observation(item["id"], window, output, label) or acquire_observation(item, window, output, label)
        for item, window, label in ((items[0], t1_window, "t1"), (items[1], t2_window, "t2"))
    ]
    return {
        "schema": "satquery.change_pair_provenance.v1",
        "pair_id": spec["pair_id"],
        "spec": spec,
        "source": {"stac_api": STAC_ROOT, "collection": collection,
                   "access": "public, anonymous HTTPS range reads of COGs"},
        "retrieved_at": retrieved,
        "same_native_grid": same_grid,
        "window_policy": (
            "T2 reuses T1's exact native pixel window" if same_grid and shift == [0, 0]
            else f"per-tile windows computed independently from the AOI (same_grid={same_grid}, t2 shift={shift}); "
                 "no resampling; the canonical pair gate decides compatibility"
        ),
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pair_ids", nargs="+")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    specs = {entry["pair_id"]: entry for entry in json.loads(args.spec.read_text())["pairs"]}
    for pair_id in args.pair_ids:
        provenance = acquire_pair(specs[pair_id], args.output_root)
        target = PROVENANCE_DIR / f"{pair_id}.provenance.json"
        target.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        transferred = sum(o["estimated_bytes_transferred"] for o in provenance["observations"])
        print(f"{pair_id}: wrote {target.relative_to(ROOT)} (~{transferred / 1e6:.1f} MB transferred)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
