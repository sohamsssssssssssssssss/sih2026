"""Acquire one bounded, co-gridded Sentinel-2 L2A / Sentinel-1 RTC case from
Microsoft Planetary Computer, plus the S2 scene classification (SCL) and the
JRC Global Surface Water occurrence layer for the same grid.

Research tooling, not a production provider. Requires `pystac-client` and
`planetary-computer` (anonymous SAS signing; no account or key). SAS tokens
are stripped from every stored href.

The output grid is the native S2 10 m grid of the selected tile, snapped
outward around the requested lon/lat bbox. S2 bands are read without
resampling. S1 is read without resampling when its grid is aligned to the S2
grid; otherwise it is reprojected with bilinear interpolation on linear
backscatter and the provenance says so. SCL (20 m) and GSW (EPSG:4326,
0.00025 deg) use nearest-neighbour onto the S2 grid.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ("B02", "B03", "B04", "B08")
OBSTRUCTED_SCL_NOTE = "SCL is retained unmodified; interpretation happens in eval/sensor_necessity.py"
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff"}
LICENSES = {
    "sentinel-2-l2a": "Copernicus Sentinel data terms (collection license 'proprietary' on Planetary Computer)",
    "sentinel-1-rtc": "CC-BY-4.0 (Planetary Computer sentinel-1-rtc collection)",
    "jrc-gsw": "JRC Global Surface Water, Pekel et al. 2016, Copernicus programme; attribution required",
}


def unsigned(href: str) -> str:
    return href.split("?", 1)[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapped_window(dataset, bounds: tuple[float, float, float, float]) -> Window:
    """Window covering `bounds` in the dataset CRS, snapped outward to whole pixels."""
    window = from_bounds(*bounds, transform=dataset.transform)
    col0, row0 = int(np.floor(window.col_off)), int(np.floor(window.row_off))
    col1 = int(np.ceil(window.col_off + window.width))
    row1 = int(np.ceil(window.row_off + window.height))
    return Window(col0, row0, col1 - col0, row1 - row0)


def grids_aligned(a_transform, b_transform) -> bool:
    """Same pixel size and origins that differ by a whole number of pixels."""
    if (a_transform.a, a_transform.e, a_transform.b, a_transform.d) != (
        b_transform.a, b_transform.e, b_transform.b, b_transform.d
    ):
        return False
    dx = (a_transform.c - b_transform.c) / a_transform.a
    dy = (a_transform.f - b_transform.f) / a_transform.e
    return float(dx).is_integer() and float(dy).is_integer()


def reproject_bounded(ds, destination: np.ndarray, transform, crs, resampling, dst_nodata) -> None:
    """Reproject only a padded source window covering the destination grid.

    `reproject(rasterio.band(ds, 1), ...)` would read the whole remote source
    (a GSW tile is 40000x40000), so read the needed window first.
    """
    height, width = destination.shape
    bounds = transform_bounds(crs, ds.crs, *rasterio.transform.array_bounds(height, width, transform), densify_pts=21)
    window = snapped_window(ds, bounds)
    window = Window(window.col_off - 2, window.row_off - 2, window.width + 4, window.height + 4)
    fill = ds.nodata if ds.nodata is not None else dst_nodata
    source = ds.read(1, window=window, boundless=True, fill_value=fill)
    reproject(source, destination, src_transform=ds.window_transform(window), src_crs=ds.crs,
              src_nodata=ds.nodata, dst_transform=transform, dst_crs=crs,
              dst_nodata=dst_nodata, resampling=resampling)


def _write(path: Path, bands: list[np.ndarray], descriptions: tuple[str, ...], profile: dict) -> None:
    profile = {**profile, "count": len(bands), "dtype": bands[0].dtype.name, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as out:
        for index, (band, description) in enumerate(zip(bands, descriptions), 1):
            out.write(band, index)
            out.set_band_description(index, description)


def acquire(s2_id: str, s1_id: str, bbox: list[float], out_dir: Path) -> dict:
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    s2 = catalog.get_collection("sentinel-2-l2a").get_item(s2_id)
    s1 = catalog.get_collection("sentinel-1-rtc").get_item(s1_id)
    if s2 is None or s1 is None:
        raise LookupError("S2 or S1 item not found in Planetary Computer STAC")
    baseline = s2.properties["s2:processing_baseline"]
    # Processing baseline >= 04.00 adds a -1000 DN offset (BOA_ADD_OFFSET).
    offset = -1000.0 if float(baseline) >= 4.0 else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(s2.assets["B02"].href) as ref:
            crs = ref.crs
            window = snapped_window(ref, transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21))
            transform = ref.window_transform(window)
        height, width = int(window.height), int(window.width)
        profile = {"driver": "GTiff", "crs": crs, "transform": transform, "width": width, "height": height}

        dn = []
        for band in S2_BANDS:
            with rasterio.open(s2.assets[band].href) as ds:
                if ds.transform != ref.transform:
                    raise ValueError(f"S2 band {band} grid differs from B02")
                dn.append(ds.read(1, window=window).astype(np.float32))
        s2_mask = np.all([values > 0 for values in dn], axis=0).astype(np.float32)
        reflectance = [np.where(s2_mask > 0, (values + offset) / 10000.0, np.nan).astype(np.float32) for values in dn]

        scl = np.zeros((height, width), dtype=np.uint8)
        with rasterio.open(s2.assets["SCL"].href) as ds:
            reproject_bounded(ds, scl, transform, crs, Resampling.nearest, 0)

        sar, s1_method = [], None
        for pol in ("vv", "vh"):
            with rasterio.open(s1.assets[pol].href) as ds:
                nodata = ds.nodata
                if ds.crs == crs and grids_aligned(ds.transform, transform):
                    s1_method = "windowed read on identical grid; no resampling"
                    grid_bounds = rasterio.transform.array_bounds(height, width, transform)
                    s1_window = snapped_window(ds, grid_bounds)
                    values = ds.read(1, window=s1_window, boundless=True, fill_value=nodata).astype(np.float32)
                else:
                    s1_method = "bilinear reprojection of linear backscatter onto the S2 grid"
                    values = np.full((height, width), np.nan, dtype=np.float32)
                    reproject_bounded(ds, values, transform, crs, Resampling.bilinear, np.nan)
            values[(values == nodata) | ~np.isfinite(values) | (values <= 0)] = np.nan
            sar.append(values)
        s1_mask = (np.isfinite(sar[0]) & np.isfinite(sar[1])).astype(np.float32)

        gsw_item = next(iter(catalog.search(collections=["jrc-gsw"], bbox=bbox).items()), None)
        if gsw_item is None:
            raise LookupError("No JRC GSW tile covers the bbox")
        occurrence = np.full((height, width), 255, dtype=np.uint8)
        with rasterio.open(gsw_item.assets["occurrence"].href) as ds:
            reproject_bounded(ds, occurrence, transform, crs, Resampling.nearest, 255)

    files = {
        "s2.tif": ([*reflectance, s2_mask], (*S2_BANDS, "dataMask")),
        "s1.tif": ([*sar, s1_mask], ("VV", "VH", "dataMask")),
        "scl.tif": ([scl], ("SCL",)),
        "gsw_occurrence.tif": ([occurrence], ("occurrence",)),
    }
    for name, (bands, descriptions) in files.items():
        _write(out_dir / name, bands, descriptions, profile)

    provenance = {
        "schema_version": "1.0",
        "source": STAC_URL,
        "retrieved_at": retrieved_at,
        "requested_bbox_lonlat": bbox,
        "grid": {"crs": crs.to_string(), "transform": list(transform)[:6], "width": width, "height": height,
                 "definition": "native S2 10 m grid of the selected tile, snapped outward around the bbox"},
        "s2": {"item_id": s2.id, "collection": "sentinel-2-l2a", "datetime": s2.properties["datetime"],
               "product_uri": s2.properties.get("s2:product_uri"), "mgrs_tile": s2.properties.get("s2:mgrs_tile"),
               "processing_baseline": baseline, "dn_offset_applied": offset, "reflectance_scale": 1 / 10000,
               "tile_cloud_cover_percent": s2.properties.get("eo:cloud_cover"),
               "assets": {band: unsigned(s2.assets[band].href) for band in (*S2_BANDS, "SCL")},
               "resampling": "none for B02/B03/B04/B08; nearest 20 m -> 10 m for SCL",
               "dataMask": "1 where all four DN > 0; clouds are NOT masked (same semantics as CDSE dataMask)",
               "license": LICENSES["sentinel-2-l2a"]},
        "s1": {"item_id": s1.id, "collection": "sentinel-1-rtc", "datetime": s1.properties["datetime"],
               "orbit_state": s1.properties.get("sat:orbit_state"), "relative_orbit": s1.properties.get("sat:relative_orbit"),
               "polarizations": s1.properties.get("sar:polarizations"), "instrument_mode": s1.properties.get("sar:instrument_mode"),
               "units": "linear gamma0 terrain-corrected backscatter (Planetary Computer RTC)",
               "assets": {pol: unsigned(s1.assets[pol].href) for pol in ("vv", "vh")},
               "resampling": s1_method, "license": LICENSES["sentinel-1-rtc"]},
        "gsw": {"item_id": gsw_item.id, "collection": "jrc-gsw", "asset": unsigned(gsw_item.assets["occurrence"].href),
                "units": "percent of valid Landsat observations 1984-2020 classified as water; 255 nodata",
                "resampling": "nearest EPSG:4326 0.00025 deg -> S2 grid", "license": LICENSES["jrc-gsw"]},
        "temporal_separation_seconds": abs((s2.datetime - s1.datetime).total_seconds()),
        "scl_note": OBSTRUCTED_SCL_NOTE,
        "files": {name: {"sha256": sha256(out_dir / name), "size_bytes": (out_dir / name).stat().st_size}
                  for name in files},
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s2-id", required=True)
    parser.add_argument("--s1-id", required=True)
    parser.add_argument("--bbox", required=True, type=float, nargs=4, metavar=("W", "S", "E", "N"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    provenance = acquire(args.s2_id, args.s1_id, args.bbox, args.out)
    print(json.dumps({"grid": provenance["grid"], "s1_resampling": provenance["s1"]["resampling"],
                      "files": provenance["files"]}, indent=2))


if __name__ == "__main__":
    main()
