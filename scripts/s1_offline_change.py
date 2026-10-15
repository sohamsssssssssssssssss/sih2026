#!/usr/bin/env python3
"""Offline Sentinel-1 capture-to-change workflow for the Kosi 2024 pair.

Reuses production TAR parsing, provenance validation, and raster validation.
Computes VV/VH log-ratio change in dB from exact pinned acquisitions.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import sentinel1 as s1


PINNED_BEFORE = "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE"
PINNED_AFTER = "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE"

EXPECTED_GRID = {
    "width": 749,
    "height": 1649,
    "epsg": 32645,
    "resolution_m": 10,
}

SAR_BANDS = ("VV", "VH", "dataMask")
NODATA_VALUE = -9999.0


class ChangeError(RuntimeError):
    """Offline change computation failed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ChangeError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def parse_tar_response(content: bytes) -> tuple[bytes, bytes]:
    """Use production validation with the captured-response EOF compatibility mode."""
    try:
        return s1._parse_tar_response(content, allow_missing_eof=True)
    except s1.CDSEError as exc:
        raise ChangeError(str(exc)) from exc


def parse_userdata(userdata_bytes: bytes, expected_product_id: str) -> dict[str, Any]:
    """Validate the exact pinned source set with the production validator."""
    try:
        return s1._parse_userdata(userdata_bytes, expected_product_id)
    except s1.CDSEError as exc:
        raise ChangeError(str(exc)) from exc


def on_grid(content: bytes, expected_grid: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Adapt the runner's grid schema to production validation."""
    grid = s1.Grid(expected_grid["epsg"], tuple(expected_grid["bounds"]),
                   expected_grid["width"], expected_grid["height"])
    try:
        return s1._on_grid(content, grid)
    except s1.CDSEError as exc:
        raise ChangeError(str(exc)) from exc


def load_tar_archive(tar_path: Path, expected_product_id: str, expected_grid: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load TAR archive, validate, and return pixels (3, H, W) and profile."""
    content = tar_path.read_bytes()
    tiff_bytes, userdata_bytes = parse_tar_response(content)
    parse_userdata(userdata_bytes, expected_product_id)
    profile, pixels = on_grid(tiff_bytes, expected_grid)
    return pixels, profile


def compute_change(vv_before: np.ndarray, vh_before: np.ndarray,
                   vv_after: np.ndarray, vh_after: np.ndarray,
                   mask_before: np.ndarray, mask_after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute VV and VH change in dB and validity mask.

    Args:
        vv_before, vh_before, vv_after, vh_after: float32 arrays in linear power
        mask_before, mask_after: float32 valid masks (1 = valid, 0 = invalid)

    Returns:
        vv_change_db, vh_change_db, valid_mask (bool)
    """
    # Valid where both masks are 1, finite, and both VV/VH are positive
    valid = (
        (mask_before > 0) & (mask_after > 0) &
        np.isfinite(mask_before) & np.isfinite(mask_after) &
        np.isfinite(vv_before) & np.isfinite(vv_after) &
        np.isfinite(vh_before) & np.isfinite(vh_after) &
        (vv_before > 0) & (vv_after > 0) &
        (vh_before > 0) & (vh_after > 0)
    )

    # Compute in float64 for precision
    vv_before_f64 = vv_before.astype(np.float64)
    vv_after_f64 = vv_after.astype(np.float64)
    vh_before_f64 = vh_before.astype(np.float64)
    vh_after_f64 = vh_after.astype(np.float64)

    vv_change_db = np.full_like(vv_before, NODATA_VALUE, dtype=np.float32)
    vh_change_db = np.full_like(vh_before, NODATA_VALUE, dtype=np.float32)

    # 10 * (log10(after) - log10(before)) = 10 * log10(after / before)
    vv_change_db[valid] = (10.0 * (np.log10(vv_after_f64[valid]) - np.log10(vv_before_f64[valid]))).astype(np.float32)
    vh_change_db[valid] = (10.0 * (np.log10(vh_after_f64[valid]) - np.log10(vh_before_f64[valid]))).astype(np.float32)

    return vv_change_db, vh_change_db, valid


def descriptive_stats(arr: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    """Compute descriptive statistics for valid pixels only."""
    vals = arr[valid]
    if vals.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def write_geotiff(output_path: Path, vv_change: np.ndarray, vh_change: np.ndarray,
                  valid_mask: np.ndarray, profile: dict[str, Any], crs: CRS) -> None:
    """Write three-band change GeoTIFF with nodata convention."""
    change_stack = np.stack([vv_change, vh_change, valid_mask.astype(np.float32)], axis=0)
    out_profile = {
        **profile,
        "count": 3,
        "dtype": "float32",
        "nodata": NODATA_VALUE,
        "compress": "lzw",
    }
    with output_path.open("wb") as f:
        with MemoryFile() as memfile:
            with memfile.open(**out_profile) as dst:
                dst.write(change_stack)
                dst.descriptions = ("VV_change_dB", "VH_change_dB", "valid_mask")
            f.write(memfile.read())


def write_json(output_path: Path, report: dict[str, Any]) -> None:
    """Write JSON report atomically."""
    tmp_path = output_path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(report, f, indent=2)
    tmp_path.replace(output_path)


def get_code_fingerprints() -> dict[str, str]:
    """Generate fingerprints for relevant code/config files."""
    files = [
        "backend/sentinel1.py",
        "models/change/sar_water.py",
        "scripts/s1_offline_change.py",
    ]
    fingerprints = {}
    for rel_path in files:
        path = ROOT / rel_path
        if path.exists():
            h = hashlib.sha256()
            h.update(path.read_bytes())
            fingerprints[rel_path] = h.hexdigest()
    return fingerprints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="BEFORE TAR archive path")
    parser.add_argument("--after", required=True, type=Path, help="AFTER TAR archive path")
    parser.add_argument("--before-sha256", required=True, help="Expected SHA256 of BEFORE archive")
    parser.add_argument("--after-sha256", required=True, help="Expected SHA256 of AFTER archive")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    args = parser.parse_args(argv)

    # Verify checksums
    verify_checksum(args.before, args.before_sha256, "BEFORE")
    verify_checksum(args.after, args.after_sha256, "AFTER")

    # Expected grid from flood-runs manifest
    flood_runs = json.loads(s1.FLOOD_RUNS_PATH.read_text(encoding="utf-8"))
    event = next(e for e in flood_runs["events"] if e["event_id"] == "kosi-2024")
    event_grid = s1.utm_grid(s1.event_request("kosi-2024")[0])
    expected_grid = {
        "width": event["grid"]["width"],
        "height": event["grid"]["height"],
        "epsg": event["grid"]["epsg"],
        "resolution_m": s1.PIXEL_SIZE_M,
        "bounds": tuple(event["grid"].get("bounds", event_grid.bounds)),
    }

    # Load and validate both archives
    pixels_before, profile_before = load_tar_archive(args.before, PINNED_BEFORE, expected_grid)
    pixels_after, profile_after = load_tar_archive(args.after, PINNED_AFTER, expected_grid)

    # Extract bands: VV(0), VH(1), dataMask(2)
    vv_before, vh_before, mask_before = pixels_before[0], pixels_before[1], pixels_before[2]
    vv_after, vh_after, mask_after = pixels_after[0], pixels_after[1], pixels_after[2]

    # Compute change
    vv_change, vh_change, valid_mask = compute_change(
        vv_before, vh_before, vv_after, vh_after, mask_before, mask_after
    )

    # Statistics
    valid_count = int(valid_mask.sum())
    total_pixels = expected_grid["width"] * expected_grid["height"]

    # Write outputs to temporary directory first
    with tempfile.TemporaryDirectory(prefix="s1_change_") as tmpdir:
        tmp_tif = Path(tmpdir) / "sar_change.tif"
        tmp_json = Path(tmpdir) / "sar_change.json"

        write_geotiff(tmp_tif, vv_change, vh_change, valid_mask, profile_before, CRS.from_epsg(expected_grid["epsg"]))

        # Compute output SHA256
        output_sha256 = sha256_file(tmp_tif)

        # Build JSON report
        report = {
            "input_archives": {
                "before": {
                    "path": str(args.before),
                    "sha256": args.before_sha256,
                    "product_id": PINNED_BEFORE,
                    "acquisition_date": "2024-09-24T12:21:46Z",
                },
                "after": {
                    "path": str(args.after),
                    "sha256": args.after_sha256,
                    "product_id": PINNED_AFTER,
                    "acquisition_date": "2024-10-06T12:21:47Z",
                },
            },
            "grid": {
                "epsg": expected_grid["epsg"],
                "width": expected_grid["width"],
                "height": expected_grid["height"],
                "resolution_m": expected_grid["resolution_m"],
                "bounds": list(expected_grid["bounds"]),
                "crs": f"EPSG:{expected_grid['epsg']}",
            },
            "formula": {
                "vv_change_db": "10 * (log10(VV_after) - log10(VV_before))",
                "vh_change_db": "10 * (log10(VH_after) - log10(VH_before))",
                "units": "dB",
                "intermediate_precision": "float64",
                "output_dtype": "float32",
            },
            "validity_criteria": {
                "description": "Pixel is valid iff both observations have valid_mask=1, finite values, and positive VV/VH",
                "valid_pixels": valid_count,
                "total_pixels": total_pixels,
                "valid_fraction": valid_count / total_pixels,
            },
            "statistics": {
                "VV_change_dB": descriptive_stats(vv_change, valid_mask),
                "VH_change_dB": descriptive_stats(vh_change, valid_mask),
            },
            "output_artifacts": {
                "sar_change_tif_sha256": output_sha256,
                "bands": ["VV_change_dB", "VH_change_dB", "valid_mask"],
                "nodata_value": NODATA_VALUE,
            },
            "code_fingerprints": get_code_fingerprints(),
            "provenance_caveat": (
                "Reconstructed processing settings are not independently verified historical request logs. "
                "The TAR archives were captured via the CDSE Process API with the following nominal settings: "
                "GAMMA0_TERRAIN calibration, COPERNICUS_30 DEM orthorectification, LEE speckle filter (5x5), "
                "IW mode, DV polarization, HIGH resolution, ORBIT mosaicking. "
                "Exact request parameters are recorded in the captured TAR's userdata.json."
            ),
        }

        write_json(tmp_json, report)

        # Atomic publish to output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        final_tif = args.output_dir / "sar_change.tif"
        final_json = args.output_dir / "sar_change.json"
        shutil.move(tmp_tif, final_tif)
        shutil.move(tmp_json, final_json)

        print(f"Wrote {final_tif} ({final_tif.stat().st_size} bytes)")
        print(f"Wrote {final_json}")
        print(f"Output SHA256: {output_sha256}")
        print(f"Valid pixels: {valid_count}/{total_pixels} ({valid_count/total_pixels*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())