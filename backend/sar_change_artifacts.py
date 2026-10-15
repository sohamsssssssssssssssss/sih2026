"""Read-only, pinned Kosi change artifacts; no acquisition, ingestion or inference."""

import hashlib
import json
import math
import re
from pathlib import Path

from fastapi import HTTPException

ARTIFACT_DIR = Path.home() / ".local/share/satquery/change-artifacts/kosi-2024"
TIFF_SHA256 = "7d63525d8851c695c9f1e5e72719b80aebd3e6d18581e875bc60ba5d28f6d3a4"
JSON_SHA256 = "792492a47b18111d0aa92a6d613ec3ec0f60a6de557e1ddfec4c597a507d2747"
INPUTS = {
    "before": {
        "product_id": "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE",
        "acquisition_date": "2024-09-24T12:21:46Z",
        "sha256": "973b026d6f19eaa469647eaa23647ab89195512769c1fe23796e9ede59a76af5",
    },
    "after": {
        "product_id": "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE",
        "acquisition_date": "2024-10-06T12:21:47Z",
        "sha256": "643092f78815e0bd735cf76e067b5075bc7fceca97d34043ce1443d378084a8c",
    },
}
GRID = {"epsg": 32645, "width": 749, "height": 1649, "resolution_m": 10,
        "bounds": [432970., 2867120., 440460., 2883610.], "crs": "EPSG:32645"}
FORMULA = {"vv_change_db": "10 * (log10(VV_after) - log10(VV_before))",
           "vh_change_db": "10 * (log10(VH_after) - log10(VH_before))",
           "units": "dB", "intermediate_precision": "float64", "output_dtype": "float32"}
CODE_IDS = {"backend/sentinel1.py": "sentinel1", "models/change/sar_water.py": "sar_water",
            "scripts/s1_offline_change.py": "s1_offline_change"}


def _require(condition):
    if not condition:
        raise ValueError("Invalid artifact metadata")


def _unique_object(pairs):
    value = dict(pairs)
    _require(len(value) == len(pairs))
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_change_artifacts():
    """Verify the immutable bundle on every read and return only allowlisted metadata."""
    raster = ARTIFACT_DIR / "sar_change.tif"
    metadata = ARTIFACT_DIR / "sar_change.json"
    try:
        if not raster.is_file() or not metadata.is_file():
            raise HTTPException(404, "Verified SAR change artifacts are unavailable.")
        metadata_bytes = metadata.read_bytes()
        if _sha256(raster) != TIFF_SHA256 or hashlib.sha256(metadata_bytes).hexdigest() != JSON_SHA256:
            raise HTTPException(409, "SAR change artifact integrity verification failed.")
        report = json.loads(metadata_bytes, object_pairs_hook=_unique_object)
        _require(report["grid"] == GRID and report["formula"] == FORMULA)
        outputs = report["output_artifacts"]
        _require(outputs["sar_change_tif_sha256"] == TIFF_SHA256)
        _require(outputs["bands"] == ["VV_change_dB", "VH_change_dB", "valid_mask"])
        _require(outputs["nodata_value"] == -9999.0)
        for side, expected in INPUTS.items():
            _require(all(report["input_archives"][side][key] == value for key, value in expected.items()))
        coverage = report["validity_criteria"]
        count = coverage["valid_pixels"]
        _require(type(count) is int and 0 < count <= GRID["width"] * GRID["height"])
        _require(type(coverage["total_pixels"]) is int and coverage["total_pixels"] == GRID["width"] * GRID["height"])
        _require(type(coverage["valid_fraction"]) in (int, float) and coverage["valid_fraction"] == count / coverage["total_pixels"])
        statistics = report["statistics"]
        _require(set(statistics) == {"VV_change_dB", "VH_change_dB"})
        for values in statistics.values():
            _require(set(values) == {"count", "min", "max", "mean", "std"})
            _require(type(values["count"]) is int and values["count"] == count)
            _require(all(type(values[key]) in (int, float) and math.isfinite(values[key])
                       for key in ("min", "max", "mean", "std")))
            _require(values["min"] <= values["mean"] <= values["max"] and values["std"] >= 0)
        fingerprints = report["code_fingerprints"]
        _require(all(isinstance(fingerprints[key], str) and re.fullmatch(r"[0-9a-f]{64}", fingerprints[key])
                   for key in CODE_IDS))
        _require(isinstance(report["provenance_caveat"], str) and report["provenance_caveat"].strip())
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise HTTPException(422, "SAR change artifact metadata is invalid.") from None
    except OSError:
        raise HTTPException(503, "SAR change artifacts could not be read.") from None
    return raster, {
        "artifact_id": "kosi-2024-sar-change",
        "scientific_result_type": "sar_backscatter_change",
        "scientific_label": "Measured SAR backscatter change; flood interpretation unvalidated.",
        "flood_interpretation_validated": False,
        "statistics": statistics,
        "formula": FORMULA,
        "grid": GRID,
        "input_archives": INPUTS,
        "output_artifact_hashes": {"sar_change.tif": TIFF_SHA256, "sar_change.json": JSON_SHA256},
        "code_fingerprints": {logical: fingerprints[key] for key, logical in CODE_IDS.items()},
        "coverage": {key: coverage[key] for key in ("valid_pixels", "total_pixels", "valid_fraction")},
        "raster_download_url": "/api/sar/change/kosi-2024/raster",
        "provenance_caveats": [
            "Preserved offline result; no acquisition or inference runs during retrieval.",
            "Processing settings are reconstructed, not independently verified historical request logs.",
            "Input hashes and exact source provenance establish identity, not flood-classification accuracy.",
            "September 24 is a pre-breach observation, not a confirmed dry baseline.",
        ],
    }
