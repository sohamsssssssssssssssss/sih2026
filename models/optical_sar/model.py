"""Deterministic joint Sentinel-2/Sentinel-1 evidence baseline."""

import hashlib
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from models.base import Model, ModelReadiness
from models.paths import public_path


OPTICAL_BANDS = ("B02", "B03", "B04", "B08", "dataMask")
SAR_BANDS = ("VV", "VH", "dataMask")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float | int]:
    percentiles = np.percentile(values, (5, 50, 95))
    return {
        "valid_pixels": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "p05": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
    }


def _validate_descriptions(dataset: Any, expected: tuple[str, ...]) -> None:
    if any(description is not None for description in dataset.descriptions) and tuple(
        dataset.descriptions
    ) != expected:
        raise ValueError(f"Raster band descriptions must be {list(expected)}")


class OpticalSARModel(Model):
    """Extract continuous optical indices and SAR backscatter summaries."""

    name = "optical-sar-deterministic"
    version = "sentinel2-indices__sentinel1-backscatter-v1"

    def readiness(self) -> ModelReadiness:
        for dependency in ("numpy", "rasterio"):
            if find_spec(dependency) is None:
                return ModelReadiness(
                    False,
                    "DEPENDENCY_UNAVAILABLE",
                    f"Required dependency {dependency} is unavailable.",
                )
        return ModelReadiness(True)

    def infer(self, image_paths: list[str], question: str) -> dict[str, Any]:
        if len(image_paths) != 2:
            raise ValueError("Optical-SAR analysis requires exactly two raster paths")
        paths = [Path(value) for value in image_paths]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("Both optical and SAR raster files are required")

        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(path)) for path in paths]
            counts = [dataset.count for dataset in datasets]
            if sorted(counts) != [len(SAR_BANDS), len(OPTICAL_BANDS)]:
                raise ValueError(
                    "Expected one 5-band B02/B03/B04/B08/dataMask raster and one "
                    "3-band VV/VH/dataMask raster"
                )
            optical_index = counts.index(len(OPTICAL_BANDS))
            sar_index = counts.index(len(SAR_BANDS))
            optical, sar = datasets[optical_index], datasets[sar_index]
            _validate_descriptions(optical, OPTICAL_BANDS)
            _validate_descriptions(sar, SAR_BANDS)
            optical_path, sar_path = paths[optical_index], paths[sar_index]
            if (
                optical.width != sar.width
                or optical.height != sar.height
                or optical.crs != sar.crs
                or optical.transform != sar.transform
            ):
                raise ValueError("Optical and SAR rasters must use the same CRS and pixel grid")

            blue, green, red, nir, optical_mask = optical.read()
            vv, vh, sar_mask = sar.read()
            optical_valid = (
                (optical_mask > 0)
                & np.isfinite(blue)
                & np.isfinite(green)
                & np.isfinite(red)
                & np.isfinite(nir)
            )
            sar_valid = (sar_mask > 0) & np.isfinite(vv) & np.isfinite(vh)
            joint_valid = optical_valid & sar_valid
            joint_count = int(np.count_nonzero(joint_valid))
            if joint_count == 0:
                raise ValueError("Optical-SAR pair has no co-valid finite pixels")

            ndvi_denominator = nir + red
            ndwi_denominator = green + nir
            ndvi_valid = optical_valid & (np.abs(ndvi_denominator) > 1e-12)
            ndwi_valid = optical_valid & (np.abs(ndwi_denominator) > 1e-12)
            if not np.any(ndvi_valid) or not np.any(ndwi_valid):
                raise ValueError("Optical raster has no valid NDVI/NDWI denominators")
            ndvi = (nir[ndvi_valid] - red[ndvi_valid]) / ndvi_denominator[ndvi_valid]
            ndwi = (green[ndwi_valid] - nir[ndwi_valid]) / ndwi_denominator[ndwi_valid]

            vv_positive = sar_valid & (vv > 0)
            vh_positive = sar_valid & (vh > 0)
            ratio_valid = vv_positive & vh_positive
            if not np.any(vv_positive) or not np.any(vh_positive) or not np.any(ratio_valid):
                raise ValueError("SAR raster has no valid positive VV/VH samples")
            vv_db = 10.0 * np.log10(vv[vv_positive])
            vh_db = 10.0 * np.log10(vh[vh_positive])
            vv_minus_vh_db = 10.0 * np.log10(vv[ratio_valid] / vh[ratio_valid])
            total = optical.width * optical.height
            evidence = [
                {
                    "type": "optical_statistics",
                    "modality": "optical",
                    "source": {"path": public_path(optical_path), "sha256": _sha256(optical_path)},
                    "bands": list(OPTICAL_BANDS),
                    "valid_pixels": int(np.count_nonzero(optical_valid)),
                    "total_pixels": total,
                    "valid_fraction": int(np.count_nonzero(optical_valid)) / total,
                    "indices": {
                        "ndvi": {"formula": "(B08-B04)/(B08+B04)", **_summary(ndvi)},
                        "ndwi": {"formula": "(B03-B08)/(B03+B08)", **_summary(ndwi)},
                    },
                },
                {
                    "type": "sar_statistics",
                    "modality": "sar",
                    "source": {"path": public_path(sar_path), "sha256": _sha256(sar_path)},
                    "bands": list(SAR_BANDS),
                    "valid_pixels": int(np.count_nonzero(sar_valid)),
                    "total_pixels": total,
                    "valid_fraction": int(np.count_nonzero(sar_valid)) / total,
                    "input_units": "linear backscatter coefficient",
                    "vv_db": _summary(vv_db),
                    "vh_db": _summary(vh_db),
                    "vv_minus_vh_db": _summary(vv_minus_vh_db),
                },
                {
                    "type": "joint_valid_coverage",
                    "modalities": ["optical", "sar"],
                    "valid_pixels": joint_count,
                    "total_pixels": total,
                    "valid_fraction": joint_count / total,
                    "crs": optical.crs.to_string() if optical.crs else None,
                    "width": optical.width,
                    "height": optical.height,
                },
            ]
        return {
            "answer": (
                f"Joint optical-SAR baseline analyzed {joint_count} co-valid pixels; "
                "continuous optical indices and SAR backscatter summaries are in the evidence."
            ),
            "confidence": None,
            "evidence": evidence,
        }
