"""Deterministic bi-temporal raster-difference baseline."""

import hashlib
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from models.base import Model, ModelReadiness
from models.change.sar_water import SAR_BANDS, is_sar, water_change
from models.paths import public_path

CHANGE_THRESHOLD = 0.1
MULTISPECTRAL_BANDS = ("B02", "B03", "B04", "B08", "dataMask")


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
        "max": float(np.max(values)),
    }


def _normalized_extent(mask: np.ndarray) -> list[float] | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    height, width = mask.shape
    return [
        float(columns.min() / width),
        float(rows.min() / height),
        float((columns.max() + 1) / width),
        float((rows.max() + 1) / height),
    ]


def _temporal_inputs(paths: list[Path], bands: list[str]) -> dict[str, Any]:
    return {
        "type": "temporal_inputs",
        "t1": {"path": public_path(paths[0]), "sha256": _sha256(paths[0])},
        "t2": {"path": public_path(paths[1]), "sha256": _sha256(paths[1])},
        "bands": bands,
        "temporal_order": "input_1_is_t1_input_2_is_t2",
    }


def _coverage(valid: np.ndarray, dataset: Any) -> dict[str, Any]:
    valid_count = int(np.count_nonzero(valid))
    total = dataset.width * dataset.height
    return {
        "type": "temporal_valid_coverage",
        "valid_pixels": valid_count,
        "total_pixels": total,
        "valid_fraction": valid_count / total,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "width": dataset.width,
        "height": dataset.height,
    }


def _sar_water_result(paths: list[Path], t1: Any, t2: Any) -> dict[str, Any]:
    water = water_change(t1, t2)
    evidence = [_temporal_inputs(paths, list(SAR_BANDS)), *water.evidence, _coverage(water.valid, t1)]
    extent = None if water.new_water is None else _normalized_extent(water.new_water)
    if extent is not None:
        evidence.append(
            {
                "type": "change_extent",
                "coordinates": extent,
                "coordinate_space": "normalized_xyxy",
                "meaning": "bounding extent of all newly water-covered pixels",
            }
        )
    return {"answer": water.answer, "confidence": None, "evidence": evidence}


class ChangeModel(Model):
    name = "change-deterministic"
    version = "bitemporal-difference-v2"

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
            raise ValueError("Bi-temporal change analysis requires exactly two raster paths")
        paths = [Path(value) for value in image_paths]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("Both T1 and T2 raster files are required")

        with ExitStack() as stack:
            t1, t2 = [stack.enter_context(rasterio.open(path)) for path in paths]
            if (
                t1.count != t2.count
                or t1.width != t2.width
                or t1.height != t2.height
                or t1.crs != t2.crs
                or t1.transform != t2.transform
            ):
                raise ValueError("T1 and T2 must have identical bands, CRS, and pixel grid")
            sar = [is_sar(dataset) for dataset in (t1, t2)]
            if any(sar):
                if not all(sar):
                    raise ValueError("T1 and T2 must both be SAR (VV/VH/dataMask) or both optical")
                return _sar_water_result(paths, t1, t2)
            if t1.count not in {3, len(MULTISPECTRAL_BANDS)}:
                raise ValueError("Change baseline requires 3-band RGB or 5-band multispectral rasters")
            first, second = t1.read(), t2.read()
            masks = np.all(t1.read_masks() > 0, axis=0) & np.all(
                t2.read_masks() > 0, axis=0
            )

            if t1.count == len(MULTISPECTRAL_BANDS):
                for dataset in (t1, t2):
                    if any(value is not None for value in dataset.descriptions) and tuple(
                        dataset.descriptions
                    ) != MULTISPECTRAL_BANDS:
                        raise ValueError(
                            f"Multispectral band descriptions must be {list(MULTISPECTRAL_BANDS)}"
                        )
                masks &= (first[4] > 0) & (second[4] > 0)
                required = [1, 2, 3]
                masks &= np.all(np.isfinite(first[required]), axis=0)
                masks &= np.all(np.isfinite(second[required]), axis=0)
                green1, red1, nir1 = first[1], first[2], first[3]
                green2, red2, nir2 = second[1], second[2], second[3]
                ndvi_den1, ndvi_den2 = nir1 + red1, nir2 + red2
                ndwi_den1, ndwi_den2 = green1 + nir1, green2 + nir2
                masks &= (
                    (np.abs(ndvi_den1) > 1e-12)
                    & (np.abs(ndvi_den2) > 1e-12)
                    & (np.abs(ndwi_den1) > 1e-12)
                    & (np.abs(ndwi_den2) > 1e-12)
                )
                if not np.any(masks):
                    raise ValueError("T1 and T2 have no co-valid spectral-index pixels")
                delta_ndvi = (
                    (nir2[masks] - red2[masks]) / ndvi_den2[masks]
                    - (nir1[masks] - red1[masks]) / ndvi_den1[masks]
                )
                delta_ndwi = (
                    (green2[masks] - nir2[masks]) / ndwi_den2[masks]
                    - (green1[masks] - nir1[masks]) / ndwi_den1[masks]
                )
                magnitude_values = np.clip(
                    (np.abs(delta_ndvi) + np.abs(delta_ndwi)) / 4.0, 0.0, 1.0
                )
                method = "spectral_index_delta"
                method_detail = {
                    "delta_ndvi": _summary(delta_ndvi),
                    "delta_ndwi": _summary(delta_ndwi),
                }
                bands = list(MULTISPECTRAL_BANDS)
            else:
                masks &= np.all(np.isfinite(first), axis=0) & np.all(
                    np.isfinite(second), axis=0
                )
                if not np.any(masks):
                    raise ValueError("T1 and T2 have no co-valid RGB pixels")
                differences = []
                for band in range(3):
                    shared = np.concatenate((first[band][masks], second[band][masks]))
                    low, high = np.percentile(shared, (2, 98))
                    if high <= low:
                        differences.append(np.zeros(int(np.count_nonzero(masks))))
                    else:
                        normalized1 = np.clip((first[band][masks] - low) / (high - low), 0, 1)
                        normalized2 = np.clip((second[band][masks] - low) / (high - low), 0, 1)
                        differences.append(np.abs(normalized2 - normalized1))
                magnitude_values = np.mean(differences, axis=0)
                method = "normalized_rgb_difference_heuristic"
                method_detail = {
                    "normalization": "shared per-band p02-p98 clip",
                    "semantic_interpretation": False,
                }
                bands = ["red", "green", "blue"]

            changed_values = magnitude_values > CHANGE_THRESHOLD
            changed_mask = np.zeros(masks.shape, dtype=bool)
            changed_mask[masks] = changed_values
            valid_count = int(np.count_nonzero(masks))
            changed_count = int(np.count_nonzero(changed_values))
            evidence = [
                _temporal_inputs(paths, bands),
                {
                    "type": "change_statistics",
                    "method": method,
                    "heuristic": True,
                    "change_magnitude": _summary(magnitude_values),
                    "threshold": CHANGE_THRESHOLD,
                    "threshold_rule": "changed when normalized change magnitude > 0.1",
                    "changed_pixels": changed_count,
                    "changed_fraction": changed_count / valid_count,
                    **method_detail,
                },
                _coverage(masks, t1),
            ]
            extent = _normalized_extent(changed_mask)
            if extent is not None:
                evidence.append(
                    {
                        "type": "change_extent",
                        "coordinates": extent,
                        "coordinate_space": "normalized_xyxy",
                        "meaning": "bounding extent of all heuristic changed pixels",
                    }
                )

        return {
            "answer": (
                f"Deterministic change baseline found {changed_count} of {valid_count} "
                "co-valid pixels above the heuristic threshold. It does not infer "
                "semantic change classes or causes."
            ),
            "confidence": None,
            "evidence": evidence,
        }
