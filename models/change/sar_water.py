"""Sentinel-1 open-water change: the SAR path of the bi-temporal change baseline.

Water is one Otsu threshold on VV backscatter in dB, pooled over both dates so
"water" means the same thing at T1 and T2. This is the baseline a trained
segmenter has to beat on held-out IoU. To swap one in, replace ``water_masks``
with a function returning ``(water_t1, water_t2, threshold_db)``; the
threshold and abstention fields are specific to this Otsu baseline.

Inputs are co-gridded VV/VH/dataMask rasters in linear backscatter, the same
band contract as the optical-SAR provider.
"""

from typing import Any, NamedTuple

import numpy as np
from rasterio.features import shapes, sieve
from rasterio.warp import transform, transform_geom

SAR_BANDS = ("VV", "VH", "dataMask")
OTSU_BINS = 256
# A handful of extreme pixels (radar shadow, calibration artefacts) must not stretch the bins.
OTSU_CLIP_PERCENTILES = (0.1, 99.9)
# ponytail: a fixed physical prior, not a calibrated value. Open water in
# C-band VV sits well below -15 dB, so a pooled Otsu split above it means the
# scene has no open-water mode to separate. Phase 4 calibration replaces this.
WATER_DB_CEILING = -15.0
# Connected change regions smaller than this are treated as speckle (0.1 ha at 10 m).
MIN_MAPPING_PIXELS = 10
MAX_GEOJSON_FEATURES = 500
EQUAL_AREA_CRS = "EPSG:6933"  # matches data/pairing.py
GEOJSON_CRS = "EPSG:4326"  # RFC 7946 GeoJSON is always WGS84 lon/lat
SQUARE_METRES_PER_HECTARE = 10_000


class WaterChange(NamedTuple):
    answer: str
    evidence: list[dict[str, Any]]
    valid: np.ndarray
    new_water: np.ndarray | None


def is_sar(dataset: Any) -> bool:
    return tuple(dataset.descriptions) == SAR_BANDS


def otsu_threshold(values: np.ndarray) -> float:
    """Histogram split that maximises between-class variance."""
    low, high = np.percentile(values, OTSU_CLIP_PERCENTILES)
    counts, edges = np.histogram(np.clip(values, low, high), bins=OTSU_BINS)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_low = np.cumsum(counts)[:-1]
    weight_high = counts.sum() - weight_low
    mass_low = np.cumsum(counts * centers)[:-1]
    mean_low = mass_low / np.maximum(weight_low, 1)
    mean_high = (np.sum(counts * centers) - mass_low) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(edges[np.argmax(between) + 1])


def water_masks(
    db_t1: np.ndarray, db_t2: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = otsu_threshold(np.concatenate((db_t1[valid], db_t2[valid])))
    return (db_t1 < threshold) & valid, (db_t2 < threshold) & valid, threshold


def _row_pixel_area_m2(grid: Any, crs: Any, height: int) -> np.ndarray:
    """True ground area of one pixel per row, from its corners in an equal-area CRS.

    ponytail: column 0 stands in for the whole row. That is exact for lat/lon
    grids and within about 0.2% across a UTM zone; go per-pixel if that matters.
    """
    rows = np.arange(height, dtype=float)
    left, right = np.zeros(height), np.ones(height)
    corners = [grid @ (left, rows), grid @ (right, rows), grid @ (right, rows + 1), grid @ (left, rows + 1)]
    xs, ys = transform(crs, EQUAL_AREA_CRS, *(np.concatenate(axis) for axis in zip(*corners)))
    xs, ys = np.reshape(xs, (4, height)), np.reshape(ys, (4, height))
    area = 0.5 * np.abs(np.sum(xs * np.roll(ys, -1, axis=0) - np.roll(xs, -1, axis=0) * ys, axis=0))
    if not np.all(np.isfinite(area)):
        raise ValueError(f"Pixel areas cannot be computed in {EQUAL_AREA_CRS} for this raster")
    return area


def _hectares(mask: np.ndarray, row_area: np.ndarray) -> float:
    return float(np.sum(mask.sum(axis=1) * row_area) / SQUARE_METRES_PER_HECTARE)


def _sieved(mask: np.ndarray) -> np.ndarray:
    """Drop specks. Sieve also fills small gaps, so AND with the input: it may only remove."""
    return sieve(mask.astype(np.uint8), size=MIN_MAPPING_PIXELS).astype(bool) & mask


def _ring_area(ring: list[tuple[float, float]]) -> float:
    x, y = np.asarray(ring).T
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _polygons(masks: dict[str, np.ndarray], dataset: Any) -> dict[str, Any]:
    """Largest change polygons first, in WGS84, with an honest truncation count."""
    pixel_area = abs(dataset.transform.determinant)
    found = []
    for change, mask in masks.items():
        for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=dataset.transform):
            rings = geometry["coordinates"]
            area = _ring_area(rings[0]) - sum(_ring_area(ring) for ring in rings[1:])
            found.append((round(area / pixel_area), change, geometry))
    found.sort(key=lambda item: item[0], reverse=True)
    kept = found[:MAX_GEOJSON_FEATURES]
    features = [
        {
            "type": "Feature",
            "properties": {"change": change, "pixels": pixels},
            "geometry": transform_geom(dataset.crs, GEOJSON_CRS, geometry),
        }
        for pixels, change, geometry in kept
    ]
    return {
        "type": "water_change_polygons",
        "format": "geojson",
        "crs": GEOJSON_CRS,
        "feature_count": len(found),
        "returned_feature_count": len(kept),
        "truncated": len(kept) < len(found),
        "ordering": "largest first",
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def water_change(t1: Any, t2: Any) -> WaterChange:
    """Open-water appearance and disappearance between two co-gridded SAR dates."""
    if t1.crs is None:
        raise ValueError("SAR water mapping requires georeferenced rasters with a CRS")
    if t1.transform.b or t1.transform.d:
        raise ValueError("SAR water mapping requires a north-up pixel grid")
    (vv1, _, mask1), (vv2, _, mask2) = t1.read(), t2.read()
    valid = (mask1 > 0) & (mask2 > 0) & np.isfinite(vv1) & np.isfinite(vv2) & (vv1 > 0) & (vv2 > 0)
    if not np.any(valid):
        raise ValueError("T1 and T2 have no co-valid positive VV pixels")
    db1, db2 = (10.0 * np.log10(np.where(valid, vv, 1.0)) for vv in (vv1, vv2))
    water1, water2, threshold = water_masks(db1, db2, valid)
    statistics: dict[str, Any] = {
        "type": "water_change_statistics",
        "method": "pooled_otsu_vv_db",
        "heuristic": True,
        "bands_used": ["VV"],
        "threshold_db": threshold,
        "water_rule": "open water when VV backscatter (dB) < threshold_db",
        "water_db_ceiling": WATER_DB_CEILING,
        "min_mapping_pixels": MIN_MAPPING_PIXELS,
        "area_method": f"per-row pixel area in {EQUAL_AREA_CRS} (equal area)",
    }
    if threshold > WATER_DB_CEILING:
        statistics.update(
            status="abstained", reason_code="NO_OPEN_WATER_MODE",
            water_t1_ha=None, water_t2_ha=None, new_water_ha=None, receded_water_ha=None,
        )
        answer = (
            "Abstained: VV backscatter has no distinct open-water mode (Otsu split at "
            f"{threshold:.1f} dB, above the {WATER_DB_CEILING:.0f} dB open-water ceiling), "
            "so no water change is reported."
        )
        return WaterChange(answer, [statistics], valid, None)

    new_water = _sieved(water2 & ~water1)
    receded = _sieved(water1 & ~water2)
    row_area = _row_pixel_area_m2(t1.transform, t1.crs, t1.height)
    new_ha, receded_ha = _hectares(new_water, row_area), _hectares(receded, row_area)
    statistics.update(
        status="measured", reason_code=None,
        water_t1_ha=_hectares(water1, row_area), water_t2_ha=_hectares(water2, row_area),
        new_water_ha=new_ha, receded_water_ha=receded_ha,
    )
    polygons = _polygons({"new_water": new_water, "receded_water": receded}, t1)
    answer = (
        f"Sentinel-1 water-change baseline: {new_ha:.1f} ha became open water between T1 and T2 "
        f"and {receded_ha:.1f} ha stopped being open water. Water is a single uncalibrated "
        f"threshold of {threshold:.1f} dB on VV backscatter; it does not identify villages, "
        "flood depth or causes, and smooth surfaces or radar shadow can read as water."
    )
    return WaterChange(answer, [statistics, polygons], valid, new_water)
