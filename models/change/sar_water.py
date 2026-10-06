"""Sentinel-1 open-water change: the SAR path of the bi-temporal change baseline.

Water is one VV threshold in dB, shared by both dates so "water" means the
same thing at T1 and T2. It is a simplified split-based threshold after Chini
et al. (2017, IEEE TGRS 55(12)): the scene is cut into fixed tiles, a tile
counts only when its own Otsu split is clearly bimodal (Ashman's D > 2, each
class at least 10% of the tile) with a water-like low class, and Otsu over the
pooled pixels of those tiles sets the threshold. A scene-wide split fails when
water is a few percent of the scene: it lands between land classes. No such
tile means no open-water mode, and the provider abstains.

This is the baseline a trained segmenter has to beat on held-out IoU. To swap
one in, replace ``split_threshold`` and ``water_masks`` with one function that
returns ``(water_t1, water_t2)``; the threshold, tile and sensitivity fields
are specific to this baseline.

Inputs are co-gridded VV/VH/dataMask rasters in linear backscatter, the same
band contract as the optical-SAR provider.
"""

from typing import Any, NamedTuple

import numpy as np
from rasterio.features import shapes, sieve
from rasterio.warp import transform, transform_geom

from models.change.villages import answer_sentence, village_flooding

SAR_BANDS = ("VV", "VH", "dataMask")
OTSU_BINS = 256
# A handful of extreme pixels (radar shadow, calibration artefacts) must not stretch the bins.
OTSU_CLIP_PERCENTILES = (0.1, 99.9)
# ponytail: a fixed physical prior, not a calibrated value. A tile counts as
# water/land only if its low class's mean C-band VV is below this and its high
# class's mean is above it, so two dark land classes (sand, smooth bare soil and
# vegetation) cannot pass as water. Phase 4 calibration replaces it.
WATER_DB_CEILING = -15.0
# ponytail: fixed tiles, not Chini's hierarchical quadrants. On the real Kosi
# 2024 pair the threshold moved under 0.5 dB between 32, 64 and 128 px tiles.
TILE_PIXELS = 64
MIN_ASHMAN_D = 2.0  # Chini et al. 2017: two Gaussians are clearly separated above 2
MIN_CLASS_FRACTION = 0.1  # Chini et al. 2017: the smaller class covers at least 10%
MIN_VALID_TILE_FRACTION = 0.5
SENSITIVITY_DB = 1.0
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


def _bimodal_water_tile(values: np.ndarray) -> bool:
    split = otsu_threshold(values)
    low, high = values[values < split], values[values >= split]
    if min(low.size, high.size) < max(MIN_CLASS_FRACTION * values.size, 2):
        return False
    spread = np.sqrt(low.var() + high.var())
    ashman_d = np.sqrt(2) * (high.mean() - low.mean()) / spread if spread else np.inf
    return bool(ashman_d > MIN_ASHMAN_D and low.mean() < WATER_DB_CEILING < high.mean())


def _tile_starts(length: int, tile: int) -> list[int]:
    """Full tiles from the origin, plus one flush with the far edge so no strip is skipped."""
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def split_threshold(
    db_t1: np.ndarray, db_t2: np.ndarray, valid: np.ndarray
) -> tuple[float | None, int]:
    """Otsu over the bimodal water tiles of both dates, and how many tiles qualified."""
    rows, cols = valid.shape
    tile_rows, tile_cols = min(TILE_PIXELS, rows), min(TILE_PIXELS, cols)
    selected = []
    for db in (db_t1, db_t2):
        for row in _tile_starts(rows, tile_rows):
            for col in _tile_starts(cols, tile_cols):
                window = (slice(row, row + tile_rows), slice(col, col + tile_cols))
                if valid[window].mean() < MIN_VALID_TILE_FRACTION:
                    continue
                values = db[window][valid[window]]
                if _bimodal_water_tile(values):
                    selected.append(values)
    if not selected:
        return None, 0
    return otsu_threshold(np.concatenate(selected)), len(selected)


def water_masks(
    db_t1: np.ndarray, db_t2: np.ndarray, valid: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    return (db_t1 < threshold) & valid, (db_t2 < threshold) & valid


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
    threshold, tile_count = split_threshold(db1, db2, valid)
    statistics: dict[str, Any] = {
        "type": "water_change_statistics",
        "method": "split_based_otsu_vv_db",
        "heuristic": True,
        "bands_used": ["VV"],
        "threshold_db": threshold,
        "bimodal_tiles": tile_count,
        "tile_rule": (
            f"{TILE_PIXELS} px tiles whose Otsu split has Ashman's D > {MIN_ASHMAN_D:g}, "
            f"each class >= {MIN_CLASS_FRACTION:.0%}, a low-class mean below and a high-class mean "
            f"above {WATER_DB_CEILING:g} dB"
        ),
        "water_rule": "open water when VV backscatter (dB) < threshold_db",
        "water_db_ceiling": WATER_DB_CEILING,
        "min_mapping_pixels": MIN_MAPPING_PIXELS,
        "area_method": f"per-row pixel area in {EQUAL_AREA_CRS} (equal area)",
    }
    if threshold is None:
        statistics.update(
            status="abstained", reason_code="NO_OPEN_WATER_MODE",
            water_t1_ha=None, water_t2_ha=None, new_water_ha=None, receded_water_ha=None,
            new_water_ha_sensitivity=None,
        )
        answer = (
            "Abstained: no part of the scene shows a distinct open-water mode in VV backscatter "
            f"(no tile was clearly bimodal with a water class below {WATER_DB_CEILING:.0f} dB), "
            "so no water change is reported."
        )
        return WaterChange(answer, [statistics], valid, None)

    water1, water2 = water_masks(db1, db2, valid, threshold)
    new_water = _sieved(water2 & ~water1)
    receded = _sieved(water1 & ~water2)
    row_area = _row_pixel_area_m2(t1.transform, t1.crs, t1.height)
    new_ha, receded_ha = _hectares(new_water, row_area), _hectares(receded, row_area)
    low_ha, high_ha = (
        _hectares(_sieved(shifted[1] & ~shifted[0]), row_area)
        for shifted in (
            water_masks(db1, db2, valid, threshold - SENSITIVITY_DB),
            water_masks(db1, db2, valid, threshold + SENSITIVITY_DB),
        )
    )
    statistics.update(
        status="measured", reason_code=None,
        water_t1_ha=_hectares(water1, row_area), water_t2_ha=_hectares(water2, row_area),
        new_water_ha=new_ha, receded_water_ha=receded_ha,
        new_water_ha_sensitivity={"threshold_minus_1db": low_ha, "threshold_plus_1db": high_ha},
    )
    polygons = _polygons({"new_water": new_water, "receded_water": receded}, t1)
    overlay = village_flooding(new_water, valid, row_area, t1)
    answer = (
        f"Sentinel-1 water-change baseline: {new_ha:.1f} ha became open water between T1 and T2 "
        f"({min(low_ha, high_ha):.1f}-{max(low_ha, high_ha):.1f} ha if the threshold moves by "
        f"{SENSITIVITY_DB:g} dB) and "
        f"{receded_ha:.1f} ha stopped being open water. {answer_sentence(overlay)} Water is an "
        f"uncalibrated split-based threshold of {threshold:.1f} dB on VV backscatter; it does not "
        "measure flood depth or causes, and smooth surfaces or radar shadow can read as water."
    )
    return WaterChange(answer, [statistics, polygons, overlay], valid, new_water)
