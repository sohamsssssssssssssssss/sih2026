"""Per-village flooded area from a new-water mask and village boundary polygons.

Boundaries are a committed subset of DataMeet's Indian Village Boundaries
(Census 2001, ODbL-1.0), clipped to the catalogued flood events; see
data/boundaries/README.md. Each village is burned onto its own pixel window by
pixel centre, so overlapping or nested polygons each keep their full area. A
village's area here is the area of its pixels inside the scene, including
pixels with no valid data: ``flooded_fraction`` divides by that area and
``observed_fraction`` says how much of it was actually seen. A village that
crosses the scene edge is flagged rather than estimated.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

VILLAGE_BOUNDARIES_PATH = Path(__file__).resolve().parents[2] / "data" / "boundaries" / "villages.v1.geojson"
BOUNDARY_SOURCE = "DataMeet Indian Village Boundaries, ODbL-1.0"
BOUNDARY_VINTAGE = "Census 2001"
BOUNDARY_CRS = "EPSG:4326"
MAX_LISTED_VILLAGES = 200
SQUARE_METRES_PER_HECTARE = 10_000
# Reprojected edges that coincide with a pixel edge land within ~1e-9 pixels of it.
PIXEL_EPSILON = 1e-6
PROPERTY_NAMES = {
    "NAME": "name", "TYPE": "type", "SUB_DIST": "sub_district",
    "DISTRICT": "district", "STATE": "state", "CEN_2001": "census_2001_code",
}


@lru_cache(maxsize=4)
def _parsed(path: str, modified_ns: int, size: int) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8"))["features"])


def _features(path: Path) -> tuple[dict[str, Any], ...]:
    """Parsed boundaries, re-read whenever the file changes on disk."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return ()
    return _parsed(str(path), stat.st_mtime_ns, stat.st_size)


def geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    points = np.concatenate([np.asarray(ring, dtype=float)[:, :2] for polygon in polygons for ring in polygon])
    return (*points.min(axis=0), *points.max(axis=0))


def bounds_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _pixel_window(geometry: dict[str, Any], dataset: Any) -> tuple[Window, bool] | None:
    """The polygon's pixel window clipped to the scene, and whether clipping cut it."""
    x0, y0, x1, y1 = geometry_bounds(geometry)
    cols, rows = ~dataset.transform @ (np.array([x0, x1]), np.array([y0, y1]))
    c0, c1 = int(np.floor(cols.min() + PIXEL_EPSILON)), int(np.ceil(cols.max() - PIXEL_EPSILON))
    r0, r1 = int(np.floor(rows.min() + PIXEL_EPSILON)), int(np.ceil(rows.max() - PIXEL_EPSILON))
    outside = c0 < 0 or r0 < 0 or c1 > dataset.width or r1 > dataset.height
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, dataset.width), min(r1, dataset.height)
    if c0 >= c1 or r0 >= r1:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0), outside


def village_flooding(
    new_water: np.ndarray, valid: np.ndarray, row_area: np.ndarray, dataset: Any
) -> dict[str, Any]:
    """Flooded hectares per village, largest first; villages with no new water are counted only."""
    item: dict[str, Any] = {
        "type": "village_flooding",
        "boundary_source": BOUNDARY_SOURCE,
        "boundary_vintage": BOUNDARY_VINTAGE,
    }
    scene = transform_bounds(dataset.crs, BOUNDARY_CRS, *dataset.bounds, densify_pts=21)
    candidates = [
        feature for feature in _features(VILLAGE_BOUNDARIES_PATH)
        if bounds_overlap(geometry_bounds(feature["geometry"]), scene)
    ]
    if not candidates:
        return {**item, "status": "no_boundaries", "villages_in_scene": None, "flooded_villages": []}

    in_scene, listed = 0, []
    for feature in candidates:
        geometry = transform_geom(BOUNDARY_CRS, dataset.crs, feature["geometry"])
        located = _pixel_window(geometry, dataset)
        if located is None:
            continue
        window, outside = located
        rows, cols = window.toslices()
        inside = rasterize(
            [(geometry, 1)], out_shape=(window.height, window.width),
            transform=window_transform(window, dataset.transform), fill=0, dtype="uint8",
        ).astype(bool)
        pixel_area = inside * row_area[rows][:, None]
        area = float(pixel_area.sum())
        if area == 0:
            continue  # no pixel centre falls inside it
        in_scene += 1
        flooded = float((pixel_area * new_water[rows, cols]).sum())
        if flooded == 0:
            continue
        observed = float((pixel_area * valid[rows, cols]).sum())
        properties = feature.get("properties") or {}
        listed.append(
            {
                **{field: properties.get(key) for key, field in PROPERTY_NAMES.items()},
                "flooded_ha": flooded / SQUARE_METRES_PER_HECTARE,
                "area_in_scene_ha": area / SQUARE_METRES_PER_HECTARE,
                "flooded_fraction": flooded / area,
                "observed_fraction": observed / area,
                "partially_outside_scene": outside,
            }
        )
    listed.sort(key=lambda village: village["flooded_ha"], reverse=True)
    return {
        **item,
        "status": "measured",
        "villages_in_scene": in_scene,
        "flooded_village_count": len(listed),
        "truncated": len(listed) > MAX_LISTED_VILLAGES,
        "flooded_villages": listed[:MAX_LISTED_VILLAGES],
    }


def answer_sentence(overlay: dict[str, Any], named: int = 5) -> str:
    if overlay["status"] == "no_boundaries":
        return "No village boundaries cover this scene, so villages are not named."
    flooded = overlay["flooded_villages"]
    if not flooded:
        return f"No village ({BOUNDARY_VINTAGE} boundaries) gained open water."
    listed = ", ".join(f"{village['name']} {village['flooded_ha']:.1f} ha" for village in flooded[:named])
    more = overlay["flooded_village_count"] - min(named, len(flooded))
    return (
        f"Villages with new open water ({BOUNDARY_VINTAGE} boundaries): {listed}"
        + (f", and {more} more." if more else ".")
    )
