"""Clip DataMeet village boundaries to the catalogued flood-event AOIs.

Source: https://github.com/datameet/indian_village_boundaries (ODbL-1.0),
community-digitised Census 2001 village polygons. Download the state files
(for example br/br.geojson and kl/kl.geojson), then run:

    python -m data.village_boundaries br.geojson kl.geojson

This writes data/boundaries/villages.v1.geojson with every village whose
bounding box overlaps the grid that `backend.sentinel1` requests for an event
in data/manifests/flood_events.v1.json. That grid is the AOI projected to UTM
and snapped outward, so it is slightly wider than the AOI itself. Properties
and coordinates are copied unchanged.
"""

import json
import sys
from pathlib import Path

from rasterio.warp import transform_bounds

from backend.sentinel1 import load_events, utm_grid
from models.change.villages import VILLAGE_BOUNDARIES_PATH, bounds_overlap, geometry_bounds


def scene_bounds(bbox: list[float]) -> tuple[float, float, float, float]:
    """Lon/lat bounds of the grid a fetch for this AOI actually covers."""
    grid = utm_grid(tuple(bbox))
    return transform_bounds(f"EPSG:{grid.epsg}", "EPSG:4326", *grid.bounds, densify_pts=21)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    aois = [scene_bounds(event["aoi"]["bbox"]) for event in load_events()]
    kept = [
        feature
        for path in paths
        for feature in json.loads(Path(path).read_text(encoding="utf-8"))["features"]
        if any(bounds_overlap(geometry_bounds(feature["geometry"]), aoi) for aoi in aois)
    ]
    VILLAGE_BOUNDARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n".join(json.dumps(feature, separators=(",", ":")) for feature in kept)
    VILLAGE_BOUNDARIES_PATH.write_text(
        '{"type":"FeatureCollection","features":[\n' + lines + "\n]}\n", encoding="utf-8"
    )
    print(f"{len(kept)} villages written to {VILLAGE_BOUNDARIES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
