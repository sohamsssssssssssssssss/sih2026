"""How cloudy Sentinel-2 optical imagery was over a flood AOI, so an answer can justify radar.

For the Sentinel-2 L2A acquisition closest to the post-event radar pass (within
OPTICAL_WINDOW), one Process API request reads the Scene Classification Layer
(SCL) on a coarse copy of the fetch's UTM grid. Only pixels with data count.

The collection id and SCL classes follow
https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html

A failed lookup (HTTP error, malformed reply, no Sentinel-2 data over the AOI)
is reported as verdict "unavailable" with its reason and null numbers; it never
raises, so it can never fail the SAR fetch.
"""

import math
from datetime import datetime, timedelta
from typing import Any

import httpx
import numpy as np
from rasterio.errors import RasterioError
from rasterio.io import MemoryFile

from backend.sentinel1 import (
    CATALOG_PAGE_SIZE, CATALOG_URL, PASS_WINDOW, PROCESS_URL, CDSEError, Grid, _iso, _json, _post,
)

COLLECTION = "sentinel-2-l2a"
OPTICAL_WINDOW = timedelta(days=3)  # either side of the post-event radar pass
OPTICAL_PIXEL_SIZE_M = 60  # coarse is enough for a fraction; SCL is native 20 m
SCL_NO_DATA = 0
# Cloud shadow (3), cloud medium (8) and high (9) probability, thin cirrus (10):
# the ground is hidden from an optical water map. Low-probability cloud (7) is
# also "unclassified", so it does not count.
SCL_OBSCURED = (3, 8, 9, 10)
# Above this share of the observed AOI obscured, an optical flood map would
# have gaps large enough to miss flooded villages, so radar is the right
# sensor. The fraction is always reported beside the verdict.
CLOUDY_THRESHOLD = 0.2

EVALSCRIPT = f"""//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["SCL", "dataMask"]}}],
    output: {{bands: 1, sampleType: "UINT8"}}
  }};
}}
function evaluatePixel(s) {{
  return [s.dataMask ? s.SCL : {SCL_NO_DATA}];
}}
"""

Scene = tuple[str, datetime]  # Sentinel-2 product id, acquisition time


def _verdict(
    verdict: str,
    reason: str,
    scene: Scene | None = None,
    cloud_fraction: float | None = None,
    valid_fraction: float | None = None,
) -> dict[str, Any]:
    return {
        "sentinel2_product_id": scene[0] if scene else None,
        "acquired_at": _iso(scene[1]) if scene else None,
        "aoi_cloud_fraction": cloud_fraction,
        "aoi_valid_fraction": valid_fraction,
        "verdict": verdict,
        "cloudy_threshold": CLOUDY_THRESHOLD,
        "reason": reason,
    }


def _acquisitions(client: httpx.Client, token: str, bbox: tuple[float, ...], around: datetime) -> list[Scene]:
    body = {
        "collections": [COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{_iso(around - OPTICAL_WINDOW)}/{_iso(around + OPTICAL_WINDOW)}",
        "limit": CATALOG_PAGE_SIZE,
    }
    page = _json(_post(client, CATALOG_URL, token=token, json=body))
    if (page.get("context") or {}).get("next"):
        raise CDSEError(f"More than {CATALOG_PAGE_SIZE} Sentinel-2 acquisitions; the closest one is uncertain")
    return [
        (str(item["id"]), datetime.fromisoformat(str(item["properties"]["datetime"]).replace("Z", "+00:00")))
        for item in page["features"]
    ]


def _aoi_fractions(client: httpx.Client, token: str, acquired_at: datetime, grid: Grid) -> tuple[float, float]:
    """(obscured share of pixels with data, share of AOI pixels with data) for one pass."""
    left, bottom, right, top = grid.bounds
    width, height = (math.ceil(span / OPTICAL_PIXEL_SIZE_M) for span in (right - left, top - bottom))
    body = {
        "input": {
            "bounds": {
                "bbox": list(grid.bounds),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{grid.epsg}"},
            },
            "data": [
                {
                    "type": COLLECTION,
                    "dataFilter": {
                        "timeRange": {"from": _iso(acquired_at - PASS_WINDOW), "to": _iso(acquired_at + PASS_WINDOW)}
                    },
                    "processing": {"downsampling": "NEAREST"},  # SCL is categorical: never blend classes
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    content = _post(client, PROCESS_URL, token=token, json=body).content
    with MemoryFile(content) as source, source.open() as dataset:
        if (dataset.count, dataset.dtypes[0], dataset.width, dataset.height) != (1, "uint8", width, height):
            raise CDSEError("CDSE returned a raster that is not one uint8 SCL band on the requested grid")
        scl = dataset.read(1)
    valid = int(np.count_nonzero(scl != SCL_NO_DATA))
    if not valid:
        raise CDSEError("Sentinel-2 has no data over the AOI on that pass")
    return int(np.isin(scl, SCL_OBSCURED).sum()) / valid, valid / scl.size


def check_optical_clouds(
    client: httpx.Client, token: str, grid: Grid, bbox: tuple[float, ...], post_event: datetime
) -> dict[str, Any]:
    """Cloud cover over the AOI of the Sentinel-2 L2A acquisition closest to ``post_event``."""
    nearest: Scene | None = None
    try:
        scenes = _acquisitions(client, token, bbox, post_event)
        if not scenes:
            return _verdict(
                "no_acquisition",
                f"No Sentinel-2 L2A acquisition within {OPTICAL_WINDOW.days} days of {_iso(post_event)}",
            )
        # A catalog datetime without a timezone raises TypeError here.
        nearest = min(scenes, key=lambda scene: (abs(scene[1] - post_event), scene[1]))
        cloud_fraction, valid_fraction = _aoi_fractions(client, token, nearest[1], grid)
    except (CDSEError, RasterioError) as exc:
        return _verdict("unavailable", str(exc), nearest)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return _verdict("unavailable", f"CDSE returned a malformed Sentinel-2 reply ({type(exc).__name__})", nearest)
    return _verdict(
        "cloudy" if cloud_fraction > CLOUDY_THRESHOLD else "clear",
        f"{cloud_fraction:.0%} of the {valid_fraction:.0%} of the AOI that Sentinel-2 observed "
        "was cloud, cirrus or cloud shadow",
        nearest, cloud_fraction, valid_fraction,
    )
