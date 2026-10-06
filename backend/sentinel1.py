"""Fetch co-gridded Sentinel-1 pairs from the Copernicus Data Space for flood mapping.

Picks the pre- and post-event acquisitions on one orbit track (same orbit
direction and relative orbit, so backscatter differences are surface change,
not a different viewing geometry), renders both through the Sentinel Hub
Process API on one fixed UTM grid, and ingests them as runtime scenes that
change_vqa reads directly.

Calibration (gamma0 with radiometric terrain correction), orthorectification
on the Copernicus 30 m DEM and a Lee speckle filter run server-side. Pixels in
or near radar shadow are marked invalid, because shadow reads as water. Bands
follow the VV/VH/dataMask contract in linear power.

Usage:
    CDSE_CLIENT_ID=... CDSE_CLIENT_SECRET=... python -m backend.sentinel1 --event kosi-2024
    CDSE_CLIENT_ID=... CDSE_CLIENT_SECRET=... python -m backend.sentinel1 \\
        --bbox 85.0 25.55 85.2 25.7 --before 2024-07-20 2024-08-02 \\
        --after 2024-08-15 2024-08-25 --pair-group patna-2024

Catalogued events live in data/manifests/flood_events.v1.json.

Create the OAuth client at https://shapps.dataspace.copernicus.eu/dashboard/
under User settings > OAuth clients.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from rasterio.crs import CRS
from rasterio.errors import RasterioError
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from backend.services import InvalidImageUpload, SceneStorageError, ingest_scene
from models.change.sar_water import SAR_BANDS

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
COLLECTION = "sentinel-1-grd"
# Unverified against a live response until the first authenticated run; the
# Process dataFilter below enforces the same mode and polarization regardless.
CATALOG_FILTER = "sar:instrument_mode = 'IW' AND s1:polarization = 'DV'"
CATALOG_PAGE_SIZE = 50
MAX_CATALOG_PAGES = 20
PIXEL_SIZE_M = 10  # Sentinel-1 IW GRD high resolution
MAX_SIDE_PIXELS = 2500  # Process API output limit per side
# One pass crosses an AOI in seconds and the next pass over it is hours away,
# so this window holds every slice of the chosen pass and nothing else.
PASS_WINDOW = timedelta(minutes=10)
SPECKLE_WINDOW_PIXELS = 5
REQUEST_TIMEOUT_SECONDS = 120.0
ERROR_EXCERPT_CHARS = 200
PROVENANCE = "cdse_process_api"
FLOOD_EVENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "manifests" / "flood_events.v1.json"

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask", "shadowMask"]}],
    output: {bands: 3, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask && !s.shadowMask ? 1 : 0];
}
"""


class CDSEError(RuntimeError):
    """CDSE could not supply a usable Sentinel-1 pair."""


@dataclass(frozen=True)
class Acquisition:
    product_id: str
    acquired_at: datetime
    orbit_state: str
    relative_orbit: int | None


@dataclass(frozen=True)
class Grid:
    epsg: int
    bounds: tuple[float, float, float, float]
    width: int
    height: int


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utm_grid(bbox: tuple[float, float, float, float]) -> Grid:
    """The bbox's local UTM zone, snapped outward to whole 10 m pixels."""
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be WEST SOUTH EAST NORTH in degrees, with west < east and south < north")
    zone = min(int(((west + east) / 2 + 180) // 6) + 1, 60)
    epsg = (32600 if (south + north) / 2 >= 0 else 32700) + zone
    left, bottom, right, top = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox, densify_pts=21)
    left, bottom = (math.floor(value / PIXEL_SIZE_M) * PIXEL_SIZE_M for value in (left, bottom))
    right, top = (math.ceil(value / PIXEL_SIZE_M) * PIXEL_SIZE_M for value in (right, top))
    width, height = (right - left) // PIXEL_SIZE_M, (top - bottom) // PIXEL_SIZE_M
    if max(width, height) > MAX_SIDE_PIXELS:
        raise ValueError(
            f"AOI is {width}x{height} pixels at {PIXEL_SIZE_M} m; the Process API allows at most "
            f"{MAX_SIDE_PIXELS} per side"
        )
    return Grid(epsg, (float(left), float(bottom), float(right), float(top)), int(width), int(height))


def _post(client: httpx.Client, url: str, *, token: str | None = None, **kwargs: Any) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = client.post(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except httpx.HTTPError as exc:
        raise CDSEError(f"CDSE request to {url} failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise CDSEError(
            f"CDSE returned HTTP {response.status_code} for {url}: {response.text[:ERROR_EXCERPT_CHARS]}"
        )
    return response


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise CDSEError(f"CDSE returned non-JSON from {response.request.url}") from exc
    if not isinstance(value, dict):
        raise CDSEError(f"CDSE returned unexpected JSON from {response.request.url}")
    return value


def access_token(client: httpx.Client, client_id: str, client_secret: str) -> str:
    response = _post(
        client, TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
    )
    token = _json(response).get("access_token")
    if not isinstance(token, str) or not token:
        raise CDSEError("CDSE token response had no access_token")
    return token


def _acquisition(feature: Any) -> Acquisition:
    try:
        properties = feature["properties"]
        acquired_at = datetime.fromisoformat(str(properties["datetime"]).replace("Z", "+00:00"))
        orbit_state = str(properties["sat:orbit_state"]).lower()
        product_id = str(feature["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CDSEError("CDSE catalog item lacks an id, datetime or orbit state") from exc
    if acquired_at.tzinfo is None:
        raise CDSEError(f"CDSE catalog datetime for {product_id} has no timezone")
    relative_orbit = properties.get("sat:relative_orbit")
    return Acquisition(
        product_id, acquired_at, orbit_state,
        relative_orbit if isinstance(relative_orbit, int) and not isinstance(relative_orbit, bool) else None,
    )


def search(
    client: httpx.Client, token: str, bbox: tuple[float, ...], window: tuple[datetime, datetime]
) -> list[Acquisition]:
    body: dict[str, Any] = {
        "collections": [COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{_iso(window[0])}/{_iso(window[1])}",
        "limit": CATALOG_PAGE_SIZE,
        "filter": CATALOG_FILTER,
        "filter-lang": "cql2-text",
    }
    found: list[Acquisition] = []
    for _ in range(MAX_CATALOG_PAGES):
        page = _json(_post(client, CATALOG_URL, token=token, json=body))
        features, context = page.get("features") or [], page.get("context") or {}
        if not isinstance(features, list) or not isinstance(context, dict):
            raise CDSEError("CDSE catalog returned a malformed search page")
        found.extend(_acquisition(feature) for feature in features)
        next_page = context.get("next")
        if not next_page:
            return found
        body = {**body, "next": next_page}
    raise CDSEError(f"CDSE catalog returned more than {MAX_CATALOG_PAGES} pages; narrow the window")


def pick_pair(before: list[Acquisition], after: list[Acquisition]) -> tuple[Acquisition, Acquisition]:
    """The shortest-interval pair on one known orbit track.

    An unknown relative orbit never matches: pairing across tracks would turn
    a viewing-geometry difference into false water change.
    """
    pairs = [
        (first, second)
        for first in before
        for second in after
        if first.relative_orbit is not None
        and (first.orbit_state, first.relative_orbit) == (second.orbit_state, second.relative_orbit)
        and first.acquired_at < second.acquired_at
    ]
    if not pairs:
        raise CDSEError(
            "No pre- and post-event Sentinel-1 acquisitions share the same orbit track "
            "(direction and relative orbit) in these windows; widen them."
        )
    return min(pairs, key=lambda pair: (pair[1].acquired_at - pair[0].acquired_at, pair[1].acquired_at))


def _with_band_descriptions(content: bytes, grid: Grid) -> tuple[bytes, float]:
    """Check the Process output landed on the requested grid, name its bands, report coverage."""
    expected = from_bounds(*grid.bounds, grid.width, grid.height)
    try:
        with MemoryFile(content) as source, source.open() as dataset:
            if not (
                dataset.count == len(SAR_BANDS)
                and dataset.dtypes == ("float32",) * len(SAR_BANDS)
                and (dataset.width, dataset.height) == (grid.width, grid.height)
                and dataset.crs == CRS.from_epsg(grid.epsg)
                and dataset.transform.almost_equals(expected)
            ):
                raise CDSEError("CDSE returned a raster that is not float32 VV/VH/dataMask on the requested grid")
            profile, data = dataset.profile, dataset.read()
    except RasterioError as exc:
        raise CDSEError("CDSE did not return a readable GeoTIFF") from exc
    with MemoryFile() as target:
        with target.open(**profile) as output:
            output.write(data)
            output.descriptions = SAR_BANDS
        return target.read(), float(np.mean(data[2] > 0))


def render(client: httpx.Client, token: str, acquisition: Acquisition, grid: Grid) -> tuple[bytes, float]:
    """GeoTIFF bytes on ``grid`` and the fraction of pixels that are valid (covered, not shadow)."""
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
                        "timeRange": {
                            "from": _iso(acquisition.acquired_at - PASS_WINDOW),
                            "to": _iso(acquisition.acquired_at + PASS_WINDOW),
                        },
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "resolution": "HIGH",
                        "orbitDirection": acquisition.orbit_state.upper(),
                    },
                    "processing": {
                        "backCoeff": "GAMMA0_TERRAIN",
                        "orthorectify": True,
                        "demInstance": "COPERNICUS_30",
                        "speckleFilter": {
                            "type": "LEE",
                            "windowSizeX": SPECKLE_WINDOW_PIXELS,
                            "windowSizeY": SPECKLE_WINDOW_PIXELS,
                        },
                    },
                }
            ],
        },
        "output": {
            "width": grid.width,
            "height": grid.height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    return _with_band_descriptions(_post(client, PROCESS_URL, token=token, json=body).content, grid)


def _ingest(
    rendered: tuple[bytes, float], acquisition: Acquisition, pair_group: str,
    optical_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raster, valid_fraction = rendered
    stored = ingest_scene(
        raster,
        f"{acquisition.product_id}.tif",
        {
            "modality": "sar",
            "sensor": "Sentinel-1",
            "acquisition_timestamp": _iso(acquisition.acquired_at),
            "polarization": "VV,VH",
            "pair_group": pair_group,
        },
        provenance=PROVENANCE,
        acquisition_id=acquisition.product_id,
        optical_check=optical_check,
    )
    return {
        "scene_id": stored["scene_id"],
        "acquisition_id": acquisition.product_id,
        "acquisition_time": _iso(acquisition.acquired_at),
        "orbit_state": acquisition.orbit_state,
        "relative_orbit": acquisition.relative_orbit,
        "valid_fraction": valid_fraction,
    }


def fetch_pair(
    client: httpx.Client,
    client_id: str,
    client_secret: str,
    bbox: tuple[float, float, float, float],
    before: tuple[datetime, datetime],
    after: tuple[datetime, datetime],
    pair_group: str | None = None,
) -> dict[str, Any]:
    """Search, pair, render and ingest.

    Nothing is stored unless both dates render. If the second scene then fails
    to store, the error names the first, already-stored scene.
    """
    if not before[0] < before[1] <= after[0] < after[1]:
        raise ValueError("Windows must be ordered: before start < before end <= after start < after end")
    grid = utm_grid(bbox)
    token = access_token(client, client_id, client_secret)
    first, second = pick_pair(search(client, token, bbox, before), search(client, token, bbox, after))
    rasters = [render(client, token, item, grid) for item in (first, second)]
    # Imported here: optical_clouds builds on this module's CDSE helpers.
    from backend.optical_clouds import check_optical_clouds

    optical_check = check_optical_clouds(client, token, grid, bbox, second.acquired_at)
    group = pair_group or f"cdse:{first.product_id}+{second.product_id}"
    stored_before = _ingest(rasters[0], first, group)
    try:
        stored_after = _ingest(rasters[1], second, group, optical_check)
    except (InvalidImageUpload, SceneStorageError) as exc:
        raise CDSEError(
            f"The post-event scene could not be stored; pre-event scene {stored_before['scene_id']} "
            "was stored and is unpaired"
        ) from exc
    return {
        "pair_group": group,
        "crs": f"EPSG:{grid.epsg}",
        "width": grid.width,
        "height": grid.height,
        "before": stored_before,
        "after": stored_after,
        "optical_check": optical_check,
    }


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _inclusive(start: str, end: str) -> tuple[datetime, datetime]:
    """A window of whole UTC days; END is included."""
    return _day(start), _day(end) + timedelta(days=1)


def load_events() -> list[dict[str, Any]]:
    return json.loads(FLOOD_EVENTS_PATH.read_text(encoding="utf-8"))["events"]


Window = tuple[datetime, datetime]


def event_request(event_id: str) -> tuple[tuple[float, float, float, float], Window, Window]:
    """The catalogued AOI and inclusive before/after windows for one flood event."""
    events = {event["event_id"]: event for event in load_events()}
    if event_id not in events:
        raise ValueError(f"Unknown flood event {event_id!r}; catalogued: {', '.join(sorted(events))}")
    event = events[event_id]
    windows = [_inclusive(event[name]["start"], event[name]["end"]) for name in ("before", "after")]
    return tuple(event["aoi"]["bbox"]), windows[0], windows[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a co-gridded Sentinel-1 pair from CDSE and ingest it as runtime scenes."
    )
    parser.add_argument("--event", help=f"a flood event_id from {FLOOD_EVENTS_PATH.name}")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--before", nargs=2, metavar=("START", "END"), help="inclusive YYYY-MM-DD dates")
    parser.add_argument("--after", nargs=2, metavar=("START", "END"), help="inclusive YYYY-MM-DD dates")
    parser.add_argument("--pair-group")
    args = parser.parse_args(argv)
    explicit = (args.bbox, args.before, args.after)
    if args.event and any(explicit):
        parser.error("--event replaces --bbox, --before and --after")
    if not args.event and not all(explicit):
        parser.error("give --event, or all of --bbox, --before and --after")
    try:
        if args.event:
            bbox, before, after = event_request(args.event)
        else:
            bbox, before, after = tuple(args.bbox), _inclusive(*args.before), _inclusive(*args.after)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    client_id, client_secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to an OAuth client created at "
            "https://shapps.dataspace.copernicus.eu/dashboard/ (User settings > OAuth clients).",
            file=sys.stderr,
        )
        return 2
    try:
        with httpx.Client() as client:
            result = fetch_pair(
                client, client_id, client_secret, bbox, before, after, args.pair_group or args.event
            )
    except (CDSEError, ValueError, InvalidImageUpload, SceneStorageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
