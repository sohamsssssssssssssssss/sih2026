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
import io
import json
import tarfile
import math
import os
import re
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
MAX_SIDE_PIXELS = 2500  # Process API output limit per side, so the size of one request tile
# 2x2 request tiles, 50 km at 10 m. Change inference on a synthetic SAR pair
# peaked at 1.5 GB RSS and 4.2 s at 5000 px a side (2.9 GB, 9.3 s at 7500):
# memory grows with pixel count, so this caps memory, not time. 25 Mpx is far
# under services.MAX_RASTER_PIXELS.
MAX_MOSAIC_SIDE_PIXELS = 5000
# One pass crosses an AOI in seconds and the next pass over it is hours away,
# so this window holds every slice of the chosen pass and nothing else.
PASS_WINDOW = timedelta(minutes=10)
SPECKLE_WINDOW_PIXELS = 5
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_TIFF_BYTES = 80 * 1024 * 1024
MAX_USERDATA_BYTES = 1024 * 1024
MAX_PROCESS_BYTES = MAX_TIFF_BYTES + MAX_USERDATA_BYTES + 64 * 1024
MAX_ERROR_BYTES = 4096
ERROR_EXCERPT_CHARS = 200
PROVENANCE = "cdse_process_api"
FLOOD_EVENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "manifests" / "flood_events.v1.json"
FLOOD_RUNS_PATH = Path(__file__).resolve().parents[1] / "data" / "manifests" / "cdse" / "flood-runs.v1.json"

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask", "shadowMask"]}],
    output: {id: "default", bands: 3, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  if (samples.length === 0) return [0, 0, 0];
  if (samples.length !== 1) throw new Error("Expected exactly one ORBIT sample");
  const s = samples[0];
  return [s.VV, s.VH, s.dataMask && !s.shadowMask ? 1 : 0];
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {orbits: scenes.orbits, inputMetadata: inputMetadata};
}
"""


class CDSEError(RuntimeError):
    """CDSE could not supply a usable Sentinel-1 pair."""

    def __init__(self, message: str, *, diagnostic_category: str | None = None):
        super().__init__(message)
        self.diagnostic_category = diagnostic_category


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
    if max(width, height) > MAX_MOSAIC_SIDE_PIXELS:
        raise ValueError(
            f"AOI is {width}x{height} pixels at {PIXEL_SIZE_M} m; Sentinel-1 fetches mosaic at most "
            f"{MAX_MOSAIC_SIDE_PIXELS} per side ({MAX_MOSAIC_SIDE_PIXELS * PIXEL_SIZE_M // 1000} km)"
        )
    return Grid(epsg, (float(left), float(bottom), float(right), float(top)), int(width), int(height))


def _tile(grid: Grid, row: int, col: int) -> Grid:
    """The sub-grid from pixel (row, col), at most MAX_SIDE_PIXELS a side, on ``grid``'s own pixels."""
    width, height = min(MAX_SIDE_PIXELS, grid.width - col), min(MAX_SIDE_PIXELS, grid.height - row)
    left, top = grid.bounds[0] + col * PIXEL_SIZE_M, grid.bounds[3] - row * PIXEL_SIZE_M
    return Grid(grid.epsg, (left, top - height * PIXEL_SIZE_M, left + width * PIXEL_SIZE_M, top), width, height)


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
    client: httpx.Client, token: str, bbox: tuple[float, ...], window: tuple[datetime, datetime],
    *, product_id: str | None = None,
) -> list[Acquisition]:
    body: dict[str, Any] = {
        "collections": [COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{_iso(window[0])}/{_iso(window[1])}",
        "limit": CATALOG_PAGE_SIZE,
        "filter": CATALOG_FILTER,
        "filter-lang": "cql2-text",
    }
    if product_id is not None:
        body = {"collections": [COLLECTION], "ids": [product_id], "limit": 2}
    found: list[Acquisition] = []
    for _ in range(MAX_CATALOG_PAGES):
        page = _json(_post(client, CATALOG_URL, token=token, json=body))
        features, context = page.get("features") or [], page.get("context") or {}
        if not isinstance(features, list) or not isinstance(context, dict):
            raise CDSEError("CDSE catalog returned a malformed search page")
        if product_id is not None:
            for feature in features:
                properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
                if (not isinstance(feature, dict) or feature.get("collection") != COLLECTION
                        or not isinstance(properties, dict)
                        or properties.get("sar:instrument_mode") != "IW"
                        or properties.get("s1:polarization") != "DV"
                        or properties.get("s1:resolution") != "HIGH"):
                    raise CDSEError("Pinned catalogue product collection or SAR mode mismatch")
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


def _on_grid(content: bytes, grid: Grid) -> tuple[dict[str, Any], np.ndarray]:
    """The Process output's profile and pixels, checked to have landed on the requested grid."""
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
                raise CDSEError("CDSE returned a raster that is not float32 VV/VH/dataMask on the requested grid", diagnostic_category="raster_contract_mismatch")
            return dict(dataset.profile), dataset.read()
    except RasterioError as exc:
        raise CDSEError("CDSE did not return a readable GeoTIFF", diagnostic_category="raster_decode_failed") from exc


def _process_request(client: httpx.Client, token: str, body: dict[str, Any]) -> bytes:
    """Bound Process bodies while streaming; shared _post callers remain unchanged."""
    try:
        with client.stream("POST", PROCESS_URL, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/x-tar",
        }, json=body, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            limit = MAX_PROCESS_BYTES if response.status_code == 200 else MAX_ERROR_BYTES
            declared = None
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise CDSEError("Process API invalid Content-Length") from exc
                if declared < 0 or declared > limit:
                    raise CDSEError("Process API response exceeds byte limit")
            if response.status_code == 200:
                media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if media_type not in {"application/x-tar", "application/tar"}:
                    raise CDSEError("Process API incorrect Content-Type; expected TAR")
            content = bytearray()
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                if len(content) + len(chunk) > limit:
                    raise CDSEError("Process API response exceeds byte limit")
                content.extend(chunk)
            if (declared is not None and response.headers.get("Content-Encoding", "identity") == "identity"
                    and len(content) != declared):
                raise CDSEError("Process API incomplete response")
            if response.status_code != 200:
                excerpt = content.decode("utf-8", errors="replace")[:ERROR_EXCERPT_CHARS]
                raise CDSEError(f"CDSE returned HTTP {response.status_code} for Process API: {excerpt}")
            return bytes(content)
    except httpx.HTTPError as exc:
        raise CDSEError(f"CDSE Process request failed: {type(exc).__name__}") from exc


def _parse_userdata(userdata_bytes: bytes, expected_product_id: str | None) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CDSEError("Process API userdata contains duplicate object keys", diagnostic_category="userdata_json_invalid")
            result[key] = value
        return result

    try:
        userdata = json.loads(userdata_bytes.decode("utf-8"), object_pairs_hook=unique_object)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CDSEError("Process API userdata is not valid JSON", diagnostic_category="userdata_json_invalid") from exc
    orbits = userdata.get("orbits") if isinstance(userdata, dict) else None
    if not isinstance(orbits, list) or len(orbits) != 1:
        raise CDSEError("Process API userdata must contain exactly one orbit metadata entry", diagnostic_category="source_provenance_invalid")
    product_ids = set()
    for orbit in orbits:
        tiles = orbit.get("tiles") if isinstance(orbit, dict) else None
        if not isinstance(tiles, list) or not tiles:
            raise CDSEError("Process API userdata missing source tiles", diagnostic_category="source_provenance_invalid")
        for tile in tiles:
            product_id = tile.get("sentinel1ProductId") if isinstance(tile, dict) else None
            if not isinstance(product_id, str) or not re.fullmatch(r"S1[A-Z]_[A-Z0-9_]+(?:\.SAFE)?", product_id):
                raise CDSEError("Process API userdata missing or malformed source product ID", diagnostic_category="source_provenance_invalid")
            product_ids.add(product_id)
    if expected_product_id is not None and product_ids != {expected_product_id}:
        raise CDSEError(f"Process API source product ID mismatch: expected {expected_product_id}, got {sorted(product_ids)}", diagnostic_category="source_provenance_invalid")
    return userdata


def render(client: httpx.Client, token: str, acquisition: Acquisition, grid: Grid, *,
           strict: bool = False) -> tuple[bytes, float, dict[str, Any]]:
    """GeoTIFF bytes on ``grid`` and the fraction of pixels that are valid (covered, not shadow).

    An AOI wider than one Process request is rendered tile by tile on the same
    pixel lattice and copied into place, with no resampling or overlap. Any bad
    tile fails the whole render. Unfilled pixels would stay dataMask 0 (invalid).

    For each tile, requests TIFF + userdata.json and validates the source product ID
    equals the pinned product ID in strict event mode; non-event sources are recorded.
    """
    evidence = {"tiles": [], "processing": _process_body(acquisition, grid)["input"]["data"][0]["processing"],
                "evalscript": EVALSCRIPT}
    mosaic = np.zeros((len(SAR_BANDS), grid.height, grid.width), dtype="float32")
    for row in range(0, grid.height, MAX_SIDE_PIXELS):
        for col in range(0, grid.width, MAX_SIDE_PIXELS):
            tile = _tile(grid, row, col)
            content = _process_request(client, token, _process_body(acquisition, tile))
            tar_diagnostic = {}
            tiff_bytes, userdata_bytes = _parse_tar_response(content, allow_missing_eof=True,
                                                            diagnostics=tar_diagnostic)
            userdata = _parse_userdata(userdata_bytes, acquisition.product_id if strict else None)
            evidence["tiles"].append({"bounds": list(tile.bounds), "userdata": userdata, **tar_diagnostic})
            profile, data = _on_grid(tiff_bytes, tile)
            mosaic[:, row:row + tile.height, col:col + tile.width] = data
    transform = from_bounds(*grid.bounds, grid.width, grid.height)
    profile = {**profile, "width": grid.width, "height": grid.height, "transform": transform}
    with MemoryFile() as target:
        with target.open(**profile) as output:
            output.write(mosaic)
            output.descriptions = SAR_BANDS
        return target.read(), float(np.mean(mosaic[2] > 0)), evidence


def _parse_tar_response(content: bytes, *, allow_missing_eof: bool = False,
                        diagnostics: dict[str, Any] | None = None) -> tuple[bytes, bytes]:
    """Missing EOF is opt-in only after a response stream completes without error."""
    if len(content) > MAX_PROCESS_BYTES:
        raise CDSEError("Process API response exceeds byte limit", diagnostic_category="tar_structure_invalid")
    files = {}
    complete_padding = True
    eof_mode = "standard"
    limits = {"default.tif": MAX_TIFF_BYTES, "userdata.json": MAX_USERDATA_BYTES}
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            for member in archive:
                if (member.name not in limits or member.name in files
                        or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.pax_headers):
                    raise CDSEError("Process API TAR contains unexpected or duplicate members", diagnostic_category="tar_member_invalid")
                if member.size <= 0 or member.size > limits[member.name]:
                    raise CDSEError("Process API TAR member exceeds byte limit or is empty", diagnostic_category="tar_member_invalid")
                source = archive.extractfile(member)
                if source is None:
                    raise CDSEError("Process API malformed TAR member", diagnostic_category="tar_member_invalid")
                data = source.read(member.size + 1)
                if len(data) != member.size:
                    raise CDSEError("Process API truncated TAR member", diagnostic_category="tar_member_invalid")
                padded_end = member.offset_data + ((member.size + 511) // 512) * 512
                complete_padding = complete_padding and (
                    padded_end <= len(content)
                    and not any(content[member.offset_data + member.size:padded_end])
                )
                files[member.name] = data
            tail = content[archive.offset:]
            # Extra zero padding after the two EOF blocks need not fill a block.
            if (allow_missing_eof and archive.offset == len(content)
                    and complete_padding and files.keys() == limits.keys()):
                eof_mode = "missing_accepted"
            elif len(tail) < 1024 or any(tail):
                raise CDSEError("Process API malformed TAR ending", diagnostic_category="tar_eof_invalid")
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise CDSEError("Process API malformed TAR", diagnostic_category="tar_structure_invalid") from exc
    if files.keys() != limits.keys():
        raise CDSEError("Process API TAR missing default.tif or userdata.json", diagnostic_category="tar_member_invalid")
    if diagnostics is not None:
        diagnostics["tar_eof_mode"] = eof_mode
    return files["default.tif"], files["userdata.json"]


def _process_body(acquisition: Acquisition, grid: Grid) -> dict[str, Any]:
    """One Process API request: calibrated, terrain-corrected, speckle-filtered VV/VH/dataMask on ``grid``.

    Requests ORBIT mosaicking within the selected time window, and a combined
    TIFF + userdata.json response for source provenance validation.
    """
    return {
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
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}},
                {"identifier": "userdata", "format": {"type": "application/json"}},
            ],
        },
        "evalscript": EVALSCRIPT,
    }


def _ingest(
    rendered: tuple[bytes, float, dict[str, Any]], acquisition: Acquisition, pair_group: str,
    optical_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raster, valid_fraction, process_evidence = rendered
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
        process_evidence=process_evidence,
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
    event_id: str | None = None,
) -> dict[str, Any]:
    """Search, pair, render and ingest.

    Nothing is stored unless both dates render and pass source provenance validation.
    If the second scene then fails to store, the error names the first, already-stored scene.

    Event mode queries the two exact pinned IDs and verifies their timestamps
    and orbit metadata before rendering. Non-event mode picks a shared-track pair.

    Each Process API response includes userdata.json which is parsed to validate
    that the source product ID equals the pinned product ID in strict event mode; non-event sources are recorded.
    """
    if not before[0] < before[1] <= after[0] < after[1]:
        raise ValueError("Windows must be ordered: before start < before end <= after start < after end")
    grid = utm_grid(bbox)
    pinned = load_pinned_run(event_id) if event_id is not None else None
    token = access_token(client, client_id, client_secret)
    if event_id is None:
        first, second = pick_pair(search(client, token, bbox, before), search(client, token, bbox, after))
    else:
        selected = []
        for side, window in (("before", before), ("after", after)):
            expected = pinned[side]
            candidates = search(client, token, bbox, window, product_id=expected["product_id"])
            if len(candidates) != 1 or candidates[0].product_id != expected["product_id"]:
                label = "Pre-event" if side == "before" else "Post-event"
                raise CDSEError(f"{label} product ID mismatch or missing/duplicate pinned product")
            acquisition = candidates[0]
            expected_time = datetime.fromisoformat(expected["acquired_at"].replace("Z", "+00:00"))
            if (acquisition.acquired_at != expected_time
                    or not window[0] <= acquisition.acquired_at < window[1]
                    or acquisition.orbit_state != pinned["orbit_direction"]
                    or acquisition.relative_orbit != pinned["relative_orbit"]
                    or acquisition.relative_orbit is None):
                raise CDSEError("Pinned acquisition timestamp or orbit metadata mismatch")
            selected.append(acquisition)
        first, second = selected
        if first.acquired_at >= second.acquired_at:
            raise CDSEError("Pinned acquisitions are not chronologically ordered")

    # Render and validate both acquisitions before ingesting either.
    # render() internally validates source product ID via userdata.json.
    rendered_before = render(client, token, first, grid, strict=event_id is not None)
    rendered_after = render(client, token, second, grid, strict=event_id is not None)

    # Imported here: optical_clouds builds on this module's CDSE helpers.
    from backend.optical_clouds import check_optical_clouds

    optical_check = check_optical_clouds(client, token, grid, bbox, second.acquired_at)
    group = pair_group or f"cdse:{first.product_id}+{second.product_id}"
    stored_before = _ingest(rendered_before, first, group)
    try:
        stored_after = _ingest(rendered_after, second, group, optical_check)
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
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a YYYY-MM-DD date") from exc


def _inclusive(start: str, end: str) -> tuple[datetime, datetime]:
    """A window of whole UTC days; END is included."""
    return _day(start), _day(end) + timedelta(days=1)


def load_events() -> list[dict[str, Any]]:
    return json.loads(FLOOD_EVENTS_PATH.read_text(encoding="utf-8"))["events"]


def load_pinned_run(event_id: str) -> dict[str, Any]:
    """Load one event's exact acquisition and orbit records."""
    events = json.loads(FLOOD_RUNS_PATH.read_text(encoding="utf-8"))["events"]
    for event in events:
        if event["event_id"] == event_id:
            return event
    raise CDSEError(f"Event {event_id!r} not found in flood-runs.v1.json")


def load_expected_products(event_id: str) -> tuple[str, str]:
    """Load the expected pre-event and post-event product IDs from flood-runs.v1.json."""
    event = load_pinned_run(event_id)
    return event["before"]["product_id"], event["after"]["product_id"]


Window = tuple[datetime, datetime]


def event_request(event_id: str) -> tuple[tuple[float, float, float, float], Window, Window]:
    """The catalogued AOI and inclusive before/after windows for one flood event."""
    events = {event["event_id"]: event for event in load_events()}
    if event_id not in events:
        raise ValueError(f"Unknown flood event {event_id!r}; catalogued: {', '.join(sorted(events))}")
    event = events[event_id]
    windows = [_inclusive(event[name]["start"], event[name]["end"]) for name in ("before", "after")]
    return tuple(event["aoi"]["bbox"]), windows[0], windows[1]


Days = tuple[str, str] | list[str]  # inclusive YYYY-MM-DD start and end


def pair_request(
    event_id: str | None,
    bbox: tuple[float, ...] | list[float] | None,
    before: Days | None,
    after: Days | None,
) -> tuple[tuple[float, float, float, float], Window, Window]:
    """A catalogued event's request, or an explicit bbox with inclusive day windows.

    Callers (the CLI and the HTTP route) enforce that exactly one form is given.
    """
    if event_id:
        return event_request(event_id)
    return tuple(bbox), _inclusive(*before), _inclusive(*after)


CREDENTIALS_HELP = (
    "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to an OAuth client created at "
    "https://shapps.dataspace.copernicus.eu/dashboard/ (User settings > OAuth clients)."
)


def cdse_credentials() -> tuple[str, str] | None:
    """The OAuth client id and secret from the environment, or None if either is unset."""
    client_id, client_secret = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    return (client_id, client_secret) if client_id and client_secret else None


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
        bbox, before, after = pair_request(args.event, args.bbox, args.before, args.after)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    credentials = cdse_credentials()
    if credentials is None:
        print(CREDENTIALS_HELP, file=sys.stderr)
        return 2
    try:
        with httpx.Client() as client:
            result = fetch_pair(client, *credentials, bbox, before, after, args.pair_group or args.event, event_id=args.event)
    except (CDSEError, ValueError, InvalidImageUpload, SceneStorageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
