"""Geocode a place name to OpenStreetMap candidates and a Sentinel-1 AOI.

``geocode`` asks OpenStreetMap Nominatim (India only, top 5) through an
injected httpx.Client, at most one request per second as the Nominatim usage
policy requires, and caches answers by normalised place name. ``aoi_for``
turns one candidate into a fetchable bbox plus a sentence naming the rule
that produced it, in the style of data/manifests/flood_events.v1.json.
Coordinates only ever come from Nominatim: no result is an empty list, never
a guess.

Ceilings: the throttle and cache are per process (the app runs one worker),
and requests are serialised under one lock, so a slow Nominatim answer holds
up other lookups for up to REQUEST_TIMEOUT_SECONDS. The cache never expires
and drops its oldest entry past MAX_CACHE_ENTRIES. The square uses a
spherical 111.32 km per degree, within about 1% across India.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.sentinel1 import utm_grid

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "satquery-sih2026/0.1 (+https://github.com/sohamsssssssssssssssss/sih2026)"
COUNTRY_CODES = "in"
RESULT_LIMIT = 5
MIN_REQUEST_INTERVAL_SECONDS = 1.0  # Nominatim usage policy: an absolute maximum of 1 request/s
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_PLACE_CHARS = 200
MAX_CACHE_ENTRIES = 1024
DEFAULT_SIDE_KM = 20.0  # the declared square used for Silchar and Aluva in flood_events.v1.json
KM_PER_DEGREE = 111.32
BBOX_DECIMALS = 4  # about 11 m, as in flood_events.v1.json

BBox = tuple[float, float, float, float]  # west, south, east, north in degrees


class GeocodeError(RuntimeError):
    """Nominatim could not supply a usable answer."""


@dataclass(frozen=True)
class Place:
    name: str | None
    display_name: str
    lat: float
    lon: float
    bbox: BBox
    osm_type: str
    osm_id: int
    addresstype: str | None
    importance: float | None


_LOCK = threading.Lock()
_CACHE: dict[str, tuple[Place, ...]] = {}
_last_request_at: float | None = None


def geocode(client: httpx.Client, place: str) -> list[Place]:
    """Nominatim's candidates for ``place`` in India, best first; [] when it knows none."""
    query = " ".join(place.split())
    if not query or len(query) > MAX_PLACE_CHARS:
        raise ValueError(f"A place name of 1 to {MAX_PLACE_CHARS} characters is required.")
    key = query.casefold()
    # ponytail: one lock covers cache and request; per-key locks if lookups ever get busy
    with _LOCK:
        places = _CACHE.get(key)
        if places is None:
            _wait_turn()
            places = _search(client, query)
            if len(_CACHE) >= MAX_CACHE_ENTRIES:
                del _CACHE[next(iter(_CACHE))]
            _CACHE[key] = places
    return list(places)


def _wait_turn() -> None:
    global _last_request_at
    if _last_request_at is not None:
        wait = _last_request_at + MIN_REQUEST_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    _last_request_at = time.monotonic()


def _search(client: httpx.Client, query: str) -> tuple[Place, ...]:
    params = {"q": query, "format": "jsonv2", "countrycodes": COUNTRY_CODES, "limit": RESULT_LIMIT}
    try:
        response = client.get(
            SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except httpx.HTTPError as exc:
        raise GeocodeError(f"Nominatim request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise GeocodeError(f"Nominatim returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodeError("Nominatim returned non-JSON") from exc
    if not isinstance(payload, list):
        raise GeocodeError("Nominatim returned unexpected JSON")
    return tuple(_place(entry) for entry in payload[:RESULT_LIMIT])


def _text(value: Any) -> str | None:
    return str(value) if value else None


def _place(entry: Any) -> Place:
    try:
        south, north, west, east = (float(value) for value in entry["boundingbox"])
        importance = entry.get("importance")
        place = Place(
            name=_text(entry.get("name")),
            display_name=str(entry["display_name"]),
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            bbox=(west, south, east, north),
            osm_type=str(entry["osm_type"]),
            osm_id=int(entry["osm_id"]),
            addresstype=_text(entry.get("addresstype")),
            importance=None if importance is None else float(importance),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise GeocodeError("Nominatim returned a malformed result") from exc
    west, south, east, north = place.bbox
    in_range = (
        -90 <= place.lat <= 90
        and -180 <= place.lon <= 180
        and -90 <= south <= north <= 90
        and -180 <= west <= east <= 180
    )
    if not in_range:  # NaN fails every comparison, so it lands here too
        raise GeocodeError("Nominatim returned coordinates out of range")
    return place


def aoi_for(place: Place, side_km: float = DEFAULT_SIDE_KM) -> tuple[BBox, str]:
    """A Sentinel-1 fetch AOI for ``place`` and the rule that produced it.

    A relation is an administrative boundary, so its OSM bbox is used when it
    fits the mosaic limit (backend.sentinel1.utm_grid). A node's OSM box is a
    fixed-size placeholder and a way may be a road or river, so nodes, ways
    and oversized relations get a declared ``side_km`` square centred on the
    OSM point. Raises ValueError when that square itself cannot be fetched.
    """
    if not (math.isfinite(side_km) and side_km > 0):
        raise ValueError("side_km must be a positive number of kilometres.")
    label = ", ".join(part for part in (place.name or place.display_name, place.addresstype) if part)
    source = f"OpenStreetMap {place.osm_type} {place.osm_id} ({label})"
    rejected = ""
    if place.osm_type == "relation":
        try:
            utm_grid(place.bbox)
        except ValueError as exc:
            rejected = f" Its bounding box was not used: {exc}."
        else:
            return place.bbox, f"Bounding box of {source}, from Nominatim."
    square = _square(place.lat, place.lon, side_km)
    utm_grid(square)
    return square, (
        f"{side_km:g} km square centred on {source}, from Nominatim. "
        f"The square is a declared choice, not a boundary.{rejected}"
    )


def _square(lat: float, lon: float, side_km: float) -> BBox:
    half_lat = side_km / 2 / KM_PER_DEGREE
    half_lon = half_lat / math.cos(math.radians(lat))
    west, south, east, north = (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
    return (
        round(west, BBOX_DECIMALS),
        round(south, BBOX_DECIMALS),
        round(east, BBOX_DECIMALS),
        round(north, BBOX_DECIMALS),
    )
