"""Nominatim geocoding against httpx.MockTransport; no network."""

import json
import math
from pathlib import Path

import httpx
import pytest

import backend.geocode as geocode
from backend.geocode import USER_AGENT, GeocodeError, Place, aoi_for
from backend.sentinel1 import utm_grid

FLOOD_EVENTS = Path(__file__).resolve().parents[1] / "data" / "manifests" / "flood_events.v1.json"

# Trimmed from one live Nominatim search for "Patna" on 2026-10-06.
PATNA_CITY = {
    "osm_type": "way", "osm_id": 383774533, "lat": "25.6093239", "lon": "85.1235252",
    "importance": 0.6283141403317186, "addresstype": "city", "name": "Patna",
    "display_name": "Patna, Patna Rural, Patna, Bihar, India",
    "boundingbox": ["25.5389546", "25.6510479", "85.0118412", "85.2641902"],
}
PATNA_DISTRICT = {
    "osm_type": "relation", "osm_id": 1960296, "lat": "25.4680538", "lon": "85.1952852",
    "importance": 0.5903928151504004, "addresstype": "state_district", "name": "Patna",
    "display_name": "Patna, Bihar, India",
    "boundingbox": ["25.2032601", "25.7328751", "84.6869288", "86.0689271"],
}


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(geocode, "time", fake)
    monkeypatch.setattr(geocode, "_CACHE", {})
    monkeypatch.setattr(geocode, "_last_request_at", None)
    return fake


def client(payload=None, *, status: int = 200, seen: list | None = None, content: bytes | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def place_from(entry: dict) -> Place:
    (place,) = geocode.geocode(client([entry]), entry["name"] + str(entry["osm_id"]))
    return place


def test_geocode_returns_candidates_in_nominatim_order() -> None:
    places = geocode.geocode(client([PATNA_CITY, PATNA_DISTRICT]), "Patna")
    assert places[0] == Place(
        name="Patna",
        display_name="Patna, Patna Rural, Patna, Bihar, India",
        lat=25.6093239,
        lon=85.1235252,
        bbox=(85.0118412, 25.5389546, 85.2641902, 25.6510479),
        osm_type="way",
        osm_id=383774533,
        addresstype="city",
        importance=0.6283141403317186,
    )
    assert [(p.osm_type, p.osm_id) for p in places] == [("way", 383774533), ("relation", 1960296)]


def test_request_follows_nominatim_usage_policy() -> None:
    seen: list[httpx.Request] = []
    geocode.geocode(client([], seen=seen), "  Patna  ")
    (request,) = seen
    assert str(request.url.copy_with(query=None)) == "https://nominatim.openstreetmap.org/search"
    assert dict(request.url.params) == {"q": "Patna", "format": "jsonv2", "countrycodes": "in", "limit": "5"}
    assert request.headers["User-Agent"] == USER_AGENT
    assert USER_AGENT == "satquery-sih2026/0.1 (+https://github.com/sohamsssssssssssssssss/sih2026)"


def test_answers_are_cached_by_normalised_place() -> None:
    seen: list[httpx.Request] = []
    http = client([PATNA_CITY], seen=seen)
    first = geocode.geocode(http, "Patna")
    assert geocode.geocode(http, "  PATNA ") == first
    assert len(seen) == 1


def test_no_result_is_an_empty_list_and_is_cached() -> None:
    seen: list[httpx.Request] = []
    http = client([], seen=seen)
    assert geocode.geocode(http, "Atlantis") == []
    assert geocode.geocode(http, "atlantis") == []
    assert len(seen) == 1


def test_requests_are_throttled_to_one_per_second(clock: FakeTime) -> None:
    http = client([])
    geocode.geocode(http, "Patna")
    geocode.geocode(http, "Gaya")
    assert clock.sleeps == [1.0]
    clock.now += 5
    geocode.geocode(http, "Pune")
    assert clock.sleeps == [1.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": 503, "payload": {"error": "busy"}},
        {"content": b"<html>not json</html>"},
        {"payload": {"error": "not a list"}},
        {"payload": [{"osm_type": "node", "osm_id": 1}]},
        {"payload": [{**PATNA_CITY, "lat": "95.0"}]},
        {"payload": [{**PATNA_CITY, "boundingbox": ["1", "2", "3"]}]},
        {"payload": [{**PATNA_CITY, "lon": "nan"}]},
    ],
)
def test_bad_responses_raise_geocode_error_and_are_not_cached(kwargs: dict) -> None:
    seen: list[httpx.Request] = []
    http = client(seen=seen, **kwargs)
    for _ in range(2):
        with pytest.raises(GeocodeError):
            geocode.geocode(http, "Patna")
    assert len(seen) == 2


def test_transport_failure_raises_geocode_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(GeocodeError, match="ConnectError"):
        geocode.geocode(httpx.Client(transport=httpx.MockTransport(refuse)), "Patna")


@pytest.mark.parametrize("place", ["", "   ", "x" * 201])
def test_unusable_place_is_rejected_before_any_request(place: str) -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(ValueError):
        geocode.geocode(client([], seen=seen), place)
    assert seen == []


def test_relation_that_fits_uses_its_osm_bounding_box() -> None:
    kosi = next(e for e in json.loads(FLOOD_EVENTS.read_text())["events"] if e["event_id"] == "kosi-2024")
    south, north = kosi["aoi"]["bbox"][1], kosi["aoi"]["bbox"][3]
    west, east = kosi["aoi"]["bbox"][0], kosi["aoi"]["bbox"][2]
    kiratpur = place_from({
        "osm_type": "relation", "osm_id": 10286965, "lat": "26.0", "lon": "86.37",
        "addresstype": "county", "name": "Kiratpur", "display_name": "Kiratpur, Darbhanga, Bihar, India",
        "boundingbox": [str(south), str(north), str(west), str(east)],
    })
    bbox, derivation = aoi_for(kiratpur)
    assert bbox == tuple(kosi["aoi"]["bbox"])
    assert derivation == "Bounding box of OpenStreetMap relation 10286965 (Kiratpur, county), from Nominatim."


@pytest.mark.parametrize("event_id", ["silchar-2022", "kerala-2018-periyar"])
def test_point_square_matches_the_catalogued_squares(event_id: str) -> None:
    event = next(e for e in json.loads(FLOOD_EVENTS.read_text())["events"] if e["event_id"] == event_id)
    west, south, east, north = event["aoi"]["bbox"]
    point = Place(
        name="Town", display_name="Town, India", lat=(south + north) / 2, lon=(west + east) / 2,
        bbox=(west, south, east, north), osm_type="node", osm_id=1, addresstype="town", importance=None,
    )
    bbox, derivation = aoi_for(point)
    assert bbox == pytest.approx(tuple(event["aoi"]["bbox"]), abs=1e-4)
    assert derivation.startswith("20 km square centred on OpenStreetMap node 1 (Town, town)")


def test_way_gets_a_declared_square_around_its_point() -> None:
    bbox, derivation = aoi_for(place_from(PATNA_CITY))
    west, south, east, north = bbox
    assert ((west + east) / 2, (south + north) / 2) == pytest.approx((85.1235252, 25.6093239), abs=1e-4)
    grid = utm_grid(bbox)  # the UTM envelope of a degree square grows off the central meridian
    assert grid.width == pytest.approx(2000, rel=0.02)
    assert grid.height == pytest.approx(2000, rel=0.02)
    assert derivation == (
        "20 km square centred on OpenStreetMap way 383774533 (Patna, city), from Nominatim. "
        "The square is a declared choice, not a boundary."
    )


def test_relation_too_large_for_the_mosaic_falls_back_to_a_square() -> None:
    bbox, derivation = aoi_for(place_from(PATNA_DISTRICT))
    west, south, east, north = bbox
    assert ((west + east) / 2, (south + north) / 2) == pytest.approx((85.1952852, 25.4680538), abs=1e-4)
    assert derivation.startswith("20 km square centred on OpenStreetMap relation 1960296 (Patna, state_district)")
    assert "Its bounding box was not used" in derivation
    assert "5000 per side" in derivation


@pytest.mark.parametrize("side_km", [0, -5, math.nan, 60])
def test_unusable_square_side_is_rejected(side_km: float) -> None:
    with pytest.raises(ValueError):
        aoi_for(place_from(PATNA_CITY), side_km)
