"""Event selection preflight: mock HTTP and rendering, no imagery or optical work."""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import httpx
import pytest

from backend import sentinel1 as s1
from backend.test_sentinel1 import feature


def kosi_acquisitions():
    pinned = s1.load_pinned_run("kosi-2024")
    return [s1.Acquisition(record["product_id"], datetime.fromisoformat(record["acquired_at"].replace("Z", "+00:00")),
                           pinned["orbit_direction"], pinned["relative_orbit"])
            for record in (pinned["before"], pinned["after"])]


def test_event_selects_pins_instead_of_closer_pair(monkeypatch):
    import backend.optical_clouds as optical

    first, second = kosi_acquisitions()
    distractor = replace(first, product_id="S1A_DISTRACTOR", acquired_at=first.acquired_at + timedelta(days=1))
    requests, renders, ingests = [], [], []

    def handler(request):
        assert str(request.url) == s1.CATALOG_URL
        body = json.loads(request.content)
        requests.append(body)
        assert body == {"collections": [s1.COLLECTION], "ids": [first.product_id if len(requests) == 1 else second.product_id], "limit": 2}
        selected = next(item for item in [distractor, first, second] if item.product_id in body["ids"])
        return httpx.Response(200, json={"features": [feature(selected)]})

    monkeypatch.setattr(s1, "access_token", lambda *args: "offline")
    monkeypatch.setattr(s1, "pick_pair", lambda *args: pytest.fail("Heuristic selection forbidden"))
    monkeypatch.setattr(s1, "render", lambda client, token, item, grid, *, strict: renders.append((item, strict)) or (b"offline", 1, {}))
    monkeypatch.setattr(optical, "check_optical_clouds", lambda *args: {"verdict": "offline_stub"})
    monkeypatch.setattr(s1, "_ingest", lambda rendered, item, *args: ingests.append(item) or {"scene_id": item.product_id})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        s1.fetch_pair(client, "offline", "offline", *s1.event_request("kosi-2024"), event_id="kosi-2024")
    assert renders == [(first, True), (second, True)]
    assert ingests == [first, second]


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("bad", ["missing", "duplicate", "id", "timestamp", "orbit", "direction", "unknown_orbit"])
def test_bad_pinned_selection_stops_before_render_or_ingest(monkeypatch, side, bad):
    first, second = kosi_acquisitions()
    candidates = [[first], [second]]
    target = candidates[side][0]
    if bad == "missing":
        candidates[side] = []
    elif bad == "duplicate":
        candidates[side] *= 2
    else:
        updates = {"id": {"product_id": "S1A_WRONG"},
                   "timestamp": {"acquired_at": target.acquired_at + timedelta(seconds=1)},
                   "orbit": {"relative_orbit": 86}, "direction": {"orbit_state": "descending"},
                   "unknown_orbit": {"relative_orbit": None}}
        candidates[side] = [replace(target, **updates[bad])]
    monkeypatch.setattr(s1, "access_token", lambda *args: "offline")
    monkeypatch.setattr(s1, "search", lambda *args, **kwargs: candidates.pop(0))
    monkeypatch.setattr(s1, "render", lambda *args, **kwargs: pytest.fail("Render forbidden"))
    monkeypatch.setattr(s1, "_ingest", lambda *args: pytest.fail("Ingest forbidden"))
    s1_request = s1.event_request("kosi-2024")
    with pytest.raises(s1.CDSEError):
        s1.fetch_pair(None, "offline", "offline", *s1_request, event_id="kosi-2024")


@pytest.mark.parametrize("bad", ["collection", "mode", "polarization", "resolution"])
def test_pinned_catalogue_mode_rejected(bad):
    item = feature(kosi_acquisitions()[0])
    if bad == "collection":
        item["collection"] = "wrong"
    else:
        item["properties"][{"mode": "sar:instrument_mode", "polarization": "s1:polarization",
                             "resolution": "s1:resolution"}[bad]] = "wrong"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"features": [item]}))) as client:
        with pytest.raises(s1.CDSEError, match="SAR mode mismatch"):
            s1.search(client, "offline", s1.event_request("kosi-2024")[0],
                      s1.event_request("kosi-2024")[1], product_id=item["id"])


def test_kosi_grid_and_request_budget():
    bbox, before, after = s1.event_request("kosi-2024")
    pinned = s1.load_pinned_run("kosi-2024")
    first, second = kosi_acquisitions()
    assert before[0] <= first.acquired_at < before[1] <= after[0] <= second.acquired_at < after[1]
    grid = s1.utm_grid(bbox)
    assert (grid.epsg, grid.width, grid.height) == (32645, 749, 1649)
    assert pinned["grid"] == {"epsg": grid.epsg, "width": grid.width, "height": grid.height}
    tiles = [s1._tile(grid, row, col) for row in range(0, grid.height, s1.MAX_SIDE_PIXELS)
             for col in range(0, grid.width, s1.MAX_SIDE_PIXELS)]
    assert len(tiles) == 1
    assert 2 * len(tiles) == 2
    assert grid.width * grid.height * 3 * 4 < s1.MAX_TIFF_BYTES


def test_unknown_event_fails_before_authentication(monkeypatch):
    monkeypatch.setattr(s1, "access_token", lambda *args: pytest.fail("OAuth forbidden"))
    with pytest.raises(s1.CDSEError, match="not found"):
        s1.fetch_pair(None, "offline", "offline", *s1.event_request("kosi-2024"), event_id="unknown")


def test_pinned_timestamp_outside_requested_window_fails(monkeypatch):
    acquisitions = kosi_acquisitions()
    monkeypatch.setattr(s1, "access_token", lambda *args: "offline")
    monkeypatch.setattr(s1, "search", lambda *args, **kwargs: [acquisitions[0]])
    monkeypatch.setattr(s1, "render", lambda *args, **kwargs: pytest.fail("Render forbidden"))
    bbox, before, after = s1.event_request("kosi-2024")
    before = (before[0], acquisitions[0].acquired_at)
    with pytest.raises(s1.CDSEError, match="timestamp or orbit"):
        s1.fetch_pair(None, "offline", "offline", bbox, before, after, event_id="kosi-2024")
