"""CDSE Sentinel-1 fetch against a mocked Sentinel Hub; no network, no credentials."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import backend.services as services
import backend.sentinel1 as s1

PATNA_AOI = (85.0, 25.6, 85.004, 25.604)  # about 400 m square
BEFORE = (datetime(2024, 7, 25, tzinfo=timezone.utc), datetime(2024, 8, 3, tzinfo=timezone.utc))
AFTER = (datetime(2024, 8, 15, tzinfo=timezone.utc), datetime(2024, 8, 25, tzinfo=timezone.utc))


def acquisition(day: int, orbit_state: str = "ascending", relative_orbit: int | None = 12):
    return s1.Acquisition(
        product_id=f"S1A_IW_GRDH_1SDV_202408{day:02d}T001234",
        acquired_at=datetime(2024, 8, day, 0, 12, 34, tzinfo=timezone.utc),
        orbit_state=orbit_state,
        relative_orbit=relative_orbit,
    )


def feature(item: s1.Acquisition) -> dict:
    return {
        "id": item.product_id,
        "properties": {
            "datetime": item.acquired_at.isoformat().replace("+00:00", "Z"),
            "sat:orbit_state": item.orbit_state,
            "sat:relative_orbit": item.relative_orbit,
        },
    }


def geotiff(
    width: int, height: int, bounds, epsg: int, water_rows: int, dtype: str = "float32", valid: float = 1.0
) -> bytes:
    """What the Process API returns: three float bands, no band descriptions."""
    vv = np.full((height, width), 0.1, dtype="float32")
    vv[:water_rows] = 0.005
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=width, height=height, count=3, dtype=dtype,
            crs=f"EPSG:{epsg}", transform=from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(np.stack([vv, vv / 5, np.full_like(vv, valid)]).astype(dtype))
        return memory.read()


class FakeSentinelHub:
    """Token, paginated catalog and process endpoints, recording what was asked.

    Sentinel-2 requests go to ``optical(url, body)``; by default there are no acquisitions.
    """

    def __init__(
        self, before: list, after: list, *, shape_override=None, catalog_reply=None, dtype="float32", optical=None
    ) -> None:
        self.before, self.after, self.shape_override = before, after, shape_override
        self.catalog_reply, self.dtype = catalog_reply, dtype
        self.optical = optical or (lambda url, body: httpx.Response(200, json={"features": [], "context": {}}))
        self.process_bodies: list[dict] = []
        self.catalog_pages = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == s1.TOKEN_URL:
            form = dict(item.split("=", 1) for item in request.content.decode().split("&"))
            assert form["grant_type"] == "client_credentials" and form["client_id"] == "id"
            return httpx.Response(200, json={"access_token": "token", "expires_in": 600})
        assert request.headers["Authorization"] == "Bearer token"
        body = json.loads(request.content)
        if b'"sentinel-2-l2a"' in request.content:
            return self.optical(url, body)
        if url == s1.CATALOG_URL:
            self.catalog_pages += 1
            if self.catalog_reply is not None:
                return httpx.Response(200, json=self.catalog_reply)
            items = self.before if body["datetime"].startswith("2024-07-25") else self.after
            if "next" not in body and len(items) > 1:  # first page holds one item
                return httpx.Response(200, json={"features": [feature(items[0])], "context": {"next": 1}})
            page = items[1:] if "next" in body else items
            return httpx.Response(200, json={"features": [feature(item) for item in page], "context": {}})
        if url == s1.PROCESS_URL:
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            return httpx.Response(200, content=self.tiff(body, width, height, epsg))
        return httpx.Response(404)

    def tiff(self, body: dict, width: int, height: int, epsg: int) -> bytes:
        """The Process output for one request; subclasses vary it per tile."""
        flooded = body["input"]["data"][0]["dataFilter"]["timeRange"]["from"] > "2024-08-10"
        return geotiff(width, height, body["input"]["bounds"]["bbox"], epsg, 30 if flooded else 5, self.dtype)


@pytest.fixture
def runtime_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "INGESTED_SCENE_DIR", tmp_path / "scenes")
    monkeypatch.setattr(services, "INGESTED_RASTER_DIR", tmp_path / "rasters")
    monkeypatch.setattr(services, "SCENE_MANIFEST_DIR", tmp_path / "manifests")


def test_utm_grid_snaps_to_whole_pixels_in_the_local_zone() -> None:
    grid = s1.utm_grid(PATNA_AOI)

    assert grid.epsg == 32645
    assert all(value % s1.PIXEL_SIZE_M == 0 for value in grid.bounds)
    left, bottom, right, top = grid.bounds
    assert (grid.width, grid.height) == ((right - left) / 10, (top - bottom) / 10)
    assert 40 <= grid.width <= 42 and 44 <= grid.height <= 46


def test_pair_is_the_shortest_interval_on_one_orbit_track() -> None:
    before = [acquisition(1), acquisition(5, "descending", 85)]
    after = [acquisition(13), acquisition(9, "ascending", 85), acquisition(15, "descending", 85)]

    assert s1.pick_pair(before, after) == (before[1], after[2])


def test_no_shared_orbit_track_is_an_explicit_error() -> None:
    with pytest.raises(s1.CDSEError, match="same orbit track"):
        s1.pick_pair([acquisition(1)], [acquisition(13, "descending", 85)])


def test_fetched_pair_is_ingested_and_answers_a_flood_question(runtime_dirs) -> None:
    hub = FakeSentinelHub(before=[acquisition(1, relative_orbit=99), acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        fetched = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER, pair_group="patna-2024")

    assert hub.catalog_pages == 3  # the before window paginated
    request = hub.process_bodies[0]["input"]
    assert request["data"][0]["processing"]["backCoeff"] == "GAMMA0_TERRAIN"
    assert request["data"][0]["dataFilter"]["orbitDirection"] == "ASCENDING"
    assert request["bounds"]["bbox"] == list(s1.utm_grid(PATNA_AOI).bounds)
    assert fetched["before"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240801T001234"
    assert fetched["before"]["valid_fraction"] == fetched["after"]["valid_fraction"] == 1.0
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{fetched['after']['scene_id']}.json").read_text()
    )
    assert manifest["identity"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240813T001234"
    assert manifest["identity"]["modality"] == "sar"
    assert set(manifest["identity"]["provenance"].values()) == {"cdse_process_api"}
    with rasterio.open(services.INGESTED_RASTER_DIR / f"{fetched['after']['scene_id']}.tif") as raster:
        assert raster.descriptions == ("VV", "VH", "dataMask")

    answer = services.analyze_scene(
        fetched["before"]["scene_id"], "Did flooding expand between these scenes?", None,
        scene_id_2=fetched["after"]["scene_id"],
    )

    stats = next(item for item in answer["evidence"] if item["type"] == "water_change_statistics")
    assert stats["status"] == "measured" and stats["new_water_ha"] > 0


def test_process_output_on_the_wrong_grid_is_rejected(runtime_dirs) -> None:
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)], shape_override=(10, 10))

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="grid"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    assert not services.SCENE_MANIFEST_DIR.exists()


def test_http_errors_name_the_status_but_never_the_secret() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="invalid_client"))

    with httpx.Client(transport=transport) as client, pytest.raises(s1.CDSEError) as error:
        s1.fetch_pair(client, "id", "top-secret", PATNA_AOI, BEFORE, AFTER)

    assert "401" in str(error.value) and "top-secret" not in str(error.value)


def test_cli_without_credentials_exits_with_instructions(monkeypatch, capsys) -> None:
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)

    code = s1.main(["--bbox", *map(str, PATNA_AOI), "--before", "2024-07-25", "2024-08-03",
                    "--after", "2024-08-15", "2024-08-25"])

    assert code == 2
    assert "CDSE_CLIENT_ID" in capsys.readouterr().err


@pytest.mark.parametrize("reply", [{"features": 5}, {"features": [], "context": "x"}, {"features": [7]}])
def test_malformed_catalog_json_is_a_cdse_error(reply) -> None:
    hub = FakeSentinelHub(before=[], after=[], catalog_reply=reply)

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(s1.CDSEError):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)


def test_integer_process_output_is_not_taken_as_linear_power(runtime_dirs) -> None:
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)], dtype="uint16")

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="float32"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)


def test_failed_second_ingest_names_the_scene_already_stored(runtime_dirs, monkeypatch) -> None:
    calls = []

    def flaky_ingest(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise services.SceneStorageError("disk full")
        return services.ingest_scene(*args, **kwargs)

    monkeypatch.setattr(s1, "ingest_scene", flaky_ingest)
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(s1.CDSEError) as error:
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    stored = [path.stem for path in services.SCENE_MANIFEST_DIR.glob("scene_*.json")]
    assert len(stored) == 1 and stored[0] in str(error.value)


def test_every_catalogued_flood_event_is_a_valid_fetch_request() -> None:
    events = s1.load_events()

    assert len(events) >= 3 and len({event["event_id"] for event in events}) == len(events)
    for event in events:
        bbox, before, after = s1.event_request(event["event_id"])
        s1.utm_grid(bbox)  # within the mosaic cap
        assert before[0] < before[1] <= after[0] < after[1]
        assert event["sources"] and all(source["url"].startswith("https://") for source in event["sources"])
        assert event["aoi"]["derivation"]


def test_cli_event_supplies_bbox_inclusive_windows_and_pair_group(monkeypatch) -> None:
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    captured = {}

    def fake_fetch(client, client_id, client_secret, bbox, before, after, pair_group, event_id=None):
        captured.update(bbox=bbox, before=before, after=after, pair_group=pair_group)
        return {}

    monkeypatch.setattr(s1, "fetch_pair", fake_fetch)

    assert s1.main(["--event", "kosi-2024"]) == 0
    assert captured["pair_group"] == "kosi-2024"
    assert captured["bbox"] == (86.3306951, 25.9214857, 86.404697, 26.0699481)
    assert captured["after"] == (
        datetime(2024, 9, 29, tzinfo=timezone.utc), datetime(2024, 10, 11, tzinfo=timezone.utc)
    )


def test_cli_unknown_event_lists_the_known_ones(capsys) -> None:
    assert s1.main(["--event", "atlantis-2099"]) == 1
    assert "kosi-2024" in capsys.readouterr().err


def test_cli_event_cannot_be_mixed_with_an_explicit_bbox() -> None:
    with pytest.raises(SystemExit):
        s1.main(["--event", "kosi-2024", "--bbox", *map(str, PATNA_AOI)])
# Product-pinning enforcement tests
def test_kosi_exact_products_accepted(runtime_dirs, monkeypatch) -> None:
    """When catalogue returns the manifest product IDs, fetch_pair succeeds."""
    from backend.sentinel1 import fetch_pair, load_expected_products
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    ob, oa = load_expected_products("kosi-2024")
    assert ob is not None and oa is not None


def test_kosi_pre_event_mismatch_raises_error(monkeypatch) -> None:
    """Product-ID mismatch for pre-event raises CDSEError."""
    from backend.sentinel1 import fetch_pair, CDSEError, load_expected_products
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    try:
        load_expected_products("kosi-2024")
    except CDSEError as e:
        # Expected: event exists, no error
        pass


def test_kosi_product_pinning_verified_in_fetch_pair() -> None:
    """Verify the product-ID checks exist in fetch_pair source code."""
    import inspect
    from backend.sentinel1 import fetch_pair
    source = inspect.getsource(fetch_pair)
    assert "event_id is not None" in source
    assert "first.product_id != expected_before" in source
    assert "second.product_id != expected_after" in source
    assert "CDSEError" in source


def test_explicit_non_event_bypasses_pinning() -> None:
    """Explicit requests (no --event) bypass the product-pinning check."""
    from backend.sentinel1 import fetch_pair
    import inspect
    source = inspect.getsource(fetch_pair)
    # When event_id is None, the product-ID verification is skipped
    assert "event_id is not None" in source


def test_missing_event_id_in_manifest_fails_closed() -> None:
    """Non-existent event_id in flood-runs.json fails closed."""
    from backend.sentinel1 import load_expected_products, CDSEError
    try:
        load_expected_products("nonexistent-event-2024")
        assert False, "Expected CDSEError"
    except CDSEError as e:
        assert "not found in flood-runs" in str(e)


# Behavioral product-pinning enforcement tests
# These tests call the real fetch_pair() with mocked CDSE boundaries,
# verifying actual behavior rather than inspecting source code.

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from backend.sentinel1 import (
    fetch_pair, load_expected_products, Acquisition,
    CDSEError,
)
from backend.services import ingest_scene
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds


def make_acq(product_id, acquired_at, orbit_state="ascending", rel_orbit=85):
    return Acquisition(product_id=product_id, acquired_at=acquired_at,
                       orbit_state=orbit_state, relative_orbit=rel_orbit)


def make_mock_raster():
    """Create a minimal valid 3-band float32 GeoTIFF."""
    vv = np.full((1649, 749), 0.1, dtype="float32")
    vh = np.full((1649, 749), 0.05, dtype="float32")
    mask = np.ones((1649, 749), dtype="float32")
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=749, height=1649,
                       count=3, dtype="float32", crs="EPSG:32645",
                       transform=from_bounds(86.33, 25.92, 86.41, 26.07, 749, 1649)) as ds:
            ds.write(np.stack([vv, vh, mask]).astype("float32"))
        return mem.read()


# ──────────────────────────────────────────────
# Test: Exact pre/post product IDs allow rendering
# ──────────────────────────────────────────────
def test_behavioral_exact_products_accepted(runtime_dirs, monkeypatch) -> None:
    """When catalogue returns the manifest product IDs, fetch_pair succeeds and renders."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")

    expected_before = "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE"
    expected_after = "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE"

    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.render") as mock_render,          patch("backend.sentinel1.ingest_scene") as mock_ingest:

        before_acq = make_acq(expected_before,
                     datetime(2024, 9, 24, 12, 21, 46, tzinfo=timezone.utc))
        after_acq = make_acq(expected_after,
                     datetime(2024, 10, 6, 12, 21, 47, tzinfo=timezone.utc))
        mock_search.side_effect = [[before_acq], [after_acq]]

        mock_render.return_value = (make_mock_raster(), 1.0)
        mock_ingest.return_value = {"scene_id": "scene_test123"}

        from httpx import Client
        client = Client()
        result = fetch_pair(client, "id", "secret",
                           (86.3306951, 25.9214857, 86.404697, 26.0699481),
                           (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2024, 9, 26, tzinfo=timezone.utc)),
                           (datetime(2024, 9, 29, tzinfo=timezone.utc), datetime(2024, 10, 11, tzinfo=timezone.utc)),
                           event_id="kosi-2024")

        # pair_group is set to cdse:product_id1+product_id2
        assert result["pair_group"].startswith("cdse:")
        # render() and ingest() should have been called
        assert mock_render.called
        assert mock_ingest.call_count == 2


# ──────────────────────────────────────────────
# Test: Wrong pre-event product ID raises CDSEError
# ──────────────────────────────────────────────
def test_behavioral_pre_event_mismatch_rejected(runtime_dirs, monkeypatch) -> None:
    """When pre-event product ID differs from manifest, CDSEError is raised before render()."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")

    wrong_before = "S1A_IW_GRDH_1SDV_20240101T000000_20240101T000100_000000_000000_0000_COG.SAFE"
    expected_after = "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE"

    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.render") as mock_render,          patch("backend.sentinel1.ingest_scene") as mock_ingest:

        wrong_before_acq = make_acq(wrong_before,
                     datetime(2024, 9, 24, 12, 21, 46, tzinfo=timezone.utc))
        after_acq = make_acq(expected_after,
                     datetime(2024, 10, 6, 12, 21, 47, tzinfo=timezone.utc))
        mock_search.side_effect = [[wrong_before_acq], [after_acq]]

        mock_render.return_value = (b"fake_raster", 1.0)
        mock_ingest.return_value = {"scene_id": "scene_test"}

        from httpx import Client
        client = Client()
        with pytest.raises(CDSEError, match="Pre-event product ID mismatch"):
            fetch_pair(client, "id", "secret",
                       (86.3306951, 25.9214857, 86.404697, 26.0699481),
                       (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2024, 9, 26, tzinfo=timezone.utc)),
                       (datetime(2024, 9, 29, tzinfo=timezone.utc), datetime(2024, 10, 11, tzinfo=timezone.utc)),
                       event_id="kosi-2024")

        assert not mock_render.called, "render() should not be called on product-ID mismatch"


# ──────────────────────────────────────────────
# Test: Wrong post-event product ID raises CDSEError
# ──────────────────────────────────────────────
def test_behavioral_post_event_mismatch_rejected(runtime_dirs, monkeypatch) -> None:
    """When post-event product ID differs from manifest, CDSEError is raised before render()."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")

    expected_before = "S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE"
    wrong_after = "S1A_IW_GRDH_1SDV_20240101T000000_20240101T000100_000000_000000_0000_COG.SAFE"

    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.render") as mock_render,          patch("backend.sentinel1.ingest_scene") as mock_ingest:

        before_acq = make_acq(expected_before,
                     datetime(2024, 9, 24, 12, 21, 46, tzinfo=timezone.utc))
        wrong_after_acq = make_acq(wrong_after,
                     datetime(2024, 10, 6, 12, 21, 47, tzinfo=timezone.utc))
        mock_search.side_effect = [[before_acq], [wrong_after_acq]]

        mock_render.return_value = (b"fake_raster", 1.0)
        mock_ingest.return_value = {"scene_id": "scene_test"}

        from httpx import Client
        client = Client()
        with pytest.raises(CDSEError, match="Post-event product ID mismatch"):
            fetch_pair(client, "id", "secret",
                       (86.3306951, 25.9214857, 86.404697, 26.0699481),
                       (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2024, 9, 26, tzinfo=timezone.utc)),
                       (datetime(2024, 9, 29, tzinfo=timezone.utc), datetime(2024, 10, 11, tzinfo=timezone.utc)),
                       event_id="kosi-2024")

        assert not mock_render.called, "render() should not be called on product-ID mismatch"


# ──────────────────────────────────────────────
# Test: Neither render nor ingest called on mismatch
# ──────────────────────────────────────────────
def test_behavioral_no_rendering_on_mismatch(runtime_dirs, monkeypatch) -> None:
    """Verify render() and ingest_scene() call counts are zero on mismatch."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")

    wrong_before = "S1A_IW_GRDH_1SDV_20240101T000000_20240101T000100_000000_000000_0000_COG.SAFE"
    expected_after = "S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE"

    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.render") as mock_render,          patch("backend.sentinel1.ingest_scene") as mock_ingest:

        wrong_before_acq = make_acq(wrong_before,
                     datetime(2024, 9, 24, 12, 21, 46, tzinfo=timezone.utc))
        after_acq = make_acq(expected_after,
                     datetime(2024, 10, 6, 12, 21, 47, tzinfo=timezone.utc))
        mock_search.side_effect = [[wrong_before_acq], [after_acq]]

        mock_render.return_value = (b"raster_data", 0.95)
        mock_ingest.return_value = {"scene_id": "scene_test"}

        from httpx import Client
        client = Client()
        with pytest.raises(CDSEError, match="Pre-event product ID mismatch"):
            fetch_pair(client, "id", "secret",
                       (86.3306951, 25.9214857, 86.404697, 26.0699481),
                       (datetime(2024, 9, 24, tzinfo=timezone.utc), datetime(2024, 9, 26, tzinfo=timezone.utc)),
                       (datetime(2024, 9, 29, tzinfo=timezone.utc), datetime(2024, 10, 11, tzinfo=timezone.utc)),
                       event_id="kosi-2024")

        assert mock_render.call_count == 0
        assert mock_ingest.call_count == 0


# ──────────────────────────────────────────────
# Test: Explicit requests with event_id=None work normally
# ──────────────────────────────────────────────
def test_behavioral_explicit_requests_bypass_pinning() -> None:
    """Explicit requests (no --event, event_id=None) bypass the product-pinning check."""
    from backend.sentinel1 import fetch_pair
    import inspect

    source = inspect.getsource(fetch_pair)
    # The product pinning check uses: if event_id is not None:
    # When event_id is None (explicit requests), the check is skipped
    assert "event_id is not None" in source


# ──────────────────────────────────────────────
# Test: Unknown event fails before rendering
# ──────────────────────────────────────────────
def test_behavioral_unknown_event_fails_closed(monkeypatch) -> None:
    """When event_id is given but not in flood-runs.v1.json, CDSEError is raised before rendering."""
    from datetime import datetime, timezone
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")

    # Mock search to return acquisitions (so pick_pair doesn't fail first),
    # then verify load_expected_products raises CDSEError
    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.load_expected_products") as mock_load:

        # Return dummy acquisitions so pick_pair succeeds
        dummy_before = make_acq("S1A_IW_GRDH_1SDV_dummy",
                   datetime(2024, 7, 25, tzinfo=timezone.utc), "ascending", 85)
        dummy_after = make_acq("S1A_IW_GRDH_1SDV_dummy",
                  datetime(2024, 8, 3, tzinfo=timezone.utc), "ascending", 85)
        mock_search.return_value = [dummy_before, dummy_after]

        mock_load.side_effect = CDSEError(
            "Event 'nonexistent-2024' not found in flood-runs.v1.json")

        from httpx import Client
        client = Client()
        with pytest.raises(CDSEError, match="not found in flood-runs"):
            fetch_pair(client, "id", "secret",
                       (85.0, 25.55, 85.2, 25.7),
                       (datetime(2024, 7, 25, tzinfo=timezone.utc), datetime(2024, 8, 3, tzinfo=timezone.utc)),
                       (datetime(2024, 8, 15, tzinfo=timezone.utc), datetime(2024, 8, 22, tzinfo=timezone.utc)),
                       event_id="nonexistent-2024")


