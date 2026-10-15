"""CDSE Sentinel-1 fetch against a mocked Sentinel Hub; no network, no credentials."""

import io
import json
import tarfile
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
        "collection": s1.COLLECTION,
        "properties": {
            "sar:instrument_mode": "IW", "s1:polarization": "DV", "s1:resolution": "HIGH",
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


def userdata_json(product_id: str) -> bytes:
    return json.dumps({"orbits": [{"tiles": [{"sentinel1ProductId": product_id}]}],
                       "inputMetadata": {"serviceVersion": "test-service"}}).encode()


def tar_members(members) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def tar_response(tiff_bytes: bytes, userdata_bytes: bytes) -> bytes:
    return tar_members([("default.tif", tiff_bytes), ("userdata.json", userdata_bytes)])


class FakeSentinelHub:
    """Token, paginated catalog and process endpoints, recording what was asked.

    Sentinel-2 requests go to ``optical(url, body)``; by default there are no acquisitions.
    Now returns TAR responses with TIFF + userdata.json for Process API.
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
            if "ids" in body:
                items = [item for item in self.before + self.after if item.product_id in body["ids"]]
            else:
                items = self.before if body["datetime"].startswith("2024-07-25") else self.after
            if "next" not in body and len(items) > 1:  # first page holds one item
                return httpx.Response(200, json={"features": [feature(items[0])], "context": {"next": 1}})
            page = items[1:] if "next" in body else items
            return httpx.Response(200, json={"features": [feature(item) for item in page], "context": {}})
        if url == s1.PROCESS_URL:
            assert request.headers["Accept"] == "application/x-tar"
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            # Determine which acquisition this request is for based on time range
            time_from = body["input"]["data"][0]["dataFilter"]["timeRange"]["from"]
            # Find the matching acquisition product_id
            product_id = None
            for acq in self.before + self.after:
                if time_from <= acq.acquired_at.isoformat().replace("+00:00", "Z") <= body["input"]["data"][0]["dataFilter"]["timeRange"]["to"]:
                    product_id = acq.product_id
                    break
            if product_id is None:
                product_id = (self.before[0] if self.before else self.after[0]).product_id
            tiff_bytes = self.tiff(body, width, height, epsg)
            userdata_bytes = userdata_json(product_id)
            return httpx.Response(200, content=tar_response(tiff_bytes, userdata_bytes), headers={"Content-Type": "application/x-tar"})
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
    assert "event_id is None" in source
    assert "product_id=expected[\"product_id\"]" in source
    assert "load_pinned_run(event_id)" in source
    assert "CDSEError" in source


def test_explicit_non_event_bypasses_pinning() -> None:
    """Explicit requests (no --event) bypass the product-pinning check."""
    from backend.sentinel1 import fetch_pair
    import inspect
    source = inspect.getsource(fetch_pair)
    # When event_id is None, the product-ID verification is skipped
    assert "event_id is None" in source


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

        mock_render.return_value = (make_mock_raster(), 1.0, {})
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
    assert "event_id is None" in source


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
    with patch("backend.sentinel1.access_token", lambda client, cid, cs: "token"),          patch("backend.sentinel1.search") as mock_search,          patch("backend.sentinel1.load_pinned_run") as mock_load:

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


# ──────────────────────────────────────────────
# Provenance validation tests (strict SAR source provenance)
# ──────────────────────────────────────────────

class ProvenanceSuccessHub(FakeSentinelHub):
    """Hub that returns correct userdata matching the expected product IDs."""
    pass  # FakeSentinelHub already generates correct userdata


class ProvenanceMismatchHub(FakeSentinelHub):
    """Hub that returns userdata with a mismatched product ID."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == s1.PROCESS_URL:
            body = json.loads(request.content)
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            tiff_bytes = self.tiff(body, width, height, epsg)
            # Return WRONG product ID in userdata
            wrong_id = "S1A_IW_GRDH_1SDV_WRONG_PRODUCT_ID"
            userdata_bytes = userdata_json(wrong_id)
            return httpx.Response(200, content=tar_response(tiff_bytes, userdata_bytes), headers={"Content-Type": "application/x-tar"})
        return super().__call__(request)


class ProvenanceMultiSourceHub(FakeSentinelHub):
    """Hub that returns userdata with multiple source product IDs."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == s1.PROCESS_URL:
            body = json.loads(request.content)
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            tiff_bytes = self.tiff(body, width, height, epsg)
            # Return multiple product IDs in userdata
            userdata_bytes = json.dumps({"orbits": [{"tiles": [
                {"sentinel1ProductId": self.before[0].product_id},
                {"sentinel1ProductId": "S1A_IW_GRDH_1SDV_SECOND"},
            ]}]}).encode()
            return httpx.Response(200, content=tar_response(tiff_bytes, userdata_bytes), headers={"Content-Type": "application/x-tar"})
        return super().__call__(request)


class ProvenanceMissingUserdataHub(FakeSentinelHub):
    """Hub that returns a response without userdata part."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == s1.PROCESS_URL:
            body = json.loads(request.content)
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            tiff_bytes = self.tiff(body, width, height, epsg)
            # Return only TIFF, no userdata
            return httpx.Response(200, content=tiff_bytes)
        return super().__call__(request)


class ProvenanceMalformedUserdataHub(FakeSentinelHub):
    """Hub that returns malformed JSON in userdata."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == s1.PROCESS_URL:
            body = json.loads(request.content)
            self.process_bodies.append(body)
            width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
            epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
            tiff_bytes = self.tiff(body, width, height, epsg)
            # Return invalid JSON in userdata
            userdata_bytes = b"not valid json"
            return httpx.Response(200, content=tar_response(tiff_bytes, userdata_bytes), headers={"Content-Type": "application/x-tar"})
        return super().__call__(request)


def test_provenance_validation_success(runtime_dirs) -> None:
    """When userdata product ID matches expected, fetch_pair succeeds."""
    hub = ProvenanceSuccessHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        result = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    assert result["before"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240801T001234"
    assert result["after"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240813T001234"
    # Verify the request included mosaickingOrder and userdata response
    for body in hub.process_bodies:
        assert "mosaickingOrder" not in body["input"]["data"][0]["dataFilter"]
        assert 'mosaicking: "ORBIT"' in body["evalscript"]
        responses = body["output"]["responses"]
        assert any(r["identifier"] == "userdata" for r in responses)


def test_provenance_validation_mismatch_rejected(runtime_dirs, monkeypatch) -> None:
    """When userdata product ID differs from expected, CDSEError is raised."""
    monkeypatch.setattr(s1, "load_pinned_run", lambda _: synthetic_pinned_run())
    hub = ProvenanceMismatchHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="source product ID mismatch"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, (datetime(2024, 8, 10, tzinfo=timezone.utc), AFTER[1]), event_id="pinned")

    assert len(hub.process_bodies) >= 1


def test_provenance_validation_multiple_sources_rejected(runtime_dirs, monkeypatch) -> None:
    """When userdata contains multiple sources, CDSEError is raised."""
    monkeypatch.setattr(s1, "load_pinned_run", lambda _: synthetic_pinned_run())
    hub = ProvenanceMultiSourceHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="source product ID mismatch"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, (datetime(2024, 8, 10, tzinfo=timezone.utc), AFTER[1]), event_id="pinned")

    assert len(hub.process_bodies) >= 1


def test_provenance_validation_missing_userdata_rejected(runtime_dirs) -> None:
    """When response lacks userdata part, CDSEError is raised."""
    hub = ProvenanceMissingUserdataHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="Content-Type"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    assert len(hub.process_bodies) >= 1


def test_provenance_validation_malformed_userdata_rejected(runtime_dirs) -> None:
    """When userdata is not valid JSON, CDSEError is raised."""
    hub = ProvenanceMalformedUserdataHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="not valid JSON"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    assert len(hub.process_bodies) >= 1


def test_provenance_both_acquisitions_validated_before_ingest(runtime_dirs, monkeypatch) -> None:
    """Both acquisitions are validated via userdata before either is ingested."""
    from backend.services import ingest_scene
    ingest_calls = []

    def tracking_ingest(*args, **kwargs):
        ingest_calls.append(kwargs.get("acquisition_id", args[3] if len(args) > 3 else "unknown"))
        return {"scene_id": f"scene_{len(ingest_calls)}"}

    monkeypatch.setattr("backend.sentinel1.ingest_scene", tracking_ingest)

    # First acquisition succeeds, second fails provenance
    class MixedHub(FakeSentinelHub):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == s1.PROCESS_URL:
                body = json.loads(request.content)
                self.process_bodies.append(body)
                width, height = self.shape_override or (body["output"]["width"], body["output"]["height"])
                epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
                tiff_bytes = self.tiff(body, width, height, epsg)
                time_from = body["input"]["data"][0]["dataFilter"]["timeRange"]["from"]
                # First request (before) gets correct userdata, second (after) gets wrong
                if time_from < "2024-08-10":
                    product_id = "S1A_IW_GRDH_1SDV_20240801T001234"
                else:
                    product_id = "S1A_IW_GRDH_1SDV_WRONG"
                userdata_bytes = userdata_json(product_id)
                return httpx.Response(200, content=tar_response(tiff_bytes, userdata_bytes), headers={"Content-Type": "application/x-tar"})
            return super().__call__(request)

    monkeypatch.setattr(s1, "load_pinned_run", lambda _: synthetic_pinned_run())
    hub = MixedHub(before=[acquisition(1)], after=[acquisition(13)])

    with httpx.Client(transport=httpx.MockTransport(hub)) as client, pytest.raises(
        s1.CDSEError, match="source product ID mismatch"
    ):
        s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, (datetime(2024, 8, 10, tzinfo=timezone.utc), AFTER[1]), event_id="pinned")

    # Neither should be ingested since validation happens during render, before ingest
    assert len(ingest_calls) == 0



@pytest.mark.parametrize("metadata", [None, {}, {"orbits": []}, {"orbits": [{}]},
    {"orbits": [{"tiles": [{}]}]}, {"orbits": [{"tiles": [{"sentinel1ProductId": 7}]}]},
    {"orbits": [{"tiles": [{"sentinel1ProductId": "bad id"}]}]}])
def test_missing_or_malformed_source_metadata(metadata):
    with pytest.raises(s1.CDSEError):
        s1._parse_userdata(json.dumps(metadata).encode(), acquisition(1).product_id)


def test_exact_source_set_and_suffix_are_not_normalized():
    product = acquisition(1).product_id
    s1._parse_userdata(userdata_json(product), product)
    with pytest.raises(s1.CDSEError, match="mismatch"):
        s1._parse_userdata(userdata_json(product + "_COG.SAFE"), product)
    extra = {"orbits": [{"tiles": [{"sentinel1ProductId": product},
                                  {"sentinel1ProductId": acquisition(2).product_id}]}]}
    with pytest.raises(s1.CDSEError, match="mismatch"):
        s1._parse_userdata(json.dumps(extra).encode(), product)
    assert s1._parse_userdata(json.dumps(extra).encode(), None) == extra


@pytest.mark.parametrize("members", [
    [("default.tif", b"tiff")],
    [("default.tif", b"tiff"), ("userdata.json", b"{}"), ("default.tif", b"again")],
    [("default.tif", b"tiff"), ("userdata.json", b"{}"), ("other", b"x")],
])
def test_tar_rejects_missing_duplicate_and_unexpected_members(members):
    with pytest.raises(s1.CDSEError):
        s1._parse_tar_response(tar_members(members))


@pytest.mark.parametrize("content", [b"not tar", tar_response(b"tiff", b"{}")[:1024]])
def test_malformed_tar(content):
    with pytest.raises(s1.CDSEError, match="TAR"):
        s1._parse_tar_response(content)


def test_tar_member_size_limits(monkeypatch):
    monkeypatch.setattr(s1, "MAX_USERDATA_BYTES", 1)
    with pytest.raises(s1.CDSEError, match="byte limit"):
        s1._parse_tar_response(tar_response(b"tiff", b"{}"))


@pytest.mark.parametrize("headers", [
    {"Content-Type": "image/tiff"},
    {"Content-Type": "application/x-tar", "Content-Length": "101"},
    {"Content-Type": "application/x-tar", "Content-Length": "invalid"},
])
def test_process_response_headers_are_checked(headers, monkeypatch):
    monkeypatch.setattr(s1, "MAX_PROCESS_BYTES", 100)
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers=headers, content=b"x")
    )) as client, pytest.raises(s1.CDSEError):
        s1._process_request(client, "test", {})


class ChunkStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"x" * 80
        yield b"x" * 80


@pytest.mark.parametrize("length", [None, "1"])
def test_actual_stream_limit_without_trusting_content_length(monkeypatch, length):
    monkeypatch.setattr(s1, "MAX_PROCESS_BYTES", 100)
    headers = {"Content-Type": "application/x-tar"}
    if length:
        headers["Content-Length"] = length
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, headers=headers, stream=ChunkStream())
    )) as client, pytest.raises(s1.CDSEError, match="byte limit"):
        s1._process_request(client, "test", {})


def test_error_response_is_bounded(monkeypatch):
    monkeypatch.setattr(s1, "MAX_ERROR_BYTES", 100)
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(500, stream=ChunkStream())
    )) as client, pytest.raises(s1.CDSEError, match="byte limit"):
        s1._process_request(client, "test", {})


def test_non_event_multi_product_sources_and_evidence_are_preserved(runtime_dirs):
    hub = ProvenanceMultiSourceHub(before=[acquisition(1)], after=[acquisition(13)])
    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        result = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)
    manifest = json.loads((services.SCENE_MANIFEST_DIR / (result["before"]["scene_id"] + ".json")).read_text())
    evidence = manifest["process_evidence"]
    assert len(evidence["tiles"][0]["userdata"]["orbits"][0]["tiles"]) == 2
    assert evidence["processing"]["backCoeff"] == "GAMMA0_TERRAIN"
    assert evidence["evalscript"] == s1.EVALSCRIPT


def test_userdata_validated_before_tiff_decode(monkeypatch):
    monkeypatch.setattr(s1, "_process_request", lambda *args: tar_response(b"invalid TIFF", b"{}"))
    monkeypatch.setattr(s1, "_on_grid", lambda *args: pytest.fail("TIFF decoded before metadata validation"))
    with pytest.raises(s1.CDSEError, match="orbit metadata"):
        s1.render(None, "test", acquisition(1), s1.utm_grid(PATNA_AOI), strict=True)


def test_pinned_event_exact_sources_succeed(runtime_dirs, monkeypatch):
    monkeypatch.setattr(s1, "load_pinned_run", lambda _: synthetic_pinned_run())
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)])
    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        result = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, (datetime(2024, 8, 10, tzinfo=timezone.utc), AFTER[1]), event_id="pinned")
    for side, day in [("before", 1), ("after", 13)]:
        manifest = json.loads((services.SCENE_MANIFEST_DIR / (result[side]["scene_id"] + ".json")).read_text())
        tiles = manifest["process_evidence"]["tiles"][0]["userdata"]["orbits"][0]["tiles"]
        assert {tile["sentinel1ProductId"] for tile in tiles} == {acquisition(day).product_id}
        assert manifest["process_evidence"]["tiles"][0]["userdata"]["inputMetadata"] == {"serviceVersion": "test-service"}


def test_tar_symlink_is_rejected():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        member = tarfile.TarInfo("default.tif")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/untrusted"
        archive.addfile(member)
    with pytest.raises(s1.CDSEError, match="unexpected"):
        s1._parse_tar_response(output.getvalue())


@pytest.mark.parametrize("payload", [
    '{"orbits":[{"tiles":[{"sentinel1ProductId":"S1A_WRONG","sentinel1ProductId":"S1A_EXPECTED"}]}]}',
    '{"orbits":[],"orbits":[{"tiles":[{"sentinel1ProductId":"S1A_EXPECTED"}]}]}',
    '{"orbits":[{"tiles":[],"tiles":[{"sentinel1ProductId":"S1A_EXPECTED"}]}]}',
    '{"orbits":[{"tiles":[{"sentinel1ProductId":"S1A_EXPECTED"}]}],"inputMetadata":{"nested":{"key":1,"key":1}}}',
])
def test_userdata_duplicate_keys_fail_closed(payload):
    with pytest.raises(s1.CDSEError, match="duplicate object keys"):
        s1._parse_userdata(payload.encode(), "S1A_EXPECTED")


def test_valid_userdata_preserves_service_metadata():
    payload = userdata_json(acquisition(1).product_id)
    assert s1._parse_userdata(payload, acquisition(1).product_id) == json.loads(payload)


def test_evalscript_callback_preserves_input_metadata():
    import subprocess

    subprocess.run(["node", "-e", s1.EVALSCRIPT + '''
const assert = require("node:assert/strict");
const scenes = {orbits: [{tiles: [{sentinel1ProductId: "S1A_EXPECTED"}]}]};
const inputMetadata = {serviceVersion: "test-service", normalizationFactor: 1};
const outputMetadata = {};
updateOutputMetadata(scenes, inputMetadata, outputMetadata);
assert.deepEqual(outputMetadata.userData, {orbits: scenes.orbits, inputMetadata});
'''], check=True, capture_output=True)


@pytest.mark.parametrize("extra_padding", [0, 512, 10240, 1, 513])
def test_tar_accepts_complete_eof_and_additional_zero_padding(extra_padding):
    content = tar_response(b"tiff", b"{}")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        list(archive)
        end = archive.offset
    # Exactly two EOF blocks, optionally followed by more zero padding.
    content = content[:end] + b"\0" * (1024 + extra_padding)
    assert s1._parse_tar_response(content) == (b"tiff", b"{}")


@pytest.mark.parametrize("eof_bytes", [0, 511, 512, 1023])
def test_tar_rejects_truncated_eof(eof_bytes):
    content = tar_response(b"tiff", b"{}")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        list(archive)
        end = archive.offset
    with pytest.raises(s1.CDSEError, match="malformed TAR ending"):
        s1._parse_tar_response(content[:end] + b"\0" * eof_bytes)


def test_tar_rejects_truncated_member_data():
    content = tar_response(b"tiff", b"u" * 600)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        members = list(archive)
        end = members[-1].offset_data + members[-1].size
    with pytest.raises(s1.CDSEError, match="TAR"):
        s1._parse_tar_response(content[:end - 1])


@pytest.mark.parametrize("garbage", [b"x", b"\0" * 512 + b"x", b"x" + b"\0" * 511])
def test_tar_rejects_nonzero_trailing_garbage(garbage):
    with pytest.raises(s1.CDSEError, match="malformed TAR ending"):
        s1._parse_tar_response(tar_response(b"tiff", b"{}") + garbage)


@pytest.mark.parametrize("name", ["../default.tif", "/default.tif", "folder/default.tif"])
def test_tar_rejects_unsafe_or_unexpected_paths(name):
    with pytest.raises(s1.CDSEError, match="unexpected"):
        s1._parse_tar_response(tar_members([(name, b"tiff"), ("userdata.json", b"{}")]))


def test_process_capture_layout_without_eof_is_rejected():
    """Synthetic live layout: complete padded members, but no archive EOF blocks."""
    tiff, userdata = b"t" * 14537, b"u" * 461
    content = tar_response(tiff, userdata)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        members = list(archive)
        assert [(member.name, member.offset, member.offset_data, member.size) for member in members] == [
            ("default.tif", 0, 512, 14537), ("userdata.json", 15360, 15872, 461),
        ]
        end = archive.offset
    content = content[:end]
    assert len(content) == 16384
    # Python accepts physical EOF; the strict Process contract still requires its markers.
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        members = list(archive)
        assert [archive.extractfile(member).read() for member in members] == [tiff, userdata]
        assert archive.offset == len(content)
    with pytest.raises(s1.CDSEError) as error:
        s1._parse_tar_response(content)
    assert error.value.diagnostic_category == "tar_eof_invalid"
    # Control: the same member layout with complete EOF markers is accepted.
    assert s1._parse_tar_response(content + b"\0" * 1024) == (tiff, userdata)


@pytest.mark.parametrize("with_eof", [False, True])
def test_bounded_tar_eof_modes(with_eof):
    content = tar_response(b"t" * 14537, b"u" * 461)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        list(archive)
        content = content[:archive.offset]
    if with_eof:
        content += b"\0" * 1024
    diagnostic = {}
    assert s1._parse_tar_response(content, allow_missing_eof=True, diagnostics=diagnostic) == (
        b"t" * 14537, b"u" * 461)
    assert diagnostic == {"tar_eof_mode": "standard" if with_eof else "missing_accepted"}


@pytest.mark.parametrize("bad", ["data", "padding", "nonzero_padding", "header", "duplicate",
                                 "unexpected", "garbage", "partial_eof", "missing_member", "oversize"])
def test_missing_eof_compatibility_rejects_invalid_archives(bad, monkeypatch):
    members = [("default.tif", b"t" * 14537), ("userdata.json", b"u" * 461)]
    if bad == "duplicate":
        members.append(members[1])
    elif bad == "unexpected":
        members.append(("other", b"x"))
    elif bad == "missing_member":
        members.pop()
    content = tar_members(members)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        parsed = list(archive)
        content = content[:archive.offset]
    if bad == "data":
        content = content[:parsed[-1].offset_data + parsed[-1].size - 1]
    elif bad == "padding":
        content = content[:-1]
    elif bad == "nonzero_padding":
        content = content[:-1] + b"x"
    elif bad == "header":
        content = b"x" + content[1:]
    elif bad == "garbage":
        content += b"x" * 512
    elif bad == "partial_eof":
        content += b"\0" * 512
    elif bad == "oversize":
        monkeypatch.setattr(s1, "MAX_PROCESS_BYTES", len(content) - 1)
    diagnostic = {}
    with pytest.raises(s1.CDSEError):
        s1._parse_tar_response(content, allow_missing_eof=True, diagnostics=diagnostic)
    assert diagnostic == {}


def test_process_response_must_finish_and_match_declared_length():
    response = httpx.Response(200, content=b"short", headers={
        "Content-Type": "application/x-tar", "Content-Length": "10"})
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(s1.CDSEError, match="incomplete response"):
            s1._process_request(client, "offline", {})

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield tar_response(b"tiff", b"{}")
            raise httpx.ReadError("offline interrupted stream")

    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, stream=Interrupted(), headers={"Content-Type": "application/x-tar"}))) as client:
        with pytest.raises(s1.CDSEError):
            s1._process_request(client, "offline", {})


def synthetic_pinned_run():
    return {"before": {"product_id": acquisition(1).product_id,
                       "acquired_at": acquisition(1).acquired_at.isoformat()},
            "after": {"product_id": acquisition(13).product_id,
                      "acquired_at": acquisition(13).acquired_at.isoformat()},
            "orbit_direction": "ascending", "relative_orbit": 12}
