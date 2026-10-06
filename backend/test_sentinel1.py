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


def geotiff(width: int, height: int, bounds, epsg: int, water_rows: int, dtype: str = "float32") -> bytes:
    """What the Process API returns: three float bands, no band descriptions."""
    vv = np.full((height, width), 0.1, dtype="float32")
    vv[:water_rows] = 0.005
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=width, height=height, count=3, dtype=dtype,
            crs=f"EPSG:{epsg}", transform=from_bounds(*bounds, width, height),
        ) as dataset:
            dataset.write(np.stack([vv, vv / 5, np.ones_like(vv)]).astype(dtype))
        return memory.read()


class FakeSentinelHub:
    """Token, paginated catalog and process endpoints, recording what was asked."""

    def __init__(self, before: list, after: list, *, shape_override=None, catalog_reply=None, dtype="float32") -> None:
        self.before, self.after, self.shape_override = before, after, shape_override
        self.catalog_reply, self.dtype = catalog_reply, dtype
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
            flooded = body["input"]["data"][0]["dataFilter"]["timeRange"]["from"] > "2024-08-10"
            return httpx.Response(
                200,
                content=geotiff(
                    width, height, body["input"]["bounds"]["bbox"], epsg, 30 if flooded else 5, self.dtype
                ),
            )
        return httpx.Response(404)


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


def test_aoi_beyond_the_process_api_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="2500"):
        s1.utm_grid((85.0, 25.0, 86.0, 26.0))


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
        s1.utm_grid(bbox)  # fits one Process API request
        assert before[0] < before[1] <= after[0] < after[1]
        assert event["sources"] and all(source["url"].startswith("https://") for source in event["sources"])
        assert event["aoi"]["derivation"]


def test_cli_event_supplies_bbox_inclusive_windows_and_pair_group(monkeypatch) -> None:
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    captured = {}

    def fake_fetch(client, client_id, client_secret, bbox, before, after, pair_group):
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
