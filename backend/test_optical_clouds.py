"""Sentinel-2 cloud check beside a Sentinel-1 flood fetch, against a mocked Sentinel Hub."""

import json
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import backend.optical_clouds as optical
import backend.sentinel1 as s1
import backend.services as services
from backend.test_sentinel1 import (  # noqa: F401  (runtime_dirs is a fixture)
    AFTER, BEFORE, PATNA_AOI, FakeSentinelHub, acquisition, runtime_dirs,
)

GRID = s1.utm_grid(PATNA_AOI)
POST_EVENT = acquisition(13).acquired_at  # 2024-08-13T00:12:34Z
# Two no-data rows; cloud shadow, high-probability cloud and cirrus rows (obscured);
# a low-probability cloud row (not obscured). The rest is vegetation.
CLOUDY_ROWS = {0: 0, 1: 0, 2: 3, 3: 9, 4: 10, 5: 7}
CLEAR_ROWS = {0: 8}  # one medium-probability cloud row


def s2_feature(day: int, hour: int) -> dict:
    moment = datetime(2024, 8, day, hour, tzinfo=timezone.utc)
    return {
        "id": f"S2B_MSIL2A_202408{day:02d}T{hour:02d}0000",
        "properties": {"datetime": moment.isoformat().replace("+00:00", "Z")},
    }


class FakeSentinel2:
    """Sentinel-2 catalog and SCL Process replies, routed to by FakeSentinelHub."""

    def __init__(self, features: list, scl_rows: dict[int, int] | None = None, status: int = 200) -> None:
        self.features, self.scl_rows, self.status = features, scl_rows or {}, status
        self.process_bodies: list[dict] = []

    def __call__(self, url: str, body: dict) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status, text="optical service unavailable")
        if url == s1.CATALOG_URL:
            return httpx.Response(200, json={"features": self.features, "context": {}})
        self.process_bodies.append(body)
        width, height = body["output"]["width"], body["output"]["height"]
        scl = np.full((height, width), 4, dtype="uint8")
        for row, value in self.scl_rows.items():
            scl[row] = value
        epsg = int(body["input"]["bounds"]["properties"]["crs"].rsplit("/", 1)[1])
        with MemoryFile() as memory:
            with memory.open(
                driver="GTiff", width=width, height=height, count=1, dtype="uint8", crs=f"EPSG:{epsg}",
                transform=from_bounds(*body["input"]["bounds"]["bbox"], width, height),
            ) as dataset:
                dataset.write(scl[np.newaxis])
            return httpx.Response(200, content=memory.read())


def check(s2: FakeSentinel2) -> dict:
    hub = FakeSentinelHub(before=[], after=[], optical=s2)
    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        return optical.check_optical_clouds(client, "token", GRID, PATNA_AOI, POST_EVENT)


def manifest(scene_id: str) -> dict:
    return json.loads((services.SCENE_MANIFEST_DIR / f"{scene_id}.json").read_text())


def test_cloudy_scl_is_cloudy_and_recorded_on_the_post_event_scene(runtime_dirs) -> None:
    s2 = FakeSentinel2([s2_feature(11, 5), s2_feature(14, 5)], CLOUDY_ROWS)
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)], optical=s2)

    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        fetched = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    result = fetched["optical_check"]
    request = s2.process_bodies[0]
    height = request["output"]["height"]
    assert result["verdict"] == "cloudy"
    assert result["sentinel2_product_id"] == "S2B_MSIL2A_20240814T050000"  # closer than the 11th
    assert result["acquired_at"] == "2024-08-14T05:00:00Z"
    assert result["aoi_cloud_fraction"] == pytest.approx(3 / (height - 2))
    assert result["aoi_valid_fraction"] == pytest.approx((height - 2) / height)
    assert result["cloudy_threshold"] == optical.CLOUDY_THRESHOLD
    assert request["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert request["input"]["bounds"]["bbox"] == list(GRID.bounds)
    time_range = request["input"]["data"][0]["dataFilter"]["timeRange"]
    assert time_range["from"] < "2024-08-14T05:00:00Z" < time_range["to"]
    assert manifest(fetched["after"]["scene_id"])["optical_check"] == result
    assert "optical_check" not in manifest(fetched["before"]["scene_id"])


def test_clear_scl_is_clear() -> None:
    s2 = FakeSentinel2([s2_feature(14, 5)], CLEAR_ROWS)

    result = check(s2)

    height = s2.process_bodies[0]["output"]["height"]
    assert result["verdict"] == "clear"
    assert result["aoi_cloud_fraction"] == pytest.approx(1 / height)
    assert result["aoi_valid_fraction"] == 1.0


def test_no_sentinel2_acquisition_reports_nulls() -> None:
    s2 = FakeSentinel2([])

    result = check(s2)

    assert result["verdict"] == "no_acquisition"
    assert result["reason"]
    assert s2.process_bodies == []
    assert all(
        result[key] is None
        for key in ("sentinel2_product_id", "acquired_at", "aoi_cloud_fraction", "aoi_valid_fraction")
    )


def test_optical_http_failure_never_fails_the_sar_fetch(runtime_dirs) -> None:
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)], optical=FakeSentinel2([], status=503))

    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        fetched = s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)

    result = fetched["optical_check"]
    assert result["verdict"] == "unavailable" and "503" in result["reason"]
    assert result["aoi_cloud_fraction"] is None and result["aoi_valid_fraction"] is None
    assert manifest(fetched["after"]["scene_id"])["optical_check"] == result
    assert manifest(fetched["before"]["scene_id"])["identity"]["modality"] == "sar"
