"""Sentinel-1 AOIs wider than one Process request: tiles rendered on one grid and mosaicked."""

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

import backend.sentinel1 as s1
import backend.services as services
from backend.test_sentinel1 import (  # noqa: F401  (runtime_dirs is a fixture)
    AFTER, BEFORE, PATNA_AOI, FakeSentinelHub, acquisition, geotiff, runtime_dirs,
)

# Stands in for the Process API's 2500 px, so the 42x46 px Patna grid needs 2x2
# tiles with uneven remainders (25+17 by 25+21).
TILE_PIXELS = 25
TILES_PER_DATE = 4
PATNA_BODY = {  # what a one-request AOI sent before tiling existed
    "input": {
        "bounds": {
            "bbox": [299140.0, 2832890.0, 299560.0, 2833350.0],
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32645"},
        },
        "data": [
            {
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": "2024-08-01T00:02:34Z", "to": "2024-08-01T00:22:34Z"},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "resolution": "HIGH",
                    "orbitDirection": "ASCENDING",
                },
                "processing": {
                    "backCoeff": "GAMMA0_TERRAIN",
                    "orthorectify": True,
                    "demInstance": "COPERNICUS_30",
                    "speckleFilter": {"type": "LEE", "windowSizeX": 5, "windowSizeY": 5},
                },
            }
        ],
    },
    "output": {
        "width": 42,
        "height": 46,
        "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
    },
    "evalscript": s1.EVALSCRIPT,
}


@pytest.fixture
def small_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s1, "MAX_SIDE_PIXELS", TILE_PIXELS)


class TaggedTileHub(FakeSentinelHub):
    """Request N has N water rows, and each date's last tile is off-swath (dataMask 0)."""

    def tiff(self, body: dict, width: int, height: int, epsg: int) -> bytes:
        count = len(self.process_bodies)
        valid = 0.0 if count % TILES_PER_DATE == 0 else 1.0
        return geotiff(width, height, body["input"]["bounds"]["bbox"], epsg, count, self.dtype, valid)


class ShiftedTileHub(FakeSentinelHub):
    """The post-event date's last tile comes back one pixel east of where it was asked for."""

    def tiff(self, body: dict, width: int, height: int, epsg: int) -> bytes:
        left, bottom, right, top = body["input"]["bounds"]["bbox"]
        shift = s1.PIXEL_SIZE_M if len(self.process_bodies) == 2 * TILES_PER_DATE else 0
        return geotiff(width, height, (left + shift, bottom, right + shift, top), epsg, 5, self.dtype)


def fetch(hub: FakeSentinelHub) -> dict:
    with httpx.Client(transport=httpx.MockTransport(hub)) as client:
        return s1.fetch_pair(client, "id", "secret", PATNA_AOI, BEFORE, AFTER)


def place(grid: s1.Grid, bbox: list[float]) -> tuple[slice, slice]:
    """The rows and columns of the global grid that a tile's bbox covers, in whole pixels."""
    left, bottom, right, top = bbox
    west, north = grid.bounds[0], grid.bounds[3]
    edges = [(north - top), (north - bottom), (left - west), (right - west)]
    pixels = [edge / s1.PIXEL_SIZE_M for edge in edges]
    assert all(value == int(value) for value in pixels), "tile is off the global pixel lattice"
    first_row, end_row, first_col, end_col = map(int, pixels)
    return slice(first_row, end_row), slice(first_col, end_col)


def test_small_aoi_is_one_request_per_date_with_an_unchanged_body(runtime_dirs) -> None:
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)])

    fetch(hub)

    assert len(hub.process_bodies) == 2
    assert hub.process_bodies[0] == PATNA_BODY


def test_two_by_two_aoi_is_four_requests_that_tile_the_grid_exactly(runtime_dirs, small_tiles) -> None:
    hub = FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)])

    fetched = fetch(hub)

    grid = s1.utm_grid(PATNA_AOI)
    assert (fetched["width"], fetched["height"]) == (grid.width, grid.height)
    assert len(hub.process_bodies) == 2 * TILES_PER_DATE
    for date in (hub.process_bodies[:TILES_PER_DATE], hub.process_bodies[TILES_PER_DATE:]):
        cover = np.zeros((grid.height, grid.width), int)
        for body in date:
            rows, cols = place(grid, body["input"]["bounds"]["bbox"])
            size = (body["output"]["width"], body["output"]["height"])
            assert size == (cols.stop - cols.start, rows.stop - rows.start)
            assert max(size) <= TILE_PIXELS
            assert body["input"]["data"] == date[0]["input"]["data"]
            assert body["evalscript"] == s1.EVALSCRIPT
            cover[rows, cols] += 1
        assert (cover == 1).all()  # no gap, no overlap
        assert sum(body["output"]["width"] * body["output"]["height"] for body in date) == cover.size


def test_mosaic_is_on_the_global_grid_with_every_tile_in_its_place(runtime_dirs, small_tiles) -> None:
    hub = TaggedTileHub(before=[acquisition(1)], after=[acquisition(13)])

    fetched = fetch(hub)

    grid = s1.utm_grid(PATNA_AOI)
    expected_vv = np.full((grid.height, grid.width), 0.1, "float32")
    expected_mask = np.ones((grid.height, grid.width), "float32")
    for count, body in enumerate(hub.process_bodies[TILES_PER_DATE:], start=TILES_PER_DATE + 1):
        rows, cols = place(grid, body["input"]["bounds"]["bbox"])
        expected_vv[rows.start:rows.start + count, cols] = 0.005
        if count % TILES_PER_DATE == 0:
            expected_mask[rows, cols] = 0
    with rasterio.open(services.INGESTED_RASTER_DIR / f"{fetched['after']['scene_id']}.tif") as raster:
        assert raster.crs.to_epsg() == grid.epsg
        assert (raster.width, raster.height) == (grid.width, grid.height)
        assert raster.transform == from_bounds(*grid.bounds, grid.width, grid.height)
        assert raster.descriptions == ("VV", "VH", "dataMask")
        vv, vh, mask = raster.read()
    np.testing.assert_array_equal(vv, expected_vv)
    np.testing.assert_array_equal(vh, expected_vv / 5)
    np.testing.assert_array_equal(mask, expected_mask)
    assert 0 < expected_mask.mean() < 1
    assert fetched["after"]["valid_fraction"] == pytest.approx(expected_mask.mean())  # over the whole mosaic


def test_one_misaligned_tile_fails_the_whole_fetch_and_nothing_is_ingested(runtime_dirs, small_tiles) -> None:
    hub = ShiftedTileHub(before=[acquisition(1)], after=[acquisition(13)])

    with pytest.raises(s1.CDSEError, match="grid"):
        fetch(hub)

    assert len(hub.process_bodies) == 2 * TILES_PER_DATE
    assert not services.SCENE_MANIFEST_DIR.exists()
    assert not services.INGESTED_RASTER_DIR.exists()


def test_aoi_over_the_mosaic_cap_is_rejected_before_any_request() -> None:
    assert s1.MAX_MOSAIC_SIDE_PIXELS**2 <= services.MAX_RASTER_PIXELS
    assert s1.utm_grid((86.2, 25.92, 86.5, 26.07)).width > s1.MAX_SIDE_PIXELS  # 30 km: tiled, not rejected
    transport = httpx.MockTransport(lambda request: pytest.fail("no request may be sent"))

    with httpx.Client(transport=transport) as client, pytest.raises(
        ValueError, match=f"at most {s1.MAX_MOSAIC_SIDE_PIXELS} per side"
    ):
        s1.fetch_pair(client, "id", "secret", (85.0, 25.0, 85.6, 25.6), BEFORE, AFTER)  # about 60 km
