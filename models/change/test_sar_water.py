"""Sentinel-1 water-change path of the change baseline."""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import models.change.sar_water as sar_water
from models.change import ChangeModel

LAND_VV, WATER_VV = 0.1, 0.005  # linear gamma0: about -10 dB and -23 dB
SAR_BANDS = ("VV", "VH", "dataMask")
UTM_GRID = {"crs": "EPSG:32645", "transform": from_origin(500000, 2830000, 10, 10)}
WGS84_GRID = {"crs": "EPSG:4326", "transform": from_origin(85.0, 25.7, 0.0001, 0.0001)}


def sar_scene(water: np.ndarray, *, seed: int) -> np.ndarray:
    """VV/VH/dataMask stack with mild speckle around fixed land and water levels."""
    noise = np.random.default_rng(seed).uniform(0.9, 1.1, water.shape)
    vv = np.where(water, WATER_VV, LAND_VV) * noise
    return np.stack([vv, vv / 5, np.ones(water.shape)]).astype("float32")


def write_sar(path: Path, data: np.ndarray, *, crs, transform, descriptions=SAR_BANDS) -> str:
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
        count=data.shape[0], dtype="float32", crs=crs, transform=transform,
    ) as dataset:
        dataset.write(data)
        if descriptions:
            dataset.descriptions = descriptions
    return str(path)


def flood_pair(tmp_path: Path, grid: dict, *, speck: bool = False) -> list[str]:
    """T1 has a river along the top; T2 adds a 10x20 flooded block (200 pixels)."""
    before = np.zeros((40, 40), bool)
    before[:5] = True
    after = before.copy()
    after[10:20, 10:30] = True
    if speck:
        after[35, 35] = True  # one speckle pixel, below the minimum mapping unit
    return [
        write_sar(tmp_path / "t1.tif", sar_scene(before, seed=1), **grid),
        write_sar(tmp_path / "t2.tif", sar_scene(after, seed=2), **grid),
    ]


def evidence(result: dict, kind: str) -> dict:
    return next(item for item in result["evidence"] if item["type"] == kind)


def test_new_flood_block_is_measured_in_hectares_and_polygonized(tmp_path) -> None:
    result = ChangeModel().infer(flood_pair(tmp_path, UTM_GRID, speck=True), "Did flooding expand?")

    stats = evidence(result, "water_change_statistics")
    assert stats["status"] == "measured"
    # 200 pixels of 10 m; UTM scale factor near the central meridian keeps it within 0.1%.
    assert stats["new_water_ha"] == pytest.approx(2.0, rel=1e-3)
    assert stats["receded_water_ha"] == 0
    assert stats["water_t1_ha"] == pytest.approx(2.0, rel=1e-3)  # 5 x 40 river pixels
    assert -23 < stats["threshold_db"] < -10
    polygons = evidence(result, "water_change_polygons")
    features = polygons["geojson"]["features"]
    assert polygons["crs"] == "EPSG:4326"
    assert polygons["feature_count"] == len(features) == 1  # the speck was sieved out
    assert polygons["truncated"] is False
    assert features[0]["properties"] == {"change": "new_water", "pixels": 200}
    assert result["confidence"] is None
    assert "2.0 ha" in result["answer"]
    assert evidence(result, "temporal_inputs")["bands"] == list(SAR_BANDS)


def test_geographic_grid_gives_equal_area_hectares_and_real_coordinates(tmp_path) -> None:
    result = ChangeModel().infer(flood_pair(tmp_path, WGS84_GRID), "What changed?")

    radius = 6_371_007.2  # authalic sphere; the ellipsoidal answer differs by well under 1%
    north, south = math.radians(25.7 - 0.0010), math.radians(25.7 - 0.0020)
    expected_m2 = radius**2 * math.radians(0.0020) * (math.sin(north) - math.sin(south))
    stats = evidence(result, "water_change_statistics")
    assert stats["new_water_ha"] == pytest.approx(expected_m2 / 10_000, rel=1e-2)
    ring = evidence(result, "water_change_polygons")["geojson"]["features"][0]["geometry"][
        "coordinates"
    ][0]
    lons, lats = zip(*ring)
    assert min(lons) == pytest.approx(85.001) and max(lons) == pytest.approx(85.003)
    assert min(lats) == pytest.approx(25.698) and max(lats) == pytest.approx(25.699)
    extent = evidence(result, "change_extent")
    assert extent["coordinate_space"] == "normalized_xyxy"
    assert extent["coordinates"] == [0.25, 0.25, 0.75, 0.5]


def test_receding_water_is_reported_separately(tmp_path) -> None:
    paths = flood_pair(tmp_path, UTM_GRID)
    result = ChangeModel().infer(list(reversed(paths)), "Did the flood recede?")

    stats = evidence(result, "water_change_statistics")
    assert stats["new_water_ha"] == 0
    assert stats["receded_water_ha"] == pytest.approx(2.0, rel=1e-3)
    features = evidence(result, "water_change_polygons")["geojson"]["features"]
    assert [feature["properties"]["change"] for feature in features] == ["receded_water"]
    assert not any(item["type"] == "change_extent" for item in result["evidence"])


def test_scene_without_an_open_water_mode_abstains(tmp_path) -> None:
    dry = np.zeros((40, 40), bool)
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(dry, seed=1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", sar_scene(dry, seed=2), **UTM_GRID),
    ]

    result = ChangeModel().infer(paths, "Did flooding expand?")

    assert result["answer"].startswith("Abstained")
    assert result["confidence"] is None
    stats = evidence(result, "water_change_statistics")
    assert stats["status"] == "abstained"
    assert stats["reason_code"] == "NO_OPEN_WATER_MODE"
    assert stats["bimodal_tiles"] == 0 and stats["threshold_db"] is None
    assert stats["new_water_ha"] is None and stats["receded_water_ha"] is None
    assert not any(item["type"] == "water_change_polygons" for item in result["evidence"])


def test_largest_polygons_are_kept_when_features_are_capped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sar_water, "MAX_GEOJSON_FEATURES", 1)
    before = np.zeros((40, 40), bool)
    before[:5] = True
    after = before.copy()
    after[10:20, 10:30] = True  # 200 pixels
    after[30:35, 30:35] = True  # 25 pixels
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(before, seed=1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", sar_scene(after, seed=2), **UTM_GRID),
    ]

    polygons = evidence(ChangeModel().infer(paths, "What changed?"), "water_change_polygons")

    assert polygons["feature_count"] == 2
    assert polygons["truncated"] is True
    assert [f["properties"]["pixels"] for f in polygons["geojson"]["features"]] == [200]


def test_mixed_sar_and_optical_pair_is_rejected(tmp_path) -> None:
    water = np.zeros((40, 40), bool)
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(water, seed=1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", sar_scene(water, seed=2), **UTM_GRID, descriptions=None),
    ]

    with pytest.raises(ValueError, match="both be SAR"):
        ChangeModel().infer(paths, "What changed?")


def test_ungeoreferenced_sar_pair_is_rejected(tmp_path) -> None:
    water = np.zeros((40, 40), bool)
    grid = {"crs": None, "transform": from_origin(0, 40, 1, 1)}
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(water, seed=1), **grid),
        write_sar(tmp_path / "t2.tif", sar_scene(water, seed=2), **grid),
    ]

    with pytest.raises(ValueError, match="georeferenced"):
        ChangeModel().infer(paths, "What changed?")


def test_speckle_filter_never_fills_small_gaps_inside_change(tmp_path) -> None:
    before = np.zeros((40, 40), bool)
    before[:5] = True
    after = before.copy()
    after[10:20, 10:30] = True
    after[14, 15:17] = False  # a two-pixel dry gap inside the flooded block
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(before, seed=1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", sar_scene(after, seed=2), **UTM_GRID),
    ]

    result = ChangeModel().infer(paths, "What changed?")

    features = evidence(result, "water_change_polygons")["geojson"]["features"]
    assert [feature["properties"]["pixels"] for feature in features] == [198]
    assert evidence(result, "water_change_statistics")["new_water_ha"] == pytest.approx(1.98, rel=1e-3)


def test_one_extreme_pixel_cannot_capture_the_otsu_split() -> None:
    # Unclipped, a single -300 dB pixel wins the split on its own: the threshold
    # drops to about -298 dB and a real water/land scene reads as almost no water.
    rng = np.random.default_rng(0)
    values = np.concatenate((rng.normal(-17, 1, 5000), rng.normal(-12, 1, 5000)))

    shifted = sar_water.otsu_threshold(np.append(values, -300.0))

    assert shifted == pytest.approx(sar_water.otsu_threshold(values), abs=0.1)


def test_rotated_grid_is_rejected(tmp_path) -> None:
    water = np.zeros((40, 40), bool)
    grid = {"crs": "EPSG:32645", "transform": rasterio.Affine(0, 10, 500000, 10, 0, 2830000)}
    paths = [
        write_sar(tmp_path / "t1.tif", sar_scene(water, seed=1), **grid),
        write_sar(tmp_path / "t2.tif", sar_scene(water, seed=2), **grid),
    ]

    with pytest.raises(ValueError, match="north-up"):
        ChangeModel().infer(paths, "What changed?")


def test_small_flood_on_two_kinds_of_land_is_found_by_bimodal_tiles(tmp_path) -> None:
    # The Kosi failure: water is a few percent of the scene, so one scene-wide
    # Otsu split lands between two land classes and the old baseline abstained.
    rng = np.random.default_rng(3)
    land_db = np.where(np.arange(256)[None, :] < 128, -6.0, -11.0) * np.ones((256, 1))
    water = np.zeros((256, 256), bool)
    water[200:230, 20:60] = True  # 1,200 pixels, under 2% of the scene

    def scene(db: np.ndarray, seed: int) -> np.ndarray:
        vv = 10 ** ((db + np.random.default_rng(seed).normal(0, 0.5, db.shape)) / 10)
        return np.stack([vv, vv / 5, np.ones(db.shape)]).astype("float32")

    paths = [
        write_sar(tmp_path / "t1.tif", scene(land_db, 1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", scene(np.where(water, -20.0, land_db), 2), **UTM_GRID),
    ]
    assert sar_water.otsu_threshold(np.concatenate([land_db.ravel()] * 2)) > sar_water.WATER_DB_CEILING

    stats = evidence(ChangeModel().infer(paths, "Did flooding expand?"), "water_change_statistics")

    assert stats["status"] == "measured" and stats["bimodal_tiles"] >= 1
    assert -20 < stats["threshold_db"] < -11
    assert stats["new_water_ha"] == pytest.approx(12.0, rel=0.02)  # 1,200 pixels of 10 m
    low, high = stats["new_water_ha_sensitivity"]["threshold_minus_1db"], stats["new_water_ha_sensitivity"]["threshold_plus_1db"]
    assert low <= stats["new_water_ha"] <= high


def db_scene(db: np.ndarray, seed: int) -> np.ndarray:
    vv = 10 ** ((db + np.random.default_rng(seed).normal(0, 0.5, db.shape)) / 10)
    return np.stack([vv, vv / 5, np.ones(db.shape)]).astype("float32")


def test_two_dark_land_classes_do_not_pass_as_water(tmp_path) -> None:
    # Dark bare soil or sand next to dark vegetation: both classes below -15 dB,
    # clearly bimodal, but neither side looks like land, so this is not water/land.
    dark = np.where(np.arange(64)[None, :] < 32, -19.0, -16.0) * np.ones((64, 1))
    paths = [
        write_sar(tmp_path / "t1.tif", db_scene(dark, 1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", db_scene(dark, 2), **UTM_GRID),
    ]

    stats = evidence(ChangeModel().infer(paths, "Did flooding expand?"), "water_change_statistics")

    assert stats["status"] == "abstained" and stats["bimodal_tiles"] == 0


def test_water_in_the_edge_strip_past_the_last_full_tile_is_seen(tmp_path) -> None:
    land = np.full((100, 100), -8.0)
    flooded = land.copy()
    flooded[75:] = -20.0  # only in rows 64-99, beyond the first full 64 px tile
    paths = [
        write_sar(tmp_path / "t1.tif", db_scene(land, 1), **UTM_GRID),
        write_sar(tmp_path / "t2.tif", db_scene(flooded, 2), **UTM_GRID),
    ]

    stats = evidence(ChangeModel().infer(paths, "Did flooding expand?"), "water_change_statistics")

    assert stats["status"] == "measured"
    assert stats["new_water_ha"] == pytest.approx(25.0, rel=0.01)  # 25 rows x 100 pixels of 10 m
