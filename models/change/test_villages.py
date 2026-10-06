"""Per-village flooded area on top of the Sentinel-1 water-change path."""

import json
from pathlib import Path

import pytest
from rasterio.warp import transform_geom

import models.change.villages as villages
from models.change import ChangeModel
from models.change.test_sar_water import UTM_GRID, evidence, flood_pair


def village(name: str, rows: tuple[int, int], cols: tuple[int, int]) -> dict:
    """A rectangular village in lon/lat, drawn on the 10 m UTM test grid by pixel edges."""
    grid = UTM_GRID["transform"]
    (top, bottom), (left, right) = rows, cols
    corners = [grid @ (col, row) for col, row in ((left, top), (right, top), (right, bottom), (left, bottom), (left, top))]
    polygon = {"type": "Polygon", "coordinates": [corners]}
    return {
        "type": "Feature",
        "properties": {
            "NAME": name, "TYPE": "Village", "SUB_DIST": "Kiratpur", "DISTRICT": "Darbhanga",
            "STATE": "Bihar", "CEN_2001": f"code-{name}",
        },
        "geometry": transform_geom(UTM_GRID["crs"], "EPSG:4326", polygon),
    }


@pytest.fixture
def boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def write(*features: dict) -> None:
        path = tmp_path / "villages.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": list(features)}))
        monkeypatch.setattr(villages, "VILLAGE_BOUNDARIES_PATH", path)

    return write


def test_flooded_village_is_named_with_its_flooded_hectares(tmp_path, boundaries) -> None:
    # The flood block is rows 10-19, cols 10-29 (200 pixels); Bhubhol contains it.
    boundaries(village("Bhubhol", (8, 23), (8, 32)), village("Dryville", (30, 38), (0, 10)))

    result = ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "Which villages flooded?")

    overlay = evidence(result, "village_flooding")
    assert overlay["status"] == "measured"
    assert overlay["villages_in_scene"] == 2
    [flooded] = overlay["flooded_villages"]
    assert flooded["name"] == "Bhubhol" and flooded["district"] == "Darbhanga"
    assert flooded["census_2001_code"] == "code-Bhubhol"
    assert flooded["flooded_ha"] == pytest.approx(2.0, rel=1e-3)
    assert flooded["area_in_scene_ha"] == pytest.approx(3.6, rel=1e-3)  # 15 x 24 pixels
    assert flooded["flooded_fraction"] == pytest.approx(2.0 / 3.6, rel=1e-3)
    assert flooded["observed_fraction"] == 1.0
    assert flooded["partially_outside_scene"] is False
    assert overlay["boundary_vintage"] == "Census 2001"
    assert "Bhubhol 2.0 ha" in result["answer"]


def test_village_crossing_the_scene_edge_is_flagged(tmp_path, boundaries) -> None:
    boundaries(village("Edgepur", (-5, 12), (10, 30)))  # starts above the scene's top edge

    overlay = evidence(ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "What changed?"), "village_flooding")

    [flooded] = overlay["flooded_villages"]
    assert flooded["partially_outside_scene"] is True
    assert flooded["flooded_ha"] == pytest.approx(0.4, rel=1e-3)  # rows 10-11 of the flood block


def test_dry_villages_are_counted_but_not_listed(tmp_path, boundaries) -> None:
    boundaries(village("Dryville", (30, 38), (0, 10)))

    result = ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "Which villages flooded?")

    overlay = evidence(result, "village_flooding")
    assert overlay["villages_in_scene"] == 1 and overlay["flooded_villages"] == []
    assert "No village" in result["answer"]


def test_scene_without_village_boundaries_says_villages_are_not_named(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(villages, "VILLAGE_BOUNDARIES_PATH", tmp_path / "missing.geojson")

    result = ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "Which villages flooded?")

    overlay = evidence(result, "village_flooding")
    assert overlay["status"] == "no_boundaries"
    assert overlay["villages_in_scene"] is None and overlay["flooded_villages"] == []
    assert "villages are not named" in result["answer"]


def test_nested_villages_each_keep_their_full_area(tmp_path, boundaries) -> None:
    # Community-digitised boundaries overlap; one shared label grid would hand
    # the inner village's pixels to whichever polygon was burned last.
    boundaries(village("Outer", (8, 23), (8, 32)), village("Inner", (12, 16), (12, 16)))

    overlay = evidence(ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "What changed?"), "village_flooding")

    flooded = {item["name"]: item for item in overlay["flooded_villages"]}
    assert flooded["Outer"]["flooded_ha"] == pytest.approx(2.0, rel=1e-3)
    assert flooded["Outer"]["area_in_scene_ha"] == pytest.approx(3.6, rel=1e-3)
    assert flooded["Inner"]["flooded_ha"] == pytest.approx(0.16, rel=1e-3)


def test_village_on_the_scene_edge_is_not_flagged_as_outside(tmp_path, boundaries) -> None:
    boundaries(village("Edgeline", (0, 12), (0, 40)))  # shares the scene's top, left and right edges

    overlay = evidence(ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "What changed?"), "village_flooding")

    assert overlay["flooded_villages"][0]["partially_outside_scene"] is False


def test_a_regenerated_boundary_file_is_picked_up(tmp_path, boundaries) -> None:
    boundaries(village("Before", (8, 23), (8, 32)))
    ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "What changed?")
    boundaries(village("After", (8, 23), (8, 32)))  # same path, new contents

    overlay = evidence(ChangeModel().infer(flood_pair(tmp_path, UTM_GRID), "What changed?"), "village_flooding")

    assert [item["name"] for item in overlay["flooded_villages"]] == ["After"]
