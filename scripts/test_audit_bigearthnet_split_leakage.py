import json

from scripts.audit_bigearthnet_split_leakage import (
    TILE_ORIGINS_PATH,
    patch_bounds,
    same_footprint_across_splits,
    verify_against_reference_maps,
)

ORIGINS = json.loads(TILE_ORIGINS_PATH.read_text())["origins"]


def test_footprints_from_patch_ids_match_every_tracked_reference_map():
    result = verify_against_reference_maps(ORIGINS)

    assert result["checked"] == 100
    assert result["mismatches"] == []


def test_x_index_moves_east_and_y_index_moves_south():
    epsg, bounds = patch_bounds("S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_59_69", ORIGINS)

    assert epsg == 32633
    assert bounds == [370800.0, 5316000.0, 372000.0, 5317200.0]


def test_same_footprint_in_two_splits_is_reported():
    ids = ["S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_1_2",
           "S2B_MSIL2A_20180101T101031_N9999_R022_T33UUP_1_2",
           "S2B_MSIL2A_20180101T101031_N9999_R022_T33UUP_1_3"]

    result = same_footprint_across_splits(ids, ["train", "test", "test"])

    assert result == {"unique_footprints": 2, "footprints_in_multiple_splits": {"test+train": 1}}
