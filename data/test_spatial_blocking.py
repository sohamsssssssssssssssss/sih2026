import numpy as np
import pytest

from data.spatial_blocking import assign_blocks, block_ids, leakage_audit, nearest_distances_m


def test_nearest_distance_matches_known_great_circle():
    # One degree of latitude is ~111.2 km on the mean-radius sphere.
    distance = nearest_distances_m(np.array([[10.0, 1.0]]), np.array([[10.0, 0.0], [50.0, 50.0]]))

    assert distance[0] == pytest.approx(111_195, rel=1e-3)


def test_nearby_points_share_a_block_and_distant_points_do_not():
    blocks = block_ids([10.0, 10.001, 12.0], [50.0, 50.001, 50.0], block_size_m=10_000)

    assert blocks[0] == blocks[1] != blocks[2]


def test_whole_blocks_go_to_one_split_and_fractions_are_approximate():
    rng = np.random.default_rng(0)
    blocks = [f"b{index}" for index in rng.integers(0, 200, size=2_000)]

    splits = assign_blocks(blocks, {"train": 0.7, "test": 0.3}, seed=1)

    per_block = {}
    for block, split in zip(blocks, splits):
        assert per_block.setdefault(block, split) == split
    assert splits.count("test") / len(splits) == pytest.approx(0.3, abs=0.03)
    assert splits == assign_blocks(blocks, {"train": 0.7, "test": 0.3}, seed=1)


def test_audit_flags_adjacent_cross_split_samples():
    lons, lats = [10.0, 10.005, 20.0], [50.0, 50.0, 50.0]

    report = leakage_audit(lons, lats, ["train", "test", "test"], radii_m=(1_000,))

    assert report["test"]["within_radius"]["1000"] == 1
    assert report["test"]["nearest_train_m"]["min"] < 400
