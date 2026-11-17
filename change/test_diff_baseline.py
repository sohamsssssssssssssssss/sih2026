import numpy as np

from change.diff_baseline import features


def test_identical_pair_has_zero_global_and_grid_features():
    image = np.random.default_rng(17).integers(0, 256, (13, 17, 3), dtype=np.uint8)
    np.testing.assert_array_equal(features(image, image), np.zeros(17 * 9))


def test_swapping_negates_signed_features_and_preserves_unsigned_features():
    before = np.zeros((8, 8, 3), dtype=np.uint8)
    after = before.copy()
    after[:2, :2, 0] = 21
    forward = features(before, after).reshape(17, 9)
    back = features(after, before).reshape(17, 9)
    np.testing.assert_array_equal(back[:, :6], forward[:, :6])
    np.testing.assert_array_equal(back[:, 6:], -forward[:, 6:])
    np.testing.assert_array_equal(forward[1], [21, 0, 0, 1, 0, 0, 21, 0, 0])
    assert not forward[2:].any()
