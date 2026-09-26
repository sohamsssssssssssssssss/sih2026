import numpy as np

from eval.sensor_necessity import evaluate_arrays, frozen_rule, reference_masks, score

SIZE = 100


def _case():
    """Left half water, right half land; top half under thick cloud."""
    occurrence = np.zeros((SIZE, SIZE), dtype=np.uint8)
    occurrence[:, : SIZE // 2] = 100
    scl = np.full((SIZE, SIZE), 4, dtype=np.uint8)
    scl[: SIZE // 2] = 9
    water = occurrence == 100
    cloud = scl == 9
    green = np.where(water, 0.08, 0.06).astype(np.float32)
    nir = np.where(water, 0.02, 0.30).astype(np.float32)
    green[cloud], nir[cloud] = 0.60, 0.62  # bright cloud: NDWI slightly negative everywhere
    ones = np.ones((SIZE, SIZE), dtype=np.float32)
    optical = np.stack([ones * 0.05, green, ones * 0.05, nir, ones])
    vv = np.where(water, 0.005, 0.2).astype(np.float32)
    vh = np.where(water, 0.001, 0.04).astype(np.float32)
    sar = np.stack([vv, vh, ones])
    return optical, sar, scl, occurrence


def test_cloud_blinds_optical_but_not_sar_or_cloud_gated_fusion():
    result = evaluate_arrays(*_case(), frozen_rule())

    obstructed = result["strata"]["optically_obstructed"]
    assert obstructed["optical_only"]["water_recall"] == 0.0
    assert obstructed["optical_only"]["balanced_accuracy"] == 0.5
    assert obstructed["sar_only"]["balanced_accuracy"] == 1.0
    assert obstructed["fusion_and_frozen"]["water_recall"] == 0.0
    assert result["strata"]["all"]["fusion_cloud_gated"]["balanced_accuracy"] == 1.0
    assert result["strata"]["optically_clear"]["optical_only"]["balanced_accuracy"] == 1.0
    assert result["obstructed_fraction_of_valid"] == 0.5


def test_excluded_column_band_leaves_every_stratum():
    result = evaluate_arrays(*_case(), frozen_rule(), exclude_columns=(0, SIZE // 2))

    for stratum in result["strata"].values():
        assert stratum["sar_only"]["reference_water_pixels"] == 0
        assert stratum["sar_only"]["evaluable"] is False


def test_reference_erosion_removes_boundary_and_transitional_pixels():
    occurrence = np.zeros((20, 20), dtype=np.uint8)
    occurrence[:, :10] = 100
    occurrence[:, 10] = 50  # transitional column
    occurrence[0, 15] = 255  # nodata

    water, land = reference_masks(occurrence, erosion=1)

    assert not (water & land).any()
    assert not water[:, 9:].any() and water[5, 8]
    assert not land[:, :12].any() and land[5, 12]
    assert not land[0:2, 14:17].any()


def test_small_strata_are_not_evaluable():
    predicted = np.ones((10, 10), dtype=bool)
    water = np.zeros((10, 10), dtype=bool)
    water[:5] = True

    result = score(predicted, water, ~water, np.ones((10, 10), dtype=bool))

    assert result["evaluable"] is False
    assert result["balanced_accuracy"] is None
    assert result["water_recall"] == 1.0 and result["land_specificity"] == 0.0
