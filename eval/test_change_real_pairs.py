from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from eval.change_real_pairs import (
    api_run,
    classify,
    isolated_runtime,
    phase_correlation_shift,
    provider_rule,
    reference_box_stats,
    scl_conditioning,
)
from models.change import ChangeModel

BANDS = ("B02", "B03", "B04", "B08", "dataMask")


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=5,
        dtype="float32", crs="EPSG:32643", transform=from_origin(752740, 3122200, 10, 10),
    ) as dataset:
        dataset.write(data.astype("float32"))
        for index, name in enumerate(BANDS, 1):
            dataset.set_band_description(index, name)
    return path


def _pair(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.uniform(0.02, 0.4, size=(5, 12, 10))
    first[4] = 1
    second = first.copy()
    second[3, 2:6, 3:8] *= 3.0  # NIR jump: vegetation-like change in a block
    second[4, 0, :] = 0  # one invalid row
    return first, second


def test_replica_matches_provider_pixel_counts(tmp_path):
    first, second = _pair()
    paths = [str(_write(tmp_path / "t1.tif", first)), str(_write(tmp_path / "t2.tif", second))]

    evidence = ChangeModel().infer(paths, "q")["evidence"]
    rule = provider_rule(first.astype("float32"), second.astype("float32"))

    assert int(rule["changed"].sum()) == evidence[1]["changed_pixels"] > 0
    assert int(rule["valid"].sum()) == evidence[2]["valid_pixels"] == 110


def test_phase_correlation_recovers_known_shift_and_sign():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(64, 64))
    second = np.roll(first, shift=(-2, 3), axis=(0, 1))  # T2(x) = T1(x - d), d = (dx=3, dy=-2)

    shift = phase_correlation_shift(first, second, np.ones_like(first, dtype=bool))
    unrelated = phase_correlation_shift(first, rng.normal(size=(64, 64)), np.ones_like(first, dtype=bool))

    assert shift["dx"] == pytest.approx(3, abs=0.1)
    assert shift["dy"] == pytest.approx(-2, abs=0.1)
    assert shift["peak"] > 5 * unrelated["peak"]


def test_reference_box_and_scl_conditioning_rates():
    valid = np.ones((10, 10), dtype=bool)
    changed = np.zeros_like(valid)
    changed[0:5, 0:5] = True  # all change inside the box
    changed[9, 9] = True  # one detection outside
    rule = {"valid": valid, "changed": changed}
    scl1 = np.full((10, 10), 4)
    scl2 = scl1.copy()
    scl2[9, :] = 9  # cloud row in T2
    scl2[0, 0:5] = 6  # water appears on 5 changed pixels
    scl2[5, 0:5] = 6  # and on 5 unchanged pixels

    box = reference_box_stats(rule, [[0.0, 0.0, 0.5, 0.3], [0.0, 0.2, 0.5, 0.5]])  # overlapping union
    scl = scl_conditioning(rule, scl1, scl2)

    assert scl["scl_water_transition_fraction_of_valid"] == pytest.approx(0.1)
    assert scl["changed_fraction_on_scl_water_transition"] == pytest.approx(0.5)
    assert box["changed_fraction_inside"] == 1.0
    assert box["changed_fraction_outside"] == pytest.approx(1 / 75)
    assert box["share_of_changed_inside"] == pytest.approx(25 / 26)
    assert scl["cloud_any_fraction_of_valid"] == pytest.approx(0.1)
    assert scl["changed_fraction_on_cloud_any"] == pytest.approx(0.1)
    assert scl["share_of_changed_in_cloud_or_shadow"] == pytest.approx(1 / 26)


def _accepted(category="urban_construction", valid=1.0, clear=0.99, cloud=0.0):
    return {
        "category": category, "expected_outcome": "real_surface_change",
        "api": {"status_code": 200, "evidence": [{}, {}, {"valid_fraction": valid}]},
        "diagnostics": {"scl": {"clear_both_fraction_of_valid": clear, "cloud_any_fraction_of_valid": cloud}},
    }


def _rejection(status_code, codes):
    return {"category": "rejection", "expected_outcome": "gate_rejection",
            "expected_reason_code": "grid_alignment_incompatible",
            "api": {"status_code": status_code, "gate": {"reason_codes": codes}}}


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (_accepted(), "PASSED"),
        (_accepted(clear=0.5), "INCONCLUSIVE"),
        (_accepted(valid=0.2), "INCONCLUSIVE"),
        (_accepted(category="control_cloud_contamination", clear=0.0, cloud=1.0), "PASSED"),
        (_accepted(category="control_cloud_contamination", cloud=0.0), "INCONCLUSIVE"),
        (_rejection(422, ["grid_alignment_incompatible"]), "PASSED"),
        (_rejection(422, ["reprojection_required"]), "FAILED"),
        ({**_rejection(200, None), "api": {"status_code": 200}}, "FAILED"),
        ({"category": "water_change_loss", "expected_outcome": "real_surface_change",
          "status": "invalidated", "invalidated": {"reason": "acquisition defect"}}, "FAILED"),
    ],
)
def test_classify_reports_evidence_validity_not_baseline_correctness(result, status):
    assert classify(result)[0] == status


def test_api_run_executes_ordered_pair_and_gate_rejects_reversed(tmp_path):
    from backend.main import app

    first, second = _pair()
    paths = [_write(tmp_path / "t1.tif", first), _write(tmp_path / "t2.tif", second)]
    observations = [
        {"acquisition_time": "2021-03-05T05:41:02Z", "collection": "c", "stac_item_id": "a"},
        {"acquisition_time": "2025-03-04T05:41:06Z", "collection": "c", "stac_item_id": "b"},
    ]
    with isolated_runtime(tmp_path / "runtime") as trace_store:
        client = TestClient(app)
        accepted = api_run(client, "pair", observations, paths, trace_store)
        rejected = api_run(client, "pair-r", observations[::-1], paths[::-1], trace_store)

    assert accepted["status_code"] == 200
    assert accepted["model"]["name"] == "change-deterministic"
    assert accepted["confidence_present"] is False
    assert accepted["trace_chain_valid"] is True
    assert rejected["status_code"] == 422
    assert "acquisition_order_invalid" in rejected["gate"]["reason_codes"]
