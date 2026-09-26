import math

import pytest

from eval.abstention import (
    aurc,
    expected_calibration_error,
    grounding_records,
    risk_coverage,
    selective_accuracy_at,
    summarize,
)


def test_risk_coverage_and_aurc_on_hand_computed_example():
    records = [(0.9, True), (0.8, True), (0.7, False), (0.6, False)]

    curve = risk_coverage(records)

    assert [p["coverage"] for p in curve] == [0.25, 0.5, 0.75, 1.0]
    assert [p["risk"] for p in curve] == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert aurc(curve) == pytest.approx(0.25 * (0 + 0 + 1 / 3 + 0.5))


def test_tied_scores_enter_the_curve_together():
    curve = risk_coverage([(0.9, True), (0.9, False), (0.1, True)])

    assert [p["n_selected"] for p in curve] == [2, 3]
    assert curve[0]["risk"] == pytest.approx(0.5)


def test_abstentions_count_in_denominator_and_cap_coverage():
    summary = summarize([(0.9, True), (None, False), (None, False), (0.5, False)])

    assert summary["n_abstained"] == 2
    assert summary["max_coverage"] == 0.5
    assert summary["full_coverage_accuracy"] == 0.25
    assert summary["selective_accuracy"]["0.25"]["selective_accuracy"] == 1.0
    assert summary["selective_accuracy"]["1.0"]["coverage"] == 0.5


def test_selective_accuracy_returns_none_below_first_operating_point():
    curve = risk_coverage([(0.9, True), (0.9, True)])

    assert selective_accuracy_at(curve, 0.25) is None


def test_ece_is_zero_when_bin_accuracy_matches_score():
    records = [(0.8, True)] * 8 + [(0.8, False)] * 2

    assert expected_calibration_error(records)["ece"] == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    records = [(0.9, True)] * 5 + [(0.9, False)] * 5

    assert expected_calibration_error(records)["ece"] == pytest.approx(0.4)


def test_ece_puts_score_one_in_last_bin():
    result = expected_calibration_error([(1.0, True)], n_bins=10)

    assert result["bins"][0]["lower"] == 0.9


@pytest.mark.parametrize(
    "records",
    [[(math.nan, True)], [(0.5, 1)], [(True, True)], []],
)
def test_invalid_records_are_rejected(records):
    with pytest.raises(ValueError):
        risk_coverage(records)


def test_ece_rejects_scores_outside_unit_interval():
    with pytest.raises(ValueError):
        expected_calibration_error([(1.5, True)])


def test_grounding_adapter_treats_missing_box_as_abstention():
    results = [
        {"selected_prediction": {"confidence": 0.7}, "iou": 0.6},
        {"selected_prediction": {"confidence": 0.4}, "iou": 0.2},
        {"selected_prediction": None, "iou": 0.0},
    ]

    assert grounding_records(results) == [(0.7, True), (0.4, False), (None, False)]
