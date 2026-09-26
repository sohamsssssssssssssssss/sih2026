import json
import math
from pathlib import Path

import pytest

from eval.rsvqa_research import (
    allocate,
    mercator_scale,
    paired_comparison,
    rescore_prior,
    score,
    spatial_leakage_audit,
    stratified_subset,
    summarise,
    wilson_95,
)

NAME = "S2A_MSIL1C_20160720T105032_N0204_R051_T31UFU_20160720T105033_{col}-{row}.tif"


def write_split(root: Path, file_split: str, images: list[dict], questions: list[dict]) -> None:
    answers = [
        {"question_id": item["id"], "answer": item.pop("answer"), "active": True}
        for item in questions
    ]
    for kind, rows in (("images", images), ("questions", questions), ("answers", answers)):
        (root / f"LR_split_{file_split}_{kind}.json").write_text(json.dumps({kind: rows}))


def image(image_id: int, x: float, y: float) -> dict:
    return {
        "id": image_id,
        "active": True,
        "original_name": NAME.format(col=0, row=0),
        "upperleft_map_x": x,
        "upperleft_map_y": y,
    }


def question(question_id: int, image_id: int, kind: str, answer: str) -> dict:
    return {
        "id": question_id,
        "img_id": image_id,
        "type": kind,
        "question": f"q{question_id}",
        "answer": answer,
        "active": True,
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    # Near the equator one 3857 unit is ~one ground metre, so a 1280 m shift is half a patch.
    write_split(tmp_path, "train", [image(1, 0.0, 10000.0)], [question(1, 1, "count", "3")])
    write_split(
        tmp_path,
        "val",
        [image(2, 500000.0, 10000.0), image(3, 900000.0, 10000.0)],
        [question(10 + index, 2 + index % 2, kind, "yes") for index, kind in enumerate(
            ["presence"] * 6 + ["comp"] * 3 + ["rural_urban"]
        )],
    )
    write_split(tmp_path, "test", [image(4, 1280.0, 10000.0)], [question(20, 4, "presence", "no")])
    return tmp_path


def test_allocation_keeps_small_types_whole_and_splits_the_rest_proportionally():
    assert allocate({"a": 600, "b": 300, "small": 50}, 500, floor=100) == {
        "small": 50, "a": 300, "b": 150,
    }
    assert allocate({"a": 3, "b": 2}, 10, floor=1) == {"a": 3, "b": 2}


def test_subset_is_deterministic_stratified_and_honours_exclusions(dataset: Path):
    first = stratified_subset(dataset, "validation", 4, floor=1, seed=7, exclude_image_ids=frozenset({3}))
    again = stratified_subset(dataset, "validation", 4, floor=1, seed=7, exclude_image_ids=frozenset({3}))

    assert first == again
    assert {sample["image_id"] for sample in first["samples"]} == {2}
    # Eligible after exclusion: presence 3, comp 2 -> quotas 2.4 / 1.6 -> largest remainder 2 / 2.
    assert first["type_counts"] == {"comp": 2, "presence": 2}
    assert first["full_split_type_counts"] == {"comp": 3, "presence": 6, "rural_urban": 1}
    assert first["excluded_image_ids"] == [3]
    assert stratified_subset(dataset, "validation", 4, floor=1, seed=8)["samples"] != first["samples"]


def test_whole_split_subset_keeps_flagged_images(dataset: Path):
    locked = stratified_subset(dataset, "test", None, floor=1, seed=7, flagged_image_ids=frozenset({4}))

    assert [sample["sample_id"] for sample in locked["samples"]] == ["test-20"]
    assert locked["leakage_flagged_image_ids"] == [4]


@pytest.mark.parametrize(
    "prediction,gold,kind,strict,lenient,valid,count_bin",
    [
        ("Yes", "yes", "presence", True, True, True, None),
        ("Yes.", "yes", "presence", False, True, True, None),
        ("There is a road", "no", "comp", False, False, False, None),
        ("", "rural", "rural_urban", False, False, False, None),
        ("7", "9", "count", False, False, True, True),
        ("two", "2", "count", False, True, True, True),
        ("12", "9", "count", False, False, True, False),
        ("many", "0", "count", False, False, False, False),
    ],
)
def test_score_applies_each_metric_definition(prediction, gold, kind, strict, lenient, valid, count_bin):
    assert score(prediction, gold, kind) == {
        "strict": strict, "lenient": lenient, "valid": valid, "count_bin": count_bin,
    }


def test_summary_reports_per_type_uncertainty_and_type_weighting():
    rows = [
        {"type": "presence", "prediction": "yes", **score("yes", "yes", "presence")},
        {"type": "presence", "prediction": "no", **score("no", "yes", "presence")},
        {"type": "count", "prediction": "3", **score("3", "3", "count")},
    ]

    summary = summarise(rows, {"presence": 1, "count": 3})

    assert summary["strict_accuracy"] == pytest.approx(2 / 3)
    assert summary["per_type"]["presence"]["strict_accuracy"] == 0.5
    assert summary["per_type"]["count"]["count_bin_accuracy"] == 1.0
    assert summary["type_weighted_strict_accuracy"] == pytest.approx(0.5 * 0.25 + 1.0 * 0.75)
    assert summary["pred_yes_rate_on_binary"] == 0.5
    assert wilson_95(0, 0) is None
    low, high = wilson_95(50, 100)
    assert low < 0.5 < high and high - low == pytest.approx(0.1923, abs=1e-3)


def test_paired_comparison_uses_exact_mcnemar_on_discordant_pairs():
    base = [{"sample_id": str(index), "strict": False} for index in range(10)]
    adapted = [{"sample_id": str(index), "strict": True} for index in range(10)]

    result = paired_comparison(base, adapted)

    assert result["base_wrong_adapter_right"] == 10
    assert result["mcnemar_exact_p"] == pytest.approx(2 / 1024)
    with pytest.raises(ValueError):
        paired_comparison(base, adapted[:-1])


def test_spatial_audit_detects_half_patch_overlap(dataset: Path):
    audit = spatial_leakage_audit(dataset, verify_pixels=False)

    assert mercator_scale(0.0) == 1.0
    assert audit["test_vs_train"]["overlapping_image_ids"] == [4]
    assert audit["test_vs_train"]["max_overlap_fraction"] == pytest.approx(0.5, abs=1e-3)
    assert audit["test_vs_train"]["questions_on_overlapping_images"] == 1
    assert audit["validation_vs_train"]["overlapping_image_ids"] == []
    assert math.isclose(
        audit["test_vs_train"]["nearest_centre_distance_m"]["min"], 1280, rel_tol=1e-3
    )


def test_rescore_refuses_a_misaligned_historical_result(dataset: Path, tmp_path: Path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"results": [
        {"question": "different", "expected_answer": "no", "prediction": {"answer": "no"}}
    ]}))

    with pytest.raises(ValueError, match="misaligned"):
        rescore_prior(dataset, prior)
