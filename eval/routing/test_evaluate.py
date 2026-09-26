import pytest

from eval.routing.evaluate import (
    PAIR_CASES,
    Harness,
    analyze_label,
    load_queries,
    plan_label,
    reason_from_error,
    request_body,
    score,
    summarize_pairs,
)

PLAN_OK = {"missing_inputs": [], "rule_id": "default_single_image_vqa",
           "unavailable_reason": None, "selected_capability": "single_image_vqa"}


@pytest.mark.parametrize(
    ("detail", "reason"),
    [
        ([{"loc": ["body", "question"], "type": "string_too_short"}], "empty_question"),
        ([{"loc": ["body", "question"], "type": "string_too_long"}], "question_too_long"),
        ([{"loc": ["body", "scene_id"], "type": "missing"}], "missing_scene"),
        ([{"loc": ["body", "execution_mode"], "type": "literal_error"}], "invalid_execution_mode"),
        ({"eligible": False, "reason_codes": ["crs_missing"]}, "pair_incompatible"),
        ("Unknown capability: segmentation", "unknown_capability"),
        ("A non-empty question is required.", "empty_question"),
        ("This request requires two scenes.", "missing_second_scene"),
        ("Local scene pixels are unavailable.", "scene_not_found"),
        ("something new", "rejected_other"),
    ],
)
def test_reason_from_error_maps_api_details_to_label_vocabulary(detail, reason):
    assert reason_from_error(detail) == reason


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({}, "single_image_vqa"),
        ({"missing_inputs": ["second_scene"]}, "reject:missing_second_scene"),
        ({"rule_id": "temporal_change_then_grounding", "selected_capability": "change_vqa"},
         "reject:multi_step_not_executable"),
        ({"unavailable_reason": "Single-image SAR interpretation is not currently supported."},
         "reject:single_image_sar_unsupported"),
        ({"unavailable_reason": "Required dependency groundingdino is unavailable.",
          "selected_capability": "grounding"}, "grounding"),
    ],
)
def test_plan_label_separates_routing_from_readiness(overrides, label):
    assert plan_label(200, {**PLAN_OK, **overrides}) == label


def test_analyze_label_uses_plan_reason_for_generic_503():
    body = {"detail": "Required capability is not currently available."}

    assert analyze_label(503, body, "reject:multi_step_not_executable") == "reject:multi_step_not_executable"
    assert analyze_label(503, body, "change_vqa") == "reject:capability_unavailable"
    assert analyze_label(503, {"detail": {"reason_code": "CUDA_UNAVAILABLE"}}, "grounding") == "unavailable:CUDA_UNAVAILABLE"
    assert analyze_label(200, {"trace": {"params": {"capability": "grounding"}}}, "grounding") == "dispatched:grounding"


def test_score_counts_wrong_dispatches_for_system_and_gateless_baseline():
    unsupported = {"id": "x", "category": "unsupported", "variant": "canonical",
                   "expected": {"outcome": "reject", "reason": "unsupported_request"},
                   "_selected": "single_image_vqa"}
    refused = score(unsupported, "reject:multi_step_not_executable", "reject:multi_step_not_executable")
    executed = score(unsupported, "single_image_vqa", "dispatched:single_image_vqa")

    assert refused["correct"] and not refused["wrong_dispatch"] and refused["baseline_wrong_dispatch"]
    assert not refused["reason_correct"]
    assert not executed["correct"] and executed["wrong_dispatch"]


def test_ambiguous_item_accepts_refusal_or_licensed_capability():
    item = {"id": "a", "category": "ambiguous", "variant": "canonical",
            "expected": {"outcome": "clarify", "acceptable": ["grounding"]}}

    assert score(item, "grounding", "dispatched:grounding")["correct"]
    assert score(item, "single_image_vqa", "reject:scene_not_found")["correct"]
    assert not score(item, "single_image_vqa", "dispatched:single_image_vqa")["correct"]


def test_labelled_query_file_is_well_formed():
    queries = load_queries()

    assert len(queries) >= 150
    assert len({item["id"] for item in queries}) == len(queries)
    assert {item["expected"]["outcome"] for item in queries} == {"route", "reject", "clarify"}


def test_harness_dispatches_through_real_api_with_stubbed_providers():
    synthetic = [
        {"question": "How many trees are there?", "scenes": "single"},
        {"question": "What changed between these two images?", "scenes": "temporal_pair"},
        {"question": "Is there water?", "scenes": "single", "capability": "change_vqa"},
    ]
    with Harness("stub_ready").active() as harness:
        scenes = harness.standard_scenes()
        statuses = [harness.client.post("/api/analyze", json=request_body(item, scenes)).status_code
                    for item in synthetic]

    assert statuses == [200, 200, 422]
    assert harness.dispatched == ["single_image_vqa", "change_vqa"]


def test_pair_suite_fixture_cases_cover_positive_and_negative_workflows():
    rows = [{"expected_code": code, "dispatched": code is None, "passed": True}
            for _, _, code, _, _ in PAIR_CASES]

    summary = summarize_pairs(rows)

    assert summary["valid_pair_acceptance"]["n"] == 2
    assert summary["invalid_pair_rejection"]["rate"] == 1.0
