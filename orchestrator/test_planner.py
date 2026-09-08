"""Deterministic request-planner tests; no model execution is required."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    OPTICAL_SAR,
    SINGLE_IMAGE_VQA,
    UnknownCapability,
)
from orchestrator.planner import (
    PLANNER_VERSION,
    InvalidPlanRequest,
    PlanRequest,
    plan_request,
)


def plan(
    question: str,
    *,
    scenes: tuple[str, ...] = ("scene_a",),
    sensor: str | None = None,
    capability: str | None = None,
):
    return plan_request(
        PlanRequest(
            question=question,
            scene_ids=scenes,
            sensor=sensor,
            requested_capability=capability,
        )
    )


@pytest.mark.parametrize(
    "question",
    [
        "Is there a building in this image?",
        "Describe what is visible.",
        "Is flooding visible?",
        "How many large buildings are visible?",
    ],
)
def test_ordinary_questions_default_to_single_image_vqa(question: str) -> None:
    result = plan(question)
    assert result.selected_capability == SINGLE_IMAGE_VQA
    assert result.rule_id == "default_single_image_vqa"
    assert result.executable is True
    assert result.required_inputs == ("single_scene",)


@pytest.mark.parametrize(
    "question",
    [
        "Where is the building?",
        "Locate the road.",
        "Highlight flooding.",
        "Give me the bounding box of the bridge.",
        "Find and localize the runway.",
        "Are buildings concentrated in the north?",
    ],
)
def test_localization_questions_select_grounding(question: str) -> None:
    result = plan(question)
    assert result.selected_capability == GROUNDING
    assert result.rule_id == "grounding_spatial_localization"
    assert result.provider_available is False
    assert result.provider is None
    assert result.executable is False


@pytest.mark.parametrize(
    "question",
    [
        "What changed between these images?",
        "Has built-up area increased?",
        "Compare the before-and-after images.",
        "Did flooding expand?",
        "What changed from 2025 to 2026?",
        "Has the image changed color?",
    ],
)
def test_temporal_questions_select_change_vqa(question: str) -> None:
    result = plan(question)
    assert result.selected_capability == CHANGE_VQA
    assert result.rule_id == "change_temporal_compare"
    assert result.missing_inputs == ("second_scene",)
    assert result.executable is False


@pytest.mark.parametrize(
    "question",
    [
        "Compare the optical and SAR images.",
        "Analyze Sentinel-1 and Sentinel-2 together.",
        "What does SAR show that the optical image misses?",
        "Use SAR/optical imagery together.",
    ],
)
def test_cross_modal_questions_select_optical_sar(question: str) -> None:
    result = plan(question)
    assert result.selected_capability == OPTICAL_SAR
    assert result.rule_id == "optical_sar_cross_modal"
    assert result.required_inputs == ("optical_scene", "sar_scene")
    assert result.missing_inputs == ("second_scene",)


def test_change_precedes_grounding_for_mixed_intent() -> None:
    result = plan("Where did flooding increase between these two scenes?")
    assert result.selected_capability == CHANGE_VQA


@pytest.mark.parametrize(
    ("capability", "question"),
    [
        (GROUNDING, "Is there a building?"),
        (SINGLE_IMAGE_VQA, "Where is the building?"),
    ],
)
def test_explicit_capability_overrides_inferred_intent(
    capability: str, question: str
) -> None:
    result = plan(question, capability=capability)
    assert result.selected_capability == capability
    assert result.requested_capability == capability
    assert result.rule_id == "explicit_capability"


def test_unknown_explicit_capability_fails() -> None:
    with pytest.raises(UnknownCapability):
        plan("Question", capability="time_travel")


def test_blank_question_fails() -> None:
    with pytest.raises(InvalidPlanRequest):
        plan("  \n ")


def test_pair_capabilities_report_missing_second_scene() -> None:
    for capability in (CHANGE_VQA, OPTICAL_SAR):
        result = plan("Question", capability=capability)
        assert result.missing_inputs == ("second_scene",)
        assert result.provider_available is False


def test_pair_capability_with_two_scenes_has_no_missing_inputs() -> None:
    result = plan(
        "What changed?",
        scenes=("scene_a", "scene_b"),
        capability=CHANGE_VQA,
    )
    assert result.missing_inputs == ()
    assert result.executable is False


def test_no_scene_is_reported_as_missing() -> None:
    result = plan("Is water visible?", scenes=())
    assert result.missing_inputs == ("scene",)
    assert result.executable is False


def test_sar_sensor_does_not_claim_single_image_ai_support() -> None:
    result = plan("Is water visible?", sensor="Sentinel-1")
    assert result.selected_capability == SINGLE_IMAGE_VQA
    assert result.provider_available is True
    assert result.executable is False
    assert result.unavailable_reason == (
        "Single-image SAR interpretation is not currently supported."
    )


def test_plain_sar_word_does_not_select_cross_modal_capability() -> None:
    result = plan("Is SAR visible in this screenshot?")
    assert result.selected_capability == SINGLE_IMAGE_VQA
    assert result.executable is True


@pytest.mark.parametrize(
    "question",
    [
        "Whereas roads are visible, is there water?",
        "Is there a change of land-cover type visible in this single image?",
        "Is this a radar-shaped object?",
        "Does the image show a bounding box already drawn?",
        "Is this before sunset?",
        "Is the north area more urban?",
        "Is it somewhere near water?",
    ],
)
def test_false_positive_phrases_remain_vqa(question: str) -> None:
    assert plan(question).selected_capability == SINGLE_IMAGE_VQA


def test_normalization_is_case_punctuation_and_whitespace_insensitive() -> None:
    expected = plan("Where is the building?")
    actual = plan("  WHERE---is the building!!!  ")
    assert actual == expected


def test_plan_has_stable_audit_metadata_and_reason() -> None:
    result = plan("Where is the building?")
    assert result.planner_version == PLANNER_VERSION
    assert result.rule_id == "grounding_spatial_localization"
    assert result.reason == "The request asks for spatial localization."


def test_planning_is_deterministic_and_immutable() -> None:
    request = PlanRequest("Is water visible?", ("scene_a",))
    first = plan_request(request)
    assert first == plan_request(request)
    with pytest.raises(FrozenInstanceError):
        first.provider = "forged"  # type: ignore[misc]
    assert isinstance(first.required_inputs, tuple)


def test_planning_invokes_no_model_and_writes_no_trace() -> None:
    with (
        patch(
            "models.qwen_vl.model.QwenVLModel.infer",
            side_effect=AssertionError("model must not run"),
        ) as infer,
        patch(
            "orchestrator.trace.append_record",
            side_effect=AssertionError("trace must not be written"),
        ) as append,
    ):
        result = plan("Is water visible?")
    assert result.executable is True
    infer.assert_not_called()
    append.assert_not_called()
