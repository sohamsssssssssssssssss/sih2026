"""Execution-plan structure tests: representation, determinism, validation."""

from dataclasses import FrozenInstanceError

import pytest

from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    OPTICAL_SAR,
    SINGLE_IMAGE_VQA,
)
from orchestrator.execution_plan import (
    EXECUTION_PLAN_VERSION,
    ExecutionPlan,
    PlanStep,
    PlanValidationError,
    build_execution_plan,
    validate_plan,
)
from orchestrator.planner import (
    PLANNER_VERSION,
    Plan,
    PlanRequest,
    TEMPORAL_LOCALIZATION_RULE_ID,
    plan_request,
)


def build(
    question: str,
    *,
    scenes: tuple[str, ...] = ("scene_a",),
    capability: str | None = None,
):
    return build_execution_plan(
        plan_request(
            PlanRequest(
                question=question,
                scene_ids=scenes,
                requested_capability=capability,
            )
        )
    )


def test_ordinary_vqa_produces_one_step() -> None:
    execution = build("Is there a building in this image?")
    assert [step.capability for step in execution.steps] == [SINGLE_IMAGE_VQA]
    assert execution.steps[0].step_id == "step_1"
    assert execution.steps[0].depends_on == ()
    assert execution.executable is True


def test_grounding_produces_one_grounding_step() -> None:
    execution = build("Where is the building?")
    assert [step.capability for step in execution.steps] == [GROUNDING]
    assert execution.executable is False
    assert execution.unavailable_capabilities == (GROUNDING,)


def test_change_produces_one_change_vqa_step() -> None:
    execution = build("What changed between these images?", scenes=("a", "b"))
    assert [step.capability for step in execution.steps] == [CHANGE_VQA]
    assert execution.executable is False


def test_optical_sar_produces_one_optical_sar_step() -> None:
    execution = build("Compare the optical and SAR images.", scenes=("a", "b"))
    assert [step.capability for step in execution.steps] == [OPTICAL_SAR]
    assert execution.executable is False


def test_temporal_plus_localization_produces_change_then_grounding_chain() -> None:
    execution = build("Where did flooding increase between these two scenes?", scenes=("a", "b"))
    assert [(step.capability, step.depends_on) for step in execution.steps] == [
        (CHANGE_VQA, ()),
        (GROUNDING, ("step_1",)),
    ]
    assert execution.executable is False
    assert execution.unavailable_capabilities == (CHANGE_VQA, GROUNDING)
    assert execution.plan.rule_id == TEMPORAL_LOCALIZATION_RULE_ID


def test_single_intent_questions_stay_single_step() -> None:
    for question, capability in (
        ("What changed?", CHANGE_VQA),
        ("Where is the building?", GROUNDING),
        ("Is flooding visible?", SINGLE_IMAGE_VQA),
    ):
        execution = build(question)
        assert [step.capability for step in execution.steps] == [capability]


def test_step_ids_are_deterministic() -> None:
    execution = build("Where did flooding increase?", scenes=("a", "b"))
    assert [step.step_id for step in execution.steps] == ["step_1", "step_2"]


def test_dependencies_are_deterministic() -> None:
    execution = build("Where did flooding increase?", scenes=("a", "b"))
    assert execution.steps[0].depends_on == ()
    assert execution.steps[1].depends_on == ("step_1",)
    assert execution.steps[1].required_inputs == ("step_1.output",)


def test_same_request_produces_identical_execution_plan() -> None:
    request = PlanRequest(
        question="Where did flooding increase?",
        scene_ids=("scene_a", "scene_b"),
    )
    assert build_execution_plan(plan_request(request)) == build_execution_plan(
        plan_request(request)
    )


def test_execution_plan_version_is_present() -> None:
    execution = build("Is water visible?")
    assert execution.execution_plan_version == EXECUTION_PLAN_VERSION
    assert EXECUTION_PLAN_VERSION == "phase0-plan-v1"
    assert execution.plan.planner_version == PLANNER_VERSION


def _plan(
    *steps: PlanStep,
    rule_id: str = "default_single_image_vqa",
    selected: str = SINGLE_IMAGE_VQA,
) -> ExecutionPlan:
    base = plan_request(PlanRequest(question="Is water visible?", scene_ids=("a",)))
    decision = Plan(
        planner_version=base.planner_version,
        rule_id=rule_id,
        requested_capability=None,
        selected_capability=selected,
        executable=True,
        reason=base.reason,
        required_inputs=base.required_inputs,
        missing_inputs=(),
        provider_available=True,
        provider="qwen2.5vl-3b",
        unavailable_reason=None,
    )
    return ExecutionPlan(
        plan=decision,
        execution_plan_version=EXECUTION_PLAN_VERSION,
        steps=steps,
        unavailable_capabilities=(),
    )


def _step(step_id: str, capability: str, *depends_on: str) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        capability=capability,
        depends_on=depends_on,
        required_inputs=("single_scene",),
        provider_available=True,
        provider="qwen2.5vl-3b",
    )


def test_duplicate_step_ids_are_rejected() -> None:
    plan = _plan(_step("step_1", SINGLE_IMAGE_VQA), _step("step_1", SINGLE_IMAGE_VQA))
    with pytest.raises(PlanValidationError, match="Duplicate execution step id"):
        validate_plan(plan)


def test_unknown_dependency_is_rejected() -> None:
    plan = _plan(_step("step_1", SINGLE_IMAGE_VQA, "step_9"))
    with pytest.raises(PlanValidationError, match="Unknown dependency"):
        validate_plan(plan)


def test_self_dependency_is_rejected() -> None:
    plan = _plan(_step("step_1", SINGLE_IMAGE_VQA, "step_1"))
    with pytest.raises(PlanValidationError, match="depends on itself"):
        validate_plan(plan)


def test_dependency_cycles_are_rejected() -> None:
    plan = _plan(
        _step("step_1", CHANGE_VQA, "step_2"),
        _step("step_2", GROUNDING, "step_1"),
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(plan)


def test_dependency_ordering_is_enforced() -> None:
    plan = _plan(
        _step("step_1", GROUNDING, "step_2"),
        _step("step_2", CHANGE_VQA),
    )
    with pytest.raises(PlanValidationError, match="must precede"):
        validate_plan(plan)


def test_unknown_capability_in_step_is_rejected() -> None:
    plan = _plan(_step("step_1", "time_travel"))
    with pytest.raises(PlanValidationError, match="Unknown capability"):
        validate_plan(plan)


def test_duplicate_meaningless_steps_are_rejected() -> None:
    plan = _plan(_step("step_1", SINGLE_IMAGE_VQA), _step("step_2", SINGLE_IMAGE_VQA))
    with pytest.raises(PlanValidationError, match="Duplicate meaningless step"):
        validate_plan(plan)


def test_plan_structures_are_immutable() -> None:
    execution = build("Is water visible?")
    with pytest.raises(FrozenInstanceError):
        execution.execution_plan_version = "forged"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        execution.steps[0].capability = "forged"  # type: ignore[misc]
    assert isinstance(execution.steps, tuple)
    assert isinstance(execution.steps[0].depends_on, tuple)
