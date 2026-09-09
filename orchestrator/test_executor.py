"""Executor tests: delegation, atomicity, and no partial execution."""

from typing import Any

import pytest

from orchestrator.capabilities import (
    SINGLE_IMAGE_VQA,
    CapabilityUnavailable,
)
from orchestrator.execution_plan import (
    EXECUTION_PLAN_VERSION,
    ExecutionPlan,
    PlanStep,
    PlanValidationError,
    build_execution_plan,
)
from orchestrator.executor import execute_plan
from orchestrator.planner import PlanRequest, plan_request


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


def call(
    execution,
    *,
    route_fn,
    question: str = "Is there a building in this image?",
) -> dict[str, Any]:
    return execute_plan(
        execution,
        route_fn=route_fn,
        image_paths=["/tmp/scene.png"],
        question=question,
        base_params={"scene_id": "scene_a", "sensor": None, "execution_mode": "live"},
    )


def test_one_step_vqa_delegates_exactly_once_to_router() -> None:
    calls: list[dict[str, Any]] = []

    def route_fn(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"answer": "Yes", "trace": {"params": {}}}

    result = call(build("Is there a building in this image?"), route_fn=route_fn)
    assert len(calls) == 1
    assert result["answer"] == "Yes"


def test_executor_passes_correct_capability_and_plan_provenance() -> None:
    calls: list[dict[str, Any]] = []

    def route_fn(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"answer": "Yes", "trace": {"params": {}}}

    call(build("Is there a building in this image?"), route_fn=route_fn)
    assert calls[0]["capability"] == SINGLE_IMAGE_VQA
    assert calls[0]["planner_version"] == "phase0-rules-v1"
    assert calls[0]["planner_rule"] == "default_single_image_vqa"
    assert calls[0]["requested_capability"] is None
    assert calls[0]["execution_plan_version"] == "phase0-plan-v1"
    assert calls[0]["execution_step_id"] == "step_1"
    assert calls[0]["execution_step_index"] == 1
    assert calls[0]["execution_step_count"] == 1


def test_executor_does_not_execute_unavailable_grounding() -> None:
    def route_fn(**_: Any) -> dict[str, Any]:
        raise AssertionError("router must not run")

    with pytest.raises(CapabilityUnavailable):
        call(build("Where is the building?"), route_fn=route_fn)


def test_executor_does_not_execute_unavailable_change_vqa() -> None:
    def route_fn(**_: Any) -> dict[str, Any]:
        raise AssertionError("router must not run")

    with pytest.raises(CapabilityUnavailable):
        call(
            build("What changed between these images?", scenes=("a", "b")),
            route_fn=route_fn,
        )


def test_executor_does_not_execute_any_step_if_later_step_unavailable() -> None:
    calls: list[dict[str, Any]] = []

    def route_fn(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"answer": "Yes", "trace": {"params": {}}}

    with pytest.raises(CapabilityUnavailable):
        call(
            build("Where did flooding increase?", scenes=("a", "b")),
            route_fn=route_fn,
        )
    assert calls == []


def test_invalid_plan_is_rejected_before_execution() -> None:
    calls: list[dict[str, Any]] = []

    def route_fn(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"answer": "Yes", "trace": {"params": {}}}

    forged = build("Is water visible?")
    broken = ExecutionPlan(
        plan=forged.plan,
        execution_plan_version=forged.execution_plan_version,
        steps=(
            PlanStep(
                step_id="step_1",
                capability=SINGLE_IMAGE_VQA,
                depends_on=("ghost",),
                required_inputs=("single_scene",),
                provider_available=True,
                provider="qwen2.5vl-3b",
            ),
        ),
        unavailable_capabilities=(),
    )
    with pytest.raises(PlanValidationError, match="Unknown dependency"):
        execute_plan(
            broken,
            route_fn=route_fn,
            image_paths=["/tmp/scene.png"],
            question="Is water visible?",
            base_params={},
        )
    assert calls == []


def test_router_failure_propagates_unchanged() -> None:
    def route_fn(**_: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        call(build("Is water visible?"), route_fn=route_fn)


def test_executor_itself_does_not_write_trace() -> None:
    from unittest.mock import patch

    from orchestrator import trace as trace_store

    with (
        patch(
            "orchestrator.trace.append_record",
            side_effect=AssertionError("trace must not be written"),
        ) as append,
    ):
        call(
            build("Is water visible?"),
            route_fn=lambda **_: {"answer": "Yes", "trace": {"params": {}}},
        )
    append.assert_not_called()
