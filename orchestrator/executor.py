"""Minimal plan executor: validated, all-or-nothing plan execution.

Separation of responsibilities is deliberate and stays narrow:
planner.py creates the intent decision, execution_plan.py represents and
validates the structured plan, this module executes an already validated
plan, and router.py remains the single-provider invocation boundary (and the
only trace owner). No replanning, no autonomy, no unavailable steps.
"""

from collections.abc import Callable
from typing import Any

from orchestrator.capabilities import IMPLEMENTED_CAPABILITIES, CapabilityUnavailable
from orchestrator.execution_plan import (
    EXECUTION_PLAN_VERSION,
    ExecutionPlan,
    PlanStep,
    validate_plan,
)

RouteFn = Callable[..., dict[str, Any]]

_SINGLE_STEP_ONLY = "Execution plan is not a supported single executable step."
_REQUIRED_INPUTS = {
    capability: (
        ("optical_scene", "sar_scene")
        if capability == "optical_sar"
        else ("scene_pair",)
        if capability == "change_vqa"
        else ("single_scene",)
    )
    for capability in IMPLEMENTED_CAPABILITIES
}


def execute_plan(
    plan: ExecutionPlan,
    *,
    route_fn: RouteFn,
    image_paths: list[str],
    question: str,
    base_params: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute a structurally validated plan atomically or not at all.

    Every step must have an available provider before the first step runs;
    a plan that cannot complete never invokes any provider. The current one
    executable shape is exactly one available implemented-capability step,
    which delegates to the existing capability router with plan provenance.
    """
    validate_plan(plan)
    unavailable = [step.capability for step in plan.steps if not step.provider_available]
    if unavailable:
        raise CapabilityUnavailable(
            plan.plan.unavailable_reason or f"Provider unavailable for capability: {unavailable[0]}"
        )
    if not plan.executable:
        raise CapabilityUnavailable(
            plan.plan.unavailable_reason or "The execution plan cannot be executed."
        )
    if len(plan.steps) != 1:
        raise CapabilityUnavailable(_SINGLE_STEP_ONLY)
    step = plan.steps[0]
    _assert_single_step_shape(step)
    return route_fn(
        capability=step.capability,
        image_paths=image_paths,
        question=question,
        params=base_params,
        timeout_seconds=timeout_seconds,
        planner_version=plan.plan.planner_version,
        planner_rule=plan.plan.rule_id,
        requested_capability=plan.plan.requested_capability,
        execution_plan_version=EXECUTION_PLAN_VERSION,
        execution_step_id=step.step_id,
        execution_step_index=1,
        execution_step_count=len(plan.steps),
    )


def _assert_single_step_shape(step: PlanStep) -> None:
    if step.required_inputs != _REQUIRED_INPUTS.get(step.capability):
        raise CapabilityUnavailable(_SINGLE_STEP_ONLY)
