"""Execution-plan representation and structural validation.

Phase 0 bridge between deterministic request planning and future agentic
orchestration: an ExecutionPlan is an ordered, dependency-checked sequence of
PlanStep records. Representation is deliberately richer than execution — a
plan may be structurally valid while non-executable because capability
providers do not exist yet. Nothing here invokes a model or writes a trace.
"""

from dataclasses import dataclass

from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    OPTICAL_SAR,
    KNOWN_CAPABILITIES,
    CapabilityUnavailable,
    resolve_provider,
)
from orchestrator.planner import TEMPORAL_LOCALIZATION_RULE_ID, Plan

EXECUTION_PLAN_VERSION = "phase0-plan-v1"


class PlanValidationError(ValueError):
    """An execution plan is structurally invalid and must not run."""


@dataclass(frozen=True)
class PlanStep:
    """One ordered unit of work inside an execution plan.

    ``required_inputs`` is symbolic ("scene_pair", "step_1.output") — no
    intermediate artifact framework exists yet and none is implied.
    """

    step_id: str
    capability: str
    depends_on: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()
    provider_available: bool = False
    provider: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable structured plan derived from a planner decision.

    The embedded Plan keeps planner metadata (version, rule, requested and
    selected capability, inputs, reason); ``steps`` describes the ordered
    work. ``executable`` is true only when the planner decision is executable
    and every step has an available provider.
    """

    plan: Plan
    execution_plan_version: str
    steps: tuple[PlanStep, ...]
    unavailable_capabilities: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.plan.executable and not self.unavailable_capabilities


def validate_plan(plan: ExecutionPlan) -> None:
    """Structural validation; raises PlanValidationError on any violation."""
    steps = plan.steps
    if not steps:
        raise PlanValidationError("Execution plan contains no steps.")
    seen_ids: set[str] = set()
    for step in steps:
        if not isinstance(step.step_id, str) or not step.step_id:
            raise PlanValidationError(
                "Execution plan step ids must be non-empty strings."
            )
        if step.step_id in seen_ids:
            raise PlanValidationError(f"Duplicate execution step id: {step.step_id}")
        seen_ids.add(step.step_id)
    positions = {step.step_id: index for index, step in enumerate(steps)}
    for step in steps:
        if step.capability not in KNOWN_CAPABILITIES:
            raise PlanValidationError(
                f"Unknown capability in execution plan: {step.capability}"
            )
        for dependency in step.depends_on:
            if dependency == step.step_id:
                raise PlanValidationError(f"Step depends on itself: {step.step_id}")
            if dependency not in positions:
                raise PlanValidationError(f"Unknown dependency: {dependency}")
    _reject_cycles(steps)
    for step in steps:
        for dependency in step.depends_on:
            if positions[dependency] > positions[step.step_id]:
                raise PlanValidationError(
                    "Dependency must precede its dependent step: "
                    f"{dependency} -> {step.step_id}"
                )
    seen_shapes: set[tuple[str, tuple[str, ...]]] = set()
    for step in steps:
        shape = (step.capability, step.required_inputs)
        if shape in seen_shapes:
            raise PlanValidationError(f"Duplicate meaningless step: {step.capability}")
        seen_shapes.add(shape)


def _reject_cycles(steps: tuple[PlanStep, ...]) -> None:
    adjacency = {step.step_id: tuple(step.depends_on) for step in steps}
    state = {step_id: 0 for step_id in adjacency}  # 0 new, 1 in progress, 2 done
    for root in adjacency:
        if state[root]:
            continue
        state[root] = 1
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, edges = stack[-1]
            for dependency in edges:
                if state[dependency] == 1:
                    raise PlanValidationError(
                        "Execution plan contains a dependency cycle."
                    )
                if state[dependency] == 0:
                    state[dependency] = 1
                    stack.append((dependency, iter(adjacency[dependency])))
                    break
            else:
                state[node] = 2
                stack.pop()


def _step_required_inputs(capability: str, *, chained_output: bool) -> tuple[str, ...]:
    if chained_output:
        return ("step_1.output",)
    if capability == CHANGE_VQA:
        return ("scene_pair",)
    if capability == OPTICAL_SAR:
        return ("optical_scene", "sar_scene")
    return ("single_scene",)


def _resolve_step(
    capability: str,
    *,
    step_id: str,
    depends_on: tuple[str, ...],
    required_inputs: tuple[str, ...],
) -> PlanStep:
    try:
        resolved = resolve_provider(capability)
    except CapabilityUnavailable:
        return PlanStep(
            step_id=step_id,
            capability=capability,
            depends_on=depends_on,
            required_inputs=required_inputs,
            provider_available=False,
            provider=None,
        )
    return PlanStep(
        step_id=step_id,
        capability=capability,
        depends_on=depends_on,
        required_inputs=required_inputs,
        provider_available=True,
        provider=resolved.provider_name,
    )


def build_execution_plan(plan: Plan) -> ExecutionPlan:
    """Derive the deterministic structured plan from a planner decision.

    The combined temporal+localization rule represents the future change_vqa
    → grounding decomposition; representation never implies executability.
    """
    if plan.rule_id == TEMPORAL_LOCALIZATION_RULE_ID:
        capability_chain = (CHANGE_VQA, GROUNDING)
    else:
        capability_chain = (plan.selected_capability,)
    steps = tuple(
        _resolve_step(
            capability,
            step_id=f"step_{index}",
            depends_on=() if index == 1 else (f"step_{index - 1}",),
            required_inputs=_step_required_inputs(
                capability, chained_output=index > 1
            ),
        )
        for index, capability in enumerate(capability_chain, start=1)
    )
    unavailable = tuple(step.capability for step in steps if not step.provider_available)
    execution = ExecutionPlan(
        plan=plan,
        execution_plan_version=EXECUTION_PLAN_VERSION,
        steps=steps,
        unavailable_capabilities=unavailable,
    )
    validate_plan(execution)
    return execution
