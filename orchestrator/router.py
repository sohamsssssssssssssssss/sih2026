"""Capability-oriented routing through the public Model interface only."""

from collections.abc import Mapping
from datetime import datetime, timezone
from threading import Thread
from typing import Any

from models.paths import public_path
from orchestrator import worker as inference_worker
from orchestrator.capabilities import ResolvedProvider, require_provider_ready
from orchestrator.registry import get
from orchestrator.trace import TraceIntegrityError, append_record
from orchestrator.worker import (  # noqa: F401 - re-exported for callers
    ModelExecutionTimeout,
    RemoteInferenceError,
    WorkerCrashed,
)


class InvalidModelOutput(RuntimeError):
    """The model returned data outside its public response contract."""


class TracePersistenceError(RuntimeError):
    """A validated execution could not be recorded safely."""


def _infer_in_thread(
    model: Any, image_paths: list[str], question: str, timeout_seconds: float
) -> Any:
    """Bounded wait for a lightweight, in-process model.

    Each call gets its own daemon thread, so an execution that overruns its
    timeout is abandoned without blocking any later request (the old single
    shared executor slot did exactly that). Only non-isolated models take this
    path: deterministic, CPU-only and stateless, so a rare abandoned thread
    finishes on its own and concurrent calls are safe. Heavyweight models that
    can genuinely hang are ``isolated`` and run in a killable worker process.
    """
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = model.infer(image_paths=image_paths, question=question)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    thread = Thread(target=target, name="satquery-inference", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise ModelExecutionTimeout
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _validate_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise InvalidModelOutput
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidModelOutput
    if "evidence" in result and not isinstance(result["evidence"], list):
        raise InvalidModelOutput
    return {**result, "answer": answer.strip()}


def _validate_execution_mode(
    result: dict[str, Any], params: dict[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(params, Mapping) or params.get("execution_mode") != "live":
        raise InvalidModelOutput
    if "execution_mode" in result and result["execution_mode"] != "live":
        raise InvalidModelOutput
    return dict(params)


def _model_version(provider: ResolvedProvider, model: Any) -> str:
    """Truthful trace metadata comes from the executed model object."""
    version = getattr(model, "version", None)
    if not provider.provider_name or not isinstance(version, str) or not version:
        raise InvalidModelOutput
    return version


def route(
    capability: str,
    image_paths: list[str],
    question: str,
    params: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    planner_version: str | None = None,
    planner_rule: str | None = None,
    requested_capability: str | None = None,
    execution_plan_version: str | None = None,
    execution_step_id: str | None = None,
    execution_step_index: int | None = None,
    execution_step_count: int | None = None,
) -> dict[str, Any]:
    """Resolve the capability, invoke its provider, validate, and trace.

    The capability binding and provider identity come from the provider
    registry; execution flows through the existing model registry so the
    hardened execution seam is preserved. The resolved capability, planner
    metadata, and execution-plan provenance are recorded truthfully in the
    execution trace and cannot be forged by caller params; caller params
    remain otherwise unchanged.
    """
    resolved = require_provider_ready(capability)
    model = get(resolved.model_name)
    if timeout_seconds is None:
        result = model.infer(image_paths=image_paths, question=question)
    elif getattr(model, "isolated", False) is True:
        # Killable worker process; the in-process instance above still supplies
        # the truthful version metadata (a class attribute, no weights loaded).
        result = inference_worker.run(
            resolved.model_name, image_paths, question, timeout_seconds
        )
    else:
        result = _infer_in_thread(model, image_paths, question, timeout_seconds)
    validated = _validate_result(result)
    validated_params = _validate_execution_mode(validated, params)
    model_version = _model_version(resolved, model)
    if (planner_version is None) != (planner_rule is None):
        raise InvalidModelOutput
    if planner_version is not None and (
        not isinstance(planner_version, str)
        or not planner_version
        or not isinstance(planner_rule, str)
        or not planner_rule
    ):
        raise InvalidModelOutput
    execution_metadata = (
        execution_plan_version,
        execution_step_id,
        execution_step_index,
        execution_step_count,
    )
    plan_metadata_incomplete = (
        not isinstance(execution_plan_version, str)
        or not execution_plan_version
        or not isinstance(execution_step_id, str)
        or not execution_step_id
        or not isinstance(execution_step_index, int)
        or isinstance(execution_step_index, bool)
        or execution_step_index < 1
        or not isinstance(execution_step_count, int)
        or isinstance(execution_step_count, bool)
        or execution_step_count < 1
    )
    if any(value is not None for value in execution_metadata) and plan_metadata_incomplete:
        raise InvalidModelOutput
    reserved = {
        "capability",
        "planner_version",
        "planner_rule",
        "requested_capability",
        "execution_plan_version",
        "execution_step_id",
        "execution_step_index",
        "execution_step_count",
    }
    traced_params = {
        key: value for key, value in validated_params.items() if key not in reserved
    }
    traced_params["capability"] = resolved.capability
    if planner_version is not None:
        traced_params.update(
            {
                "planner_version": planner_version,
                "planner_rule": planner_rule,
                "requested_capability": requested_capability,
            }
        )
    if not plan_metadata_incomplete:
        traced_params.update(
            {
                "execution_plan_version": execution_plan_version,
                "execution_step_id": execution_step_id,
                "execution_step_index": execution_step_index,
                "execution_step_count": execution_step_count,
            }
        )
    try:
        trace_record = append_record(
            {
                "model_name": resolved.provider_name,
                "model_version": model_version,
                "params": traced_params,
                "input_summary": {
                    "image_paths": [public_path(path) for path in image_paths],
                    "question": question,
                    "n_images": len(image_paths),
                },
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (TraceIntegrityError, OSError) as exc:
        raise TracePersistenceError from exc
    return {**validated, "trace": trace_record}
