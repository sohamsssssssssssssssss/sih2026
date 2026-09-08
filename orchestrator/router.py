"""Capability-oriented routing through the public Model interface only."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from threading import Event
from typing import Any

from orchestrator.capabilities import ResolvedProvider, resolve_provider
from orchestrator.registry import get
from orchestrator.trace import TraceIntegrityError, append_record

# ponytail: one worker protects the singleton model; process workers are needed for cancellation.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class InvalidModelOutput(RuntimeError):
    """The model returned data outside its public response contract."""


class ModelExecutionTimeout(RuntimeError):
    """The request stopped waiting for an in-flight model execution."""


class TracePersistenceError(RuntimeError):
    """A validated execution could not be recorded safely."""


def _infer(model: Any, image_paths: list[str], question: str, abandoned: Event) -> Any:
    if abandoned.is_set():
        raise ModelExecutionTimeout
    return model.infer(image_paths=image_paths, question=question)


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
) -> dict[str, Any]:
    """Resolve the capability, invoke its provider, validate, and trace.

    The capability binding and provider identity come from the provider
    registry; execution flows through the existing model registry so the
    hardened execution seam is preserved. The resolved capability is recorded
    truthfully in the execution trace; caller params remain otherwise unchanged.
    """
    resolved = resolve_provider(capability)
    model = get(resolved.model_name)
    if timeout_seconds is None:
        result = model.infer(image_paths=image_paths, question=question)
    else:
        abandoned = Event()
        future = _INFERENCE_EXECUTOR.submit(
            _infer, model, image_paths, question, abandoned
        )
        try:
            result = future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            abandoned.set()
            future.cancel()
            raise ModelExecutionTimeout from exc
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
    reserved = {
        "capability",
        "planner_version",
        "planner_rule",
        "requested_capability",
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
    try:
        trace_record = append_record(
            {
                "model_name": resolved.provider_name,
                "model_version": model_version,
                "params": traced_params,
                "input_summary": {
                    "image_paths": image_paths,
                    "question": question,
                    "n_images": len(image_paths),
                },
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (TraceIntegrityError, OSError) as exc:
        raise TracePersistenceError from exc
    return {**validated, "trace": trace_record}
