"""Gate-before-execution orchestration and conservative contract assembly."""
from bisect import bisect_left
import hashlib
import json
import math
from pathlib import Path

from core.gate import OUTSIDE_INDIA, GateDecision, evaluate
from core.registry import ToolRegistry
from core.router import CHANGE_UNSUPPORTED, Route, route
from core.schema import Answer, Evidence, Query, Scene, ToolResult, ToolStatus
from orchestrator.trace import append_record


def record_events(trace_id: str, phase: str, events):
    if not trace_id.strip():
        raise ValueError("trace_id must be non-empty")
    for event in events:
        append_record({"trace_id": trace_id, "phase": phase, **event})


def _without_confidence(value):
    if isinstance(value, dict):
        return {k: _without_confidence(v) for k, v in value.items() if "confidence" not in k.lower()}
    if isinstance(value, list):
        return [_without_confidence(v) for v in value]
    return value


def _calibrate(result: ToolResult, path: str | Path | None):
    if path is None or result.raw_confidence is None:
        return None, "absent_map_or_raw_score", None
    digest = None
    try:
        payload = Path(path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        data = json.loads(payload)
        if (data["tool"], data["tool_version"], data["checkpoint_sha"]) != (
                result.tool, result.tool_version, result.checkpoint_sha):
            return None, "identity_mismatch", digest
        raw, calibrated = data["raw"], data["calibrated"]
        if not isinstance(raw, list) or not isinstance(calibrated, list) or len(raw) != len(calibrated) or len(raw) < 2:
            return None, "invalid_knots", digest
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and 0 <= v <= 1 for v in raw + calibrated):
            return None, "invalid_knots", digest
        if any(a >= b for a, b in zip(raw, raw[1:])) or any(a > b for a, b in zip(calibrated, calibrated[1:])):
            return None, "invalid_knots", digest
        score = result.raw_confidence
        if not raw[0] <= score <= raw[-1]:
            return None, "outside_map_range", digest
        i = max(1, bisect_left(raw, score))
        fraction = (score - raw[i - 1]) / (raw[i] - raw[i - 1])
        return calibrated[i - 1] + fraction * (calibrated[i] - calibrated[i - 1]), "mapped", digest
    except (OSError, ValueError, KeyError, TypeError):
        return None, "invalid_map", digest


def assemble(result: ToolResult | None, gate: GateDecision, status: ToolStatus | None, *,
             trace_id: str, calibration_path: str | Path | None = None) -> Answer:
    evidence = gate.evidence
    reason = gate.refusal_reason
    if reason is None and (result is None or status is None or status.status == "disabled"):
        reason = "tool_disabled" if status and status.status == "disabled" else "tool_missing"
    if reason is None and not isinstance(result, ToolResult):
        reason = "checks_disagree"
    if reason is None and (result.tool != gate.tool or (result.tool, result.tool_version) != (status.name, status.version)):
        reason = "checks_disagree"
    if reason is None and isinstance(result.value, dict) and ("error" in result.value or "error_code" in result.value):
        reason = "tool_error"
        record_events(trace_id, "execution", [{"outcome": "error", "error_type": result.value.get("error_type", "ToolError")}])
    if reason is not None:
        record_events(trace_id, "answer", [{"rule": "status", "status": "refused", "refusal_reason": reason},
                                          {"rule": "calibration", "outcome": "not_applicable"}])
        text = f"Cannot answer: {reason}."
        if reason == "not_answerable":
            policies = [e.summary.get("policy") for e in evidence]
            if "change_unsupported" in policies:
                text = CHANGE_UNSUPPORTED
            elif "out_of_region" in policies:
                text = OUTSIDE_INDIA
        return Answer(status="refused", text=text, refusal_reason=reason,
                      evidence=evidence, trace_id=trace_id)
    experimental = status.status != "validated"
    clean_result = ToolResult.model_validate({
        **result.model_dump(),
        "value": _without_confidence(result.value),
        "raw_confidence": None if experimental else result.raw_confidence,
        "evidence": [{**e.model_dump(), "summary": _without_confidence(e.summary)} for e in result.evidence],
    })
    label = Evidence(kind="text", summary={"tool_status": status.status,
                                          "label": "experimental_unvalidated" if experimental else "validated"})
    confidence, calibration, map_sha = ((None, "unvalidated_tool", None) if experimental else
                                         (None, "gate_warnings", None) if not gate.clean else _calibrate(result, calibration_path))
    answer_status = "answered" if not experimental and gate.clean else "low_confidence"
    record_events(trace_id, "answer", [{"rule": "calibration", "outcome": calibration, "confidence": confidence,
                                      "tool": result.tool, "tool_version": result.tool_version,
                                      "checkpoint_sha": result.checkpoint_sha, "map_sha256": map_sha},
                                      {"rule": "status", "status": answer_status, "tool_status": status.status,
                                       "gate_clean": gate.clean}])
    text = clean_result.value.get("answer") if isinstance(clean_result.value, dict) else None
    if not isinstance(text, str):
        text = "Tool result available in evidence."
    if any(e.summary.get("warning") == "cloud_unknown" for e in evidence):
        text += " Cloud cover not checked (no cloud metadata)."
    return Answer(status=answer_status, text=text,
                  confidence=confidence, primary=clean_result, sensors_used=("S1", "S2") if gate.mode == "S1S2" else (gate.mode,),
                  evidence=(*evidence, *clean_result.evidence, label), trace_id=trace_id)


def decide(query: Query, scene: Scene, registry: ToolRegistry, *, trace_id: str, **gate_inputs) -> tuple[Route, GateDecision]:
    choice = route(query, scene, has_second_scene=gate_inputs.get("second_scene") is not None)
    record_events(trace_id, "route", [{"intent": choice.intent, "reason": choice.reason}])
    record_events(trace_id, "route", [{"rank": i, "tool": name, "reason": choice.reason}
                                      for i, name in enumerate(choice.tools, 1)])
    decision = evaluate(query, scene, choice, registry, **gate_inputs)
    record_events(trace_id, "gate", [{**e.summary, "needs_sensor": list(e.needs_sensor)} for e in decision.evidence])
    return choice, decision


def run_query(query: Query, scene: Scene, pixels_by_sensor: dict, registry: ToolRegistry, *,
              trace_id: str, calibration_path: str | Path | None = None, **gate_inputs) -> Answer:
    """Caller supplies mode-specific pixels; cloudy optical pixels cannot stand in for radar.

    Joint optical-SAR pixels are a sensor pair.
    Pixel loaders/adapters retain responsibility for grid, units and payload checks.
    """
    _, decision = decide(query, scene, registry, trace_id=trace_id, **gate_inputs)
    if not decision.passed:
        return assemble(None, decision, None, trace_id=trace_id)
    tool = registry.get(decision.tool)
    if decision.mode not in pixels_by_sensor:
        missing = Evidence(kind="text", needs_sensor=(decision.mode,),
                           summary={"rule": "pixel_input", "outcome": "refused", "mode": decision.mode})
        record_events(trace_id, "gate", [{**missing.summary, "needs_sensor": list(missing.needs_sensor)}])
        decision = GateDecision(decision.tool, decision.mode, "sensor_missing", (*decision.evidence, missing))
        return assemble(None, decision, tool.status, trace_id=trace_id)
    record_events(trace_id, "execution", [{"tool": tool.name, "mode": decision.mode, "outcome": "start"}])
    try:
        result = tool.run(scene, query, pixels_by_sensor[decision.mode])
    except Exception as error:
        record_events(trace_id, "execution", [{"tool": tool.name, "outcome": "error", "error_type": type(error).__name__}])
        decision = GateDecision(decision.tool, decision.mode, "tool_error", decision.evidence)
        return assemble(None, decision, tool.status, trace_id=trace_id)
    return assemble(result, decision, tool.status, trace_id=trace_id, calibration_path=calibration_path)
