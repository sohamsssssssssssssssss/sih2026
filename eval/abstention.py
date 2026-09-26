"""Selective-prediction metrics over per-example (score, correct) records.

A record is ``(score, correct)``. ``score`` is a model-produced number where
higher means "more willing to answer" (e.g. a raw Grounding DINO detection
score); ``None`` means the system produced no answer for that example (it
abstained, e.g. no box above the detection threshold). Scores are never
invented: a caller without a real per-example score must not use this module.

Definitions (all fractions are over examples, ties handled by threshold):
- coverage(t)  = #{score >= t} / N, with N including abstentions
- risk(t)      = 1 - accuracy among examples with score >= t
- AURC         = right-step integral of risk over coverage, from 0 to the
                 maximum reachable coverage (< 1 when the system abstains)
- ECE          = sum_b (n_b / n_scored) * |accuracy_b - mean_score_b| over
                 equal-width score bins on [0, 1]; only meaningful if the
                 score is claimed to be a probability of correctness.
"""

import math
from collections.abc import Iterable, Sequence

Record = tuple[float | None, bool]


def validate(records: Iterable[Record]) -> list[Record]:
    checked = []
    for score, correct in records:
        if not isinstance(correct, bool):
            raise ValueError(f"correct must be bool, got {correct!r}")
        if score is not None and (isinstance(score, bool) or not math.isfinite(score)):
            raise ValueError(f"score must be a finite number or None, got {score!r}")
        checked.append((None if score is None else float(score), correct))
    if not checked:
        raise ValueError("no records")
    return checked


def risk_coverage(records: Iterable[Record]) -> list[dict]:
    """One point per distinct score threshold, highest threshold first."""
    records = validate(records)
    total = len(records)
    scored = sorted((r for r in records if r[0] is not None), key=lambda r: r[0], reverse=True)
    curve, selected, correct = [], 0, 0
    for index, (score, is_correct) in enumerate(scored):
        selected += 1
        correct += is_correct
        if index + 1 < len(scored) and scored[index + 1][0] == score:
            continue  # ties enter together; a threshold cannot split them
        curve.append({"threshold": score, "n_selected": selected, "coverage": selected / total,
                      "selective_accuracy": correct / selected, "risk": 1 - correct / selected})
    return curve


def aurc(curve: Sequence[dict]) -> float | None:
    previous, area = 0.0, 0.0
    for point in curve:
        area += (point["coverage"] - previous) * point["risk"]
        previous = point["coverage"]
    return area if curve else None


def selective_accuracy_at(curve: Sequence[dict], target_coverage: float) -> dict | None:
    """The largest-coverage operating point not exceeding the target."""
    eligible = [point for point in curve if point["coverage"] <= target_coverage + 1e-12]
    return eligible[-1] if eligible else None


def expected_calibration_error(records: Iterable[Record], n_bins: int = 15) -> dict:
    scored = [r for r in validate(records) if r[0] is not None]
    if not scored:
        return {"ece": None, "n_scored": 0, "bins": []}
    if any(not 0.0 <= score <= 1.0 for score, _ in scored):
        raise ValueError("ECE requires scores in [0, 1]")
    bins = []
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        members = [r for r in scored if lower <= r[0] < upper or (index == n_bins - 1 and r[0] == 1.0)]
        if members:
            bins.append({"lower": lower, "upper": upper, "n": len(members),
                         "mean_score": sum(r[0] for r in members) / len(members),
                         "accuracy": sum(r[1] for r in members) / len(members)})
    ece = sum(b["n"] / len(scored) * abs(b["accuracy"] - b["mean_score"]) for b in bins)
    return {"ece": ece, "n_scored": len(scored), "bins": bins}


def summarize(records: Iterable[Record], coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0)) -> dict:
    records = validate(records)
    curve = risk_coverage(records)
    return {
        "n": len(records),
        "n_abstained": sum(score is None for score, _ in records),
        "full_coverage_accuracy": sum(correct for _, correct in records) / len(records),
        "max_coverage": curve[-1]["coverage"] if curve else 0.0,
        "aurc": aurc(curve),
        "selective_accuracy": {str(c): selective_accuracy_at(curve, c) for c in coverages},
        "calibration": expected_calibration_error(records),
        "risk_coverage": curve,
    }


def grounding_records(results: Iterable[dict], iou_threshold: float = 0.5) -> list[Record]:
    """Adapt eval/suites/grounding_dior_rsvg.py per-example results.

    Score = the provider's raw detection score of the selected (top) box; an
    example with no returned box is an abstention (score None, incorrect).
    """
    records = []
    for result in results:
        top = result.get("selected_prediction")
        score = top.get("confidence") if top else None
        records.append((score, top is not None and result["iou"] >= iou_threshold))
    return records
