"""Routing evaluation over the real-phrasing question set.

The schema test always runs. The accuracy test runs once
``orchestrator.question`` exists: it plans every question and requires at
least 90% capability accuracy. Parser field accuracy (place, dates, missing)
is printed for information only; run with ``-s`` to see it on a pass.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    KNOWN_CAPABILITIES,
    OPTICAL_SAR,
    SINGLE_IMAGE_VQA,
)

QUESTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "eval" / "questions.v1.jsonl"
)
INTENT_CAPABILITY = {
    "flood_change": CHANGE_VQA,
    "change": CHANGE_VQA,
    "locate": GROUNDING,
    "describe": SINGLE_IMAGE_VQA,
    "optical_sar": OPTICAL_SAR,
    # The planner has no abstain route; unknown falls through to its default.
    "unknown": SINGLE_IMAGE_VQA,
}
MISSING_VALUES = {"place", "before_date", "after_date", "year"}
ROW_KEYS = {"id", "question", "expected", "category", "notes"}
EXPECTED_KEYS = {"capability", "intent", "place", "before", "after", "missing"}
MIN_QUESTIONS = 100
MIN_ROUTING_ACCURACY = 0.90
SCENE_PAIR = ("a", "b")
# ponytail: the parser's entry-point name is not part of the contract; extend if it differs.
PARSER_NAMES = ("parse_question", "parse")
PARSED_FIELDS = ("intent", "place", "before", "after", "missing")


def _rows() -> list[dict]:
    lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _iso(value: str | None) -> date | None:
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    assert parsed.isoformat() == value, f"not YYYY-MM-DD: {value!r}"
    return parsed


def _check_row(row: dict) -> None:
    assert set(row) == ROW_KEYS
    expected = row["expected"]
    assert set(expected) == EXPECTED_KEYS
    assert isinstance(row["question"], str) and row["question"].strip()
    assert isinstance(row["category"], str) and row["category"]
    assert isinstance(row["notes"], str)
    intent, capability = expected["intent"], expected["capability"]
    assert intent in INTENT_CAPABILITY
    assert capability in KNOWN_CAPABILITIES
    assert capability == INTENT_CAPABILITY[intent]
    place = expected["place"]
    assert place is None or (isinstance(place, str) and place.strip())
    before, after = _iso(expected["before"]), _iso(expected["after"])
    if before is not None and after is not None:
        assert before <= after
    missing = expected["missing"]
    assert isinstance(missing, list) and len(missing) == len(set(missing))
    assert set(missing) <= MISSING_VALUES
    if intent == "flood_change":
        # flood_change needs place, before and after; an absent one must be reported.
        assert ("place" in missing) == (place is None)
        if before is None:
            assert {"before_date", "year"} & set(missing)
        if after is None:
            assert {"after_date", "year"} & set(missing)
    else:
        assert not missing, "only flood_change needs place or dates"


def test_question_set_schema():
    rows = _rows()
    assert len(rows) >= MIN_QUESTIONS
    ids = [row.get("id") for row in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for row in rows:
        try:
            _check_row(row)
        except AssertionError as error:
            raise AssertionError(f"{row.get('id')}: {error}") from error


def _field_values(intent: str, place: str | None, before, after, missing) -> dict:
    return {
        "intent": intent,
        "place": place.strip().casefold() if place else None,
        "before": before.isoformat() if isinstance(before, date) else before,
        "after": after.isoformat() if isinstance(after, date) else after,
        "missing": frozenset(missing),
    }


def _report_field_accuracy(module, rows: list[dict]) -> None:
    parse = next(
        (getattr(module, name) for name in PARSER_NAMES if callable(getattr(module, name, None))),
        None,
    )
    if parse is None:
        print(f"field accuracy not measured: no {' or '.join(PARSER_NAMES)} in {module.__name__}")
        return
    wrong: dict[str, list[str]] = {field: [] for field in PARSED_FIELDS}
    try:
        for row in rows:
            got = parse(row["question"])
            actual = _field_values(*(getattr(got, field) for field in PARSED_FIELDS))
            expected = _field_values(*(row["expected"][field] for field in PARSED_FIELDS))
            for field in PARSED_FIELDS:
                if actual[field] != expected[field]:
                    wrong[field].append(row["id"])
    except Exception as error:  # informational report; routing accuracy is still asserted
        print(f"field accuracy not measured: {parse.__name__} failed: {error!r}")
        return
    print("field accuracy (informational, not asserted):")
    for field, ids in wrong.items():
        accuracy = 1 - len(ids) / len(rows)
        print(f"  {field}: {accuracy:.3f}" + (f"  wrong: {', '.join(ids)}" if ids else ""))


@pytest.mark.usefixtures("ready_providers")
def test_routing_accuracy():
    question_module = pytest.importorskip("orchestrator.question")
    from orchestrator.planner import PlanRequest, plan_request

    rows = _rows()
    misrouted = []
    for row in rows:
        plan = plan_request(PlanRequest(question=row["question"], scene_ids=SCENE_PAIR))
        expected = row["expected"]["capability"]
        if plan.selected_capability != expected:
            misrouted.append(
                f"{row['id']}: expected {expected}, got {plan.selected_capability}"
                f" | {row['question']}"
            )
    accuracy = 1 - len(misrouted) / len(rows)
    print(f"routing accuracy: {accuracy:.3f} ({len(rows) - len(misrouted)}/{len(rows)})")
    print("\n".join(misrouted))
    _report_field_accuracy(question_module, rows)
    assert accuracy >= MIN_ROUTING_ACCURACY, (
        f"routing accuracy {accuracy:.3f} < {MIN_ROUTING_ACCURACY}; misrouted:\n"
        + "\n".join(misrouted)
    )
