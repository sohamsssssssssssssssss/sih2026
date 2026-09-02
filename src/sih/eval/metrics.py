"""Scoring. OWNER: Lead. No number enters a deck unless it came through here."""

from __future__ import annotations

import re


def normalise(text: str) -> str:
    """Canonicalise a prediction for comparison against ground truth."""
    t = re.sub(r"[^a-z0-9\s]", "", str(text).strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return t


def summarise(records: list[dict]) -> dict:
    """Aggregate per-sample records into the standard result block.

    Reports binary and open-ended accuracy SEPARATELY. RSVQA is heavily
    yes/no weighted, so a degenerate always-yes model scores ~60% on the
    aggregate and looks like a working baseline. The split, plus
    pred_yes_rate_on_binary, is how you catch that.
    """
    n = max(len(records), 1)
    binary = [r for r in records if r["gold"] in ("yes", "no")]
    other = [r for r in records if r["gold"] not in ("yes", "no")]

    return {
        "n": len(records),
        "accuracy": sum(r["correct"] for r in records) / n,
        "binary_n": len(binary),
        "binary_accuracy": sum(r["correct"] for r in binary) / max(len(binary), 1),
        "open_n": len(other),
        "open_accuracy": sum(r["correct"] for r in other) / max(len(other), 1),
        "pred_yes_rate_on_binary": (
            sum(r["pred"] == "yes" for r in binary) / max(len(binary), 1)
        ),
    }


def degenerate(summary: dict) -> str | None:
    """Return a warning string if the result is degenerate, else None."""
    if summary["pred_yes_rate_on_binary"] > 0.85:
        return ("Model answers 'yes' to >85% of binary questions. "
                "This accuracy is degenerate — fix prompting before trusting it.")
    if summary["open_n"] > 20 and summary["open_accuracy"] < 0.02:
        return "Open-ended accuracy near zero — check answer normalisation."
    return None
