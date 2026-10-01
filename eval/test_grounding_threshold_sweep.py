"""CPU-only tests for the Grounding DINO threshold sweep re-scoring logic."""

import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.suites.grounding_dior_rsvg import load_test_records, stratified_sample  # noqa: E402
from eval.suites.grounding_dior_rsvg_threshold_sweep import (  # noqa: E402
    BOX_GRID,
    TEXT_GRID,
    WEAK_CATEGORIES,
    rescore_combination,
)


class _Tokenizer:
    """Token-level stub mirroring the official tokenizer call interface."""

    def __init__(self) -> None:
        self.tokens = [
            "[CLS]",
            "a",
            "small",
            "red",
            "dam",
            "in",
            "the",
            "middle",
            ".",
            "[SEP]",
        ]

    def __call__(self, caption: str) -> dict:
        return {"tokens": self.tokens}


def _reference_phrase_fn():
    """Reference label decode mirroring the official get_phrases_from_posmap."""

    def phrase_fn(logit_values, tokenized, tokenizer, text_threshold):
        keep = [value > text_threshold for value in logit_values]
        return " ".join(
            token for token, keep_it in zip(tokenized["tokens"], keep) if keep_it
        ).replace(".", "").strip()

    return phrase_fn


def _official_keep_reference(rows_logits, box_threshold):
    """Official predict() keep-mask and confidence ordering, copied line-faithfully."""
    confidences = [max(row) for row in rows_logits]
    kept_logits = [row for row in rows_logits if max(row) > box_threshold]
    return kept_logits, confidences


@pytest.fixture
def phrase_fn():
    return _reference_phrase_fn()


def test_grids_and_weak_categories() -> None:
    assert BOX_GRID == (0.20, 0.25, 0.30, 0.35)
    assert TEXT_GRID == (0.15, 0.20, 0.25)
    assert WEAK_CATEGORIES == (
        "dam",
        "tenniscourt",
        "windmill",
        "baseballfield",
        "expressway_toll_station",
        "harbor",
        "overpass",
    )


def test_rescore_matches_official_keep_and_labels(phrase_fn) -> None:
    """Kept-mask and labels must equal an independent copy of official semantics."""
    rows_logits = [
        [0.1, 0.9, 0.3],  # max 0.90 -> kept at every box threshold in the grid
        [0.2, 0.34, 0.1],  # max 0.34 -> kept at box<=0.30, dropped at 0.35
        [0.05, 0.19, 0.02],  # max 0.19 -> dropped at every box threshold
    ]
    tokenized = {"tokens": _Tokenizer().tokens}
    raw = {
        "caption": "a small red dam in the middle",
        "tokenized": tokenized,
        "rows": [
            {
                "confidence": max(logits),
                "box_cxcywh": [0.5 + 0.1 * index, 0.5, 0.2, 0.2],
                "logits": logits,
            }
            for index, logits in enumerate(rows_logits)
        ],
    }
    tokenizer = _Tokenizer()

    for box_threshold in BOX_GRID:
        rescored = rescore_combination(
            raw, tokenizer, box_threshold, 0.20, phrase_fn=phrase_fn
        )
        kept_ref, _ = _official_keep_reference(rows_logits, box_threshold)
        expected_labels = [
            phrase_fn(logits, tokenized, tokenizer, 0.20) for logits in kept_ref
        ]
        actual_labels = [box["label"] for box in rescored["evidence"]]
        assert actual_labels == expected_labels
        actual_confidences = [box["confidence"] for box in rescored["evidence"]]
        assert actual_confidences == [max(logits) for logits in kept_ref]


def test_text_threshold_does_not_change_selection(phrase_fn) -> None:
    """At fixed box_threshold, coordinates+confidence must be text-invariant."""
    raw = {
        "caption": "a small red dam in the middle",
        "tokenized": {"tokens": _Tokenizer().tokens},
        "rows": [
            {
                "confidence": confidence,
                "box_cxcywh": [0.5, 0.5, 0.2, 0.2],
                "logits": [0.1, confidence, 0.3],
            }
            for confidence in (0.19, 0.21, 0.36, 0.55)
        ],
    }
    tokenizer = _Tokenizer()

    for box_threshold in BOX_GRID:
        selections = []
        for text_threshold in TEXT_GRID:
            rescored = rescore_combination(
                raw, tokenizer, box_threshold, text_threshold, phrase_fn=phrase_fn
            )
            selections.append(
                [(box["coordinates"], box["confidence"]) for box in rescored["evidence"]]
            )
        assert selections[0] == selections[1] == selections[2]


def test_no_queries_kept_yields_no_match(phrase_fn) -> None:
    raw = {
        "caption": "a small red dam in the middle",
        "tokenized": {"tokens": _Tokenizer().tokens},
        "rows": [
            {
                "confidence": 0.05,
                "box_cxcywh": [0.5, 0.5, 0.2, 0.2],
                "logits": [0.05, 0.02, 0.01],
            }
        ],
    }
    tokenizer = _Tokenizer()
    rescored = rescore_combination(raw, tokenizer, 0.20, 0.15, phrase_fn=phrase_fn)
    assert rescored["evidence"] == []
    assert rescored["answer"] == "No match found for 'a small red dam in the middle'."


def test_sample_list_is_deterministic_and_complete(tmp_path: Path) -> None:
    """Same seed + suite sampling function -> identical, complete sample list."""
    from eval.suites.grounding_dior_rsvg import DIOR_CATEGORIES

    (tmp_path / "Annotations").mkdir()
    (tmp_path / "JPEGImages").mkdir()
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = "00001.jpg"
    for index, category in enumerate(DIOR_CATEGORIES):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = category
        box = ET.SubElement(obj, "bndbox")
        for name, value in zip(("xmin", "ymin", "xmax", "ymax"), (1, 1, 2, 2)):
            ET.SubElement(box, name).text = str(value)
        ET.SubElement(obj, "description").text = f"the {category} number {index}"
    ET.ElementTree(root).write(tmp_path / "Annotations" / "00001.xml")
    (tmp_path / "test.txt").write_text(
        "\n".join(str(i) for i in range(len(DIOR_CATEGORIES))) + "\n",
        encoding="utf-8",
    )

    _, records = load_test_records(
        tmp_path, expected_split_size=len(DIOR_CATEGORIES)
    )
    first = stratified_sample(records, len(DIOR_CATEGORIES), 26167)
    second = stratified_sample(records, len(DIOR_CATEGORIES), 26167)
    assert first == second
    assert {sample["category"] for sample in first} == set(DIOR_CATEGORIES)
    assert [s["test_index"] for s in first] == sorted(
        s["test_index"] for s in first
    )


def test_module_imports_without_gpu_stack() -> None:
    """Module must import without groundingdino/torch (CPU test env)."""
    module = importlib.import_module("eval.suites.grounding_dior_rsvg_threshold_sweep")
    importlib.reload(module)
    assert callable(module.rescore_combination)
    assert callable(module.capture_raw_queries)


@pytest.mark.parametrize("box_threshold", BOX_GRID)
@pytest.mark.parametrize("text_threshold", TEXT_GRID)
def test_rescore_handles_empty_rows(phrase_fn, box_threshold, text_threshold) -> None:
    raw = {"caption": "a dam", "tokenized": {"tokens": _Tokenizer().tokens}, "rows": []}
    rescored = rescore_combination(
        raw, _Tokenizer(), box_threshold, text_threshold, phrase_fn=phrase_fn
    )
    assert rescored == {
        "answer": "No match found for 'a dam'.",
        "evidence": [],
    }
