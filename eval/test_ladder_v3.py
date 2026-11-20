import json
from collections import Counter
from pathlib import Path

import numpy as np

from eval.ladder import CLASS_NAMES
from eval.ladder_v3 import canonical_answer, metrics, sha, tile_questions
from kaggle.run_ladder_v3 import summarize


def test_frozen_manifest_preserves_v2_order_and_balances_each_class():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/manifests/ladder-v3/questions.jsonl"
    assert sha(path) in (root / "docs/gates/LADDER-V3.md").read_text()
    tiles = [json.loads(row) for row in path.read_text().splitlines()]
    prior = json.loads((root / "outputs/results_ladder_qwen_full_v2.json").read_text())
    ids = list(dict.fromkeys(r["tile_id"].rsplit("_gsd", 1)[0] for r in prior["results"]))
    assert [t["tile"] for t in tiles] == ids and len(ids) == 200
    assert {t["domain"] for t in tiles} == {"rural"}
    expected = tile_questions(tiles)
    for tile in tiles:
        assert len(tile["image_sha256"]) == len(tile["mask_sha256"]) == 64
        assert tile["questions"] == expected[tile["tile"]]
        for row in tile["questions"]:
            if row["type"].startswith("presence:"):
                name = row["type"].split(":")[1]
                index = next(i for i, value in CLASS_NAMES.items() if value == name)
                assert (row["gold"] == "yes") == (index in tile["classes"])
    counts = Counter((r["type"], r["gold"]) for t in tiles for r in t["questions"])
    for index, name in CLASS_NAMES.items():
        n = sum(index in tile["classes"] for tile in tiles)
        task = f"presence:{name}"
        assert counts[task, "yes"] == (min(n, 200 - n) if n < 200 else 200)
        assert counts[task, "no"] == (min(n, 200 - n) if n > 0 else 200)


def test_scoring_aliases_full_label_and_one_class_balanced_accuracy():
    assert canonical_answer(" Buildings. ", ("building", "forest")) == "building"
    assert canonical_answer("Urban", ("building", "forest")) is None
    assert canonical_answer("Yes, 1", ("yes", "no")) is None
    assert canonical_answer("0", ("yes", "no")) == "no"
    draws = np.array([[0, 1], [0, 0], [1, 1]])
    rows = [{"tile": "a", "gold": "yes", "answer": "yes"},
            {"tile": "b", "gold": "no", "answer": "yes"}]
    report, accuracy = metrics(rows, ["a", "b"], draws, ("yes", "no"))
    assert report["balanced_accuracy"] == .5 and report["degenerate"]
    assert accuracy.tolist() == [.5, 1, 0]
    report, _ = metrics(rows[:1], ["a", "b"], draws, ("yes", "no"))
    assert report["balanced_accuracy"] is None and report["balanced_accuracy_ci95"] is None


def test_real_minus_grey_uses_paired_tile_draws():
    rows = [dict(tile=tile, gsd=.3, condition=condition, type="presence:water",
                 gold=gold, answer=gold if condition == "b13_real" else None)
            for tile, gold in (("x", "yes"), ("y", "no"))
            for condition in ("b13_real", "b13_constant")]
    _, contrasts = summarize(rows, ["x", "y"])
    assert all(row["image_dependent"] and row["ci95"] == [1, 1] for row in contrasts)
