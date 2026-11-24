import numpy as np
import pytest

from kaggle.run_grounding_baselines import GRID, parse_box
from tools.score_grounding_v1 import (
    control_pass, dino, draws_for, iou, metrics, paired, positive_ci, select_threshold,
)

GOLD = [.25, .25, .75, .75]


def test_iou_edges():
    assert iou(GOLD, GOLD) == 1
    assert iou([0, 0, .25, .25], GOLD) == 0
    assert iou([.25, .25, .5, .75], GOLD) == .5
    assert metrics([(.5, "valid")])["acc@0.5"] == 1


@pytest.mark.parametrize("raw", ["", "[0,0,0,2]", "[0,0,1001,2]", "[0,0,1,2,3]", "[0,0,1.0,2]", "[0,0,NaN,2]"])
def test_malformed(raw):
    assert parse_box(raw) is None
    assert iou(parse_box(raw), GOLD) == 0


def grid(winners):
    return {"id": "x", "grid": [
        {"box_threshold": b, "text_threshold": t, "detections":
         [{"box": GOLD if (b, t) in winners else [0, 0, .1, .1], "score": .8},
          {"box": GOLD, "score": .8}]}
        for b, t in GRID]}


def test_select_only_and_stable_ties():
    rows = [{"id": "x", "gold": GOLD}]
    selected = select_threshold(rows, {"x": grid(GRID[:2])})
    assert selected == GRID[1]
    assert iou(dino(grid(GRID[-1:]), selected)[0], GOLD) == 0
    assert select_threshold(rows, {"x": grid(GRID)}) == GRID[-1]


def test_paired_determinism_and_controls():
    vectors = [np.array([1, 0, 1]), np.array([1, 1])]
    first = paired(vectors, draws_for([3, 2]))
    assert first == paired(vectors, draws_for([3, 2]))
    assert control_pass(first)
    assert not control_pass(paired([[0], [0]], draws_for([1, 1])))
    assert not positive_ci({"delta": 1, "ci95": [0, 1]})
