from pathlib import Path

import pytest

from scripts.evaluate_remote_sensing_adapter import evaluate, parse_args, select_subset
from training.remote_sensing import TrainingExample

SUBSET = {
    "split": "validation",
    "full_split_type_counts": {"presence": 2, "count": 1},
    "leakage_flagged_image_ids": [9],
    "samples": [
        {"sample_id": "validation-2", "type": "count", "image_id": 9},
        {"sample_id": "validation-1", "type": "presence", "image_id": 1},
    ],
}


def example(sample_id: str, response: str, split: str = "validation") -> TrainingExample:
    return TrainingExample(Path("x.tif"), "q?", response, "RSVQA-LR", "fixture", split, sample_id)


def test_subset_selection_keeps_subset_order_and_rejects_missing_samples():
    examples = [example("validation-1", "yes"), example("validation-2", "3"), example("test-2", "3", "test")]

    selected = select_subset(examples, SUBSET)

    assert [item.sample_id for item, _ in selected] == ["validation-2", "validation-1"]
    with pytest.raises(ValueError, match="not in the validation manifest"):
        select_subset(examples[:1], SUBSET)


def test_evaluation_scores_every_sample_and_reports_footprint_disjoint_summary():
    selected = select_subset([example("validation-1", "yes"), example("validation-2", "3")], SUBSET)

    result = evaluate(selected, SUBSET, lambda item: {"validation-1": "Yes", "validation-2": "4"}[item.sample_id])

    assert result["summary"]["strict_accuracy"] == 0.5
    assert result["summary"]["per_type"]["count"]["count_bin_accuracy"] == 1.0
    assert result["footprint_disjoint_summary"]["n"] == 1
    assert [row["image_id"] for row in result["results"]] == [9, 1]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--skip-base"], "--skip-base needs --adapter-path"),
        (["--image-size", "250"], "multiple of 28"),
    ],
)
def test_invalid_arguments_are_refused(tmp_path: Path, capsys, extra, message):
    base = ["--model-path", "m", "--dataset-manifest", "d", "--subset", "s", "--out", str(tmp_path / "r.json")]

    with pytest.raises(SystemExit):
        parse_args(base + extra)
    assert message in capsys.readouterr().err


def test_existing_report_is_never_overwritten(tmp_path: Path, capsys):
    (tmp_path / "r.json").write_text("{}")

    with pytest.raises(SystemExit):
        parse_args(["--model-path", "m", "--dataset-manifest", "d", "--subset", "s", "--out", str(tmp_path / "r.json")])
    assert "never overwritten" in capsys.readouterr().err
