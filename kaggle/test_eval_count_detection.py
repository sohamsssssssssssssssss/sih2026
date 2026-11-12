"""Pure scoring checks only: the CUDA evaluation entry point is never invoked."""
import pytest

from kaggle import eval_count_detection as evaluation


def test_count_metrics():
    assert evaluation.metrics([{"expected": 2, "prediction": 2},
                               {"expected": 4, "prediction": 1}]) == {
        "n": 2, "accuracy": 0.5, "mean_absolute_error": 1.5}
    with pytest.raises(ValueError, match="empty"):
        evaluation.metrics([])


def test_manifest_hash_is_required_before_rows_are_used(tmp_path):
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text('{"question_type":"count"}')
    with pytest.raises(ValueError, match="frozen"):
        evaluation.count_rows(manifest)
