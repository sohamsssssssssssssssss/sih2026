import pytest

from scripts.check_b13_app_parity import frozen_questions, summarise


def test_selection_is_ten_per_type_in_reference_order_and_reports_missing_predictions():
    rows = [{"sample_id": f"{kind}-{i}", "question_type": kind} for i in range(12)
            for kind in ("count", "comp", "presence", "rural_urban")]
    reference = {"manifest_identity": "1eb622b1ca4bec1d2d849f049623f015f3e015c423ce8fab146e91e5b86f43ea",
                 "conditions": {"adapter_real": {"records": rows}}}
    chosen = frozen_questions(reference)
    assert len(chosen) == 40
    assert [row["sample_id"] for row in chosen[:10]] == [f"comp-{i}" for i in range(10)]
    assert not any(row["sample_id"].endswith(("-10", "-11")) for row in chosen)
    for row in chosen:
        row.update(kaggle_correct=True, seconds=0.1)
    summary = summarise(chosen)
    assert summary["count"] == {"attempted": 10, "completed": 0, "agreement": None,
                                 "accuracy": None, "kaggle_accuracy": 1.0, "seconds_per_question": None}
    chosen[0].update(prediction="yes", agrees=False, correct=True)
    assert summarise(chosen)["comp"]["agreement"] == 0
    reference["manifest_identity"] = "wrong"
    with pytest.raises(ValueError, match="frozen"):
        frozen_questions(reference)
