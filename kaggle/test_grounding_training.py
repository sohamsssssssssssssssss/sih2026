from pathlib import Path

import pytest

from kaggle.build_grounding_baseline_pack import verify_train_separation
from kaggle.run_grounding_baselines import digest
from kaggle.train_grounding_adapter import seal_tree, training_text, verify_tree


@pytest.mark.parametrize("parent,image,checksum", [("held", "other", "b"), ("train", "held", "b"), ("train", "other", "a")])
def test_train_rejects_held_out_images(parent, image, checksum):
    with pytest.raises(ValueError, match="overlaps eval/select"):
        verify_train_separation([{"parent": parent, "image": image}],
                                [{"parent": "held", "image": "held"}], [checksum], ["a"])


def test_resume_receipt(tmp_path: Path):
    state = tmp_path / "trainer_state.json"
    state.write_text('{"global_step": 4}')
    identity = {"config_sha256": "fixture"}
    pin = digest(seal_tree(tmp_path, identity))
    assert verify_tree(tmp_path, pin) == identity
    state.write_text('{"global_step": 5}')
    with pytest.raises(ValueError, match="artifact file mismatch"):
        verify_tree(tmp_path, pin)


def test_native_grounding_target():
    prompt, answer = training_text({"id": "example", "task": "grounding", "expression": "hangar",
                                   "gold": [.2, .3, .7, .8]})
    assert "hangar" in prompt
    assert answer == "[200, 300, 700, 800]"
