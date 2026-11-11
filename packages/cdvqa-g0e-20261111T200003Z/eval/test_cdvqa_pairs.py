"""CPU-only checks for the CDVQA two-image pilot. No model weights, no images, no GPU."""
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from eval.cdvqa_pairs import (CONDITIONS, QUESTION_TYPES, assert_no_pair_leak, condition_example,
                              load_pair_manifest, load_vocabulary, score, summarise)
from training.remote_sensing import QwenVQACollator, TrainingExample, messages

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data/manifests/cdvqa"
VOCAB = {"yes_no": ["yes", "no"], "buildings": ["buildings", "nvg_surface"]}


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"


class Processor:
    """Fake processor: 'full' text encodes prompt+answer, 'prompt' encodes the prompt only."""
    tokenizer = Tokenizer()

    def __init__(self):
        self.images_seen = []

    def apply_chat_template(self, conversation, **kwargs):
        return "full" if conversation[-1]["role"] == "assistant" else "prompt"

    def __call__(self, text, images, videos=None, **kwargs):
        self.images_seen.append(list(images))
        ids = [4, 5, 6, 7, 0] if text[0] == "full" else [4, 5, 0]
        return {"input_ids": torch.tensor([ids])}


def fake_process_vision_info(conversation):
    """Mimic qwen_vl_utils: return the image inputs of the conversation, in prompt order."""
    images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
    return images, None


def example(tmp_path, with_after=True):
    before = tmp_path / "im1" / "1.png"
    after = tmp_path / "im2" / "1.png"
    return TrainingExample(before, "Are there more buildings?", "yes", "CDVQA", "1", "train", "s1",
                           after_image_path=after if with_after else None, choices=("yes", "no"))


def test_collator_masks_answer_and_feeds_both_images_in_order(tmp_path):
    processor = Processor()
    item = example(tmp_path)
    batch = QwenVQACollator(processor, 392, fake_process_vision_info)([item])
    assert batch["labels"].tolist() == [[-100, -100, 6, 7, -100]]
    assert processor.images_seen[0] == [item.image_path.as_uri(), item.after_image_path.as_uri()]


def test_pair_messages_order_and_size(tmp_path):
    content = messages(example(tmp_path), 392, include_target=False)[0]["content"]
    images = [c for c in content if c["type"] == "image"]
    assert [c["image"] for c in images] == [
        (tmp_path / "im1" / "1.png").as_uri(), (tmp_path / "im2" / "1.png").as_uri()]
    assert all(c["resized_height"] == c["resized_width"] == 392 for c in images)
    text = content[-1]["text"]
    assert content[-1]["type"] == "text" and "first image is the earlier" in text
    assert "Are there more buildings? Allowed answers: yes, no." in text


def test_single_image_messages_unchanged(tmp_path):
    content = messages(example(tmp_path, with_after=False), 392, include_target=False)[0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]
    assert content[0] == {"type": "image", "image": (tmp_path / "im1" / "1.png").as_uri(),
                          "resized_height": 392, "resized_width": 392}


def test_condition_pairs(tmp_path):
    item = example(tmp_path)
    before, after = item.image_path, item.after_image_path
    real = condition_example(item, "real")
    assert (real.image_path, real.after_image_path) == (before, after)
    identical = condition_example(item, "identical_pair")
    assert (identical.image_path, identical.after_image_path) == (before, before)
    swapped = condition_example(item, "swapped_order")
    assert (swapped.image_path, swapped.after_image_path) == (after, before)
    assert set(CONDITIONS) == {"real", "identical_pair", "swapped_order"}
    with pytest.raises(ValueError):
        condition_example(example(tmp_path, with_after=False), "swapped_order")


def test_scoring_normalisation_and_malformed(tmp_path):
    vocab = {t: frozenset(v) for t, v in VOCAB.items()}
    assert score(" YES ", "yes", "yes_no", vocab) == {"correct": True, "malformed": False}
    assert score("no", "yes", "yes_no", vocab) == {"correct": False, "malformed": False}
    assert score("Maybe", "yes", "yes_no", vocab) == {"correct": False, "malformed": True}
    assert score("NVG_Surface", "nvg_surface", "buildings", vocab)["correct"] is True
    with pytest.raises(ValueError):
        score("yes", "bogus", "yes_no", vocab)


def test_summary_counts_malformed_and_macro_over_all_types():
    records = [{"question_type": t, "correct": False, "malformed": t == "change_ratio"}
               for t in QUESTION_TYPES] * 2
    result = summarise(records)
    assert result["n"] == 16 and result["malformed_n"] == 2
    assert result["per_type"]["change_ratio"]["malformed_n"] == 2
    assert result["macro_accuracy"] == pytest.approx(0.0)


def write_fixture(tmp_path, rows, vocab=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    images = tmp_path / "images"
    for row in rows:
        for name in (row["before_image"], row["after_image"]):
            (images / name).parent.mkdir(parents=True, exist_ok=True)
            (images / name).write_bytes(b"x")
    provenance = tmp_path / "provenance.json"
    types = vocab or {t: ["yes", "no"] for t in QUESTION_TYPES}
    provenance.write_text(json.dumps({"answer_vocabulary": types, "answer_choices": types}))
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest, images, provenance


def row(**overrides):
    base = {"id": "cdvqa-test1-1", "split": "test1", "question_type": "change_or_not",
            "question": "Did it change?", "answer": "yes", "before_image": "im1/1.png",
            "after_image": "im2/1.png", "pair_id": "1", "question_id": 1}
    base.update(overrides)
    return base


def test_loader_accepts_valid_rows_and_rejects_bad_ones(tmp_path):
    manifest, images, provenance = write_fixture(tmp_path, [row()])
    examples, pairs = load_pair_manifest(manifest, images, provenance)
    assert examples[0].after_image_path == (images / "im2" / "1.png").resolve()
    assert pairs == {"1"}

    for bad in (row(answer="maybe"), row(question_type="unknown"), row(pair_id="")):
        manifest, images, provenance = write_fixture(tmp_path / str(abs(hash(str(bad)))), [bad])
        with pytest.raises(ValueError):
            load_pair_manifest(manifest, images, provenance)

    manifest, images, provenance = write_fixture(tmp_path / "missing", [row()])
    (images / "im2" / "1.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_pair_manifest(manifest, images, provenance)

    manifest, images, provenance = write_fixture(tmp_path / "dup", [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_pair_manifest(manifest, images, provenance)


def test_leakage_assert_detects_shared_pair(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps({"pair_id": "7"}) + "\n")
    test.write_text(json.dumps({"pair_id": "8"}) + "\n")
    assert_no_pair_leak(train, test)
    test.write_text(json.dumps({"pair_id": "7"}) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        assert_no_pair_leak(train, test)


def test_committed_manifests_are_frozen_and_leak_free():
    provenance = MANIFESTS / "provenance.json"
    vocab = load_vocabulary(provenance)
    assert_no_pair_leak(MANIFESTS / "train-8k.v1.jsonl", MANIFESTS / "eval-test1-400.v1.jsonl")
    assert_no_pair_leak(MANIFESTS / "train-8k.v1.jsonl", MANIFESTS / "eval-test2-400.v1.jsonl")
    for name, expected_rows in (("train-8k.v1.jsonl", 8000), ("eval-test1-400.v1.jsonl", 400),
                                ("eval-test2-400.v1.jsonl", 400)):
        rows = [json.loads(line) for line in (MANIFESTS / name).read_text().splitlines()]
        assert len(rows) == expected_rows
        assert all(r["answer"].strip().casefold() in vocab[r["question_type"]] for r in rows)
        per_type = Counter(r["question_type"] for r in rows)
        assert set(per_type) == set(QUESTION_TYPES)
        if name.startswith("eval"):
            assert all(per_type[t] == 50 for t in QUESTION_TYPES)
            assert len({r["pair_id"] for r in rows}) <= 100
        else:
            assert all(per_type[t] == 1000 for t in QUESTION_TYPES)
