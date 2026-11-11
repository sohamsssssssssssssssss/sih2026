"""CDVQA two-image pilot: manifest validation, eval conditions and strict answer scoring (CPU only)."""
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from training.remote_sensing import TrainingExample

CONDITIONS = ("real", "identical_pair", "swapped_order")
QUESTION_TYPES = ("change_or_not", "increase_or_not", "decrease_or_not", "smallest_change",
                  "largest_change", "change_to_what", "change_ratio", "change_ratio_types")
REQUIRED_FIELDS = ("id", "split", "question_type", "question", "answer", "before_image",
                   "after_image", "pair_id", "question_id")


def normalise(text):
    return text.strip().casefold()


def load_vocabulary(provenance):
    """Answer vocabulary per type, as recorded by the manifest builder from the Train split."""
    vocab = json.loads(Path(provenance).read_text())["answer_vocabulary"]
    if set(vocab) != set(QUESTION_TYPES):
        raise ValueError("Vocabulary types differ from the CDVQA question types")
    return {t: frozenset(values) for t, values in vocab.items()}


def load_pair_manifest(manifest, image_root, provenance):
    """Validate a CDVQA pair manifest and return TrainingExample rows with after_image_path set.

    Every image must exist under image_root; every answer must be inside its type's vocabulary.
    """
    vocab = load_vocabulary(provenance)
    root = Path(image_root).resolve()
    examples, ids, pairs = [], set(), set()
    for number, raw in enumerate(Path(manifest).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        missing = [f for f in REQUIRED_FIELDS if not isinstance(row.get(f), (str, int)) or row.get(f) == ""]
        if missing:
            raise ValueError(f"line {number}: missing fields {missing}")
        if row["question_type"] not in vocab:
            raise ValueError(f"line {number}: unknown question_type {row['question_type']!r}")
        if normalise(row["answer"]) not in vocab[row["question_type"]]:
            raise ValueError(f"line {number}: answer outside vocabulary: {row['answer']!r}")
        if row["id"] in ids:
            raise ValueError(f"line {number}: duplicate id {row['id']}")
        ids.add(row["id"]); pairs.add(row["pair_id"])
        before, after = (root / row["before_image"]).resolve(), (root / row["after_image"]).resolve()
        for path in (before, after):
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"line {number}: image missing or outside root: {path}")
        examples.append(TrainingExample(
            image_path=before, instruction=row["question"], response=row["answer"],
            dataset="CDVQA", source=row["pair_id"], split=row["split"], sample_id=row["id"],
            after_image_path=after))
    if not examples:
        raise ValueError("Pair manifest contains no examples")
    return examples, pairs


def assert_no_pair_leak(train_manifest, eval_manifest):
    """Raise if any image pair appears in both the training and evaluation manifests."""
    def pairs(path):
        return {json.loads(line)["pair_id"] for line in Path(path).read_text().splitlines() if line.strip()}
    shared = pairs(train_manifest) & pairs(eval_manifest)
    if shared:
        raise ValueError(f"train/eval image-pair leakage: {len(shared)} pairs, e.g. {sorted(shared)[:3]}")
    return 0


def condition_example(example, condition):
    """Build the prompt image pair for an eval condition.

    real: [before, after]; identical_pair: [before, before]; swapped_order: [after, before].
    """
    if condition == "real":
        return example
    if example.after_image_path is None:
        raise ValueError("Pair conditions need after_image_path")
    if condition == "identical_pair":
        return replace(example, after_image_path=example.image_path)
    if condition == "swapped_order":
        return replace(example, image_path=example.after_image_path, after_image_path=example.image_path)
    raise ValueError(f"unknown condition {condition!r}")


def score(prediction, answer, question_type, vocab):
    """Exact match after strip+casefold, against the type's vocabulary; anything else is malformed."""
    gold = normalise(answer)
    if gold not in vocab[question_type]:
        raise ValueError(f"gold answer outside vocabulary: {answer!r}")
    pred = normalise(prediction)
    malformed = pred not in vocab[question_type]
    return {"correct": (not malformed) and pred == gold, "malformed": malformed}


def summarise(records):
    """Per-type accuracy and malformed counts; macro = equal-weight mean over the 8 types."""
    counts = Counter(r["question_type"] for r in records)
    per_type = {}
    for t in QUESTION_TYPES:
        subset = [r for r in records if r["question_type"] == t]
        per_type[t] = {"n": counts[t],
                       "accuracy": (sum(r["correct"] for r in subset) / len(subset)) if subset else None,
                       "malformed_n": sum(r["malformed"] for r in subset)}
    macro = None
    if all(per_type[t]["accuracy"] is not None for t in QUESTION_TYPES):
        macro = sum(per_type[t]["accuracy"] for t in QUESTION_TYPES) / len(QUESTION_TYPES)
    return {"n": len(records), "per_type": per_type, "macro_accuracy": macro,
            "malformed_n": sum(r["malformed"] for r in records)}
