"""CDVQA two-image pilot: manifest validation, eval conditions and strict answer scoring (CPU only)."""
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.remote_sensing import TrainingExample

CONDITIONS = ("real", "identical_pair", "swapped_order")
SCORING_VERSION = "cdvqa-pair-casefold-vocabulary-v1"
PAIR_COMPARISONS = (("adapter_real", "base_real"), ("adapter_real", "adapter_identical"),
                    ("adapter_real", "adapter_swapped"))
PAIR_CONDITION_KINDS = {"base_real": "real", "adapter_real": "real",
                        "adapter_identical": "identical_pair", "adapter_swapped": "swapped_order"}
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_choices(provenance):
    """Raw (cased) answer strings per type, as shown to the model in pair prompts."""
    choices = json.loads(Path(provenance).read_text())["answer_choices"]
    return {t: tuple(values) for t, values in choices.items()}


def load_pair_manifest(manifest, image_root, provenance):
    """Validate a CDVQA pair manifest and return TrainingExample rows with after_image_path set.

    Every image must exist under image_root; every answer must be inside its type's vocabulary.
    """
    vocab = load_vocabulary(provenance)
    choices = load_choices(provenance)
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
            after_image_path=after, choices=choices[row["question_type"]]))
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


def majority_accuracy(rows, majority):
    """Accuracy of a constant per-type answer over manifest rows (casefolded comparison)."""
    per_type = {}
    for t in QUESTION_TYPES:
        subset = [r for r in rows if r["question_type"] == t]
        correct = sum(normalise(r["answer"]) == majority[t] for r in subset)
        per_type[t] = {"n": len(subset), "accuracy": correct / len(subset) if subset else None}
    macro = None
    if all(v["accuracy"] is not None for v in per_type.values()):
        macro = sum(v["accuracy"] for v in per_type.values()) / len(QUESTION_TYPES)
    return {"per_type": per_type, "macro_accuracy": macro, "malformed_n": 0}


def evaluate_conditions(rows, examples, answer_fn, vocab, conditions):
    """Run each condition over the same rows and examples, in the same order.

    rows[i] is the manifest row for examples[i]. answer_fn(example) -> prediction string; the
    example it receives already carries the condition's image pair in prompt order.
    Returns {condition_name: [record, ...]} with exact-match scoring (malformed counted, not dropped).
    """
    if len(rows) != len(examples):
        raise ValueError("Row/example alignment mismatch")
    result = {}
    for name, condition in conditions.items():
        records = []
        for row, example in zip(rows, examples):
            if example.sample_id != row["id"]:
                raise ValueError("Example/row identity mismatch")
            prediction = answer_fn(condition_example(example, condition))
            records.append({"sample_id": row["id"], "question_id": row["question_id"],
                            "pair_id": row["pair_id"], "question_type": row["question_type"],
                            "question": row["question"], "expected": row["answer"],
                            "prediction": prediction,
                            **score(prediction, row["answer"], row["question_type"], vocab)})
        result[name] = records
    return result


def paired_differences(conditions, *, seed=17, repetitions=2000):
    """Resample image pairs with replacement; QA-weighted within type; equal-weight 8-type macro."""
    reference = conditions[next(iter(conditions))]
    identity = lambda r: (r["sample_id"], r["pair_id"], r["question_type"], r["expected"])
    for records in conditions.values():
        if [identity(r) for r in records] != [identity(r) for r in reference]:
            raise ValueError("Condition alignment mismatch")
    pairs = sorted({r["pair_id"] for r in reference})
    index = {p: i for i, p in enumerate(pairs)}
    types = QUESTION_TYPES
    counts = np.zeros((len(pairs), len(types)))
    correct = {name: np.zeros_like(counts) for name in conditions}
    for j, r in enumerate(reference):
        i, t = index[r["pair_id"]], types.index(r["question_type"])
        counts[i, t] += 1
        for name in conditions:
            correct[name][i, t] += conditions[name][j]["correct"]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pairs), size=(repetitions, len(pairs)))
    denominators = counts[draws].sum(axis=1)
    result = {}
    for a, b in PAIR_COMPARISONS:
        if a not in conditions or b not in conditions:
            continue
        delta = correct[a] - correct[b]
        totals = counts.sum(axis=0)
        point = np.divide(delta.sum(axis=0), totals, out=np.full(len(types), np.nan), where=totals > 0)
        boot = np.divide(delta[draws].sum(axis=1), denominators,
                         out=np.full_like(denominators, np.nan), where=denominators > 0)

        def interval(values):
            valid = values[np.isfinite(values)]
            return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) >= 2 else None,
                    "valid_replicates": int(len(valid))}
        per_type = {t: {"difference": float(point[k]) if np.isfinite(point[k]) else None,
                        **interval(boot[:, k])} for k, t in enumerate(types)}
        macro_boot = boot.mean(axis=1)
        result[f"{a}_minus_{b}"] = {"per_type": per_type,
            "macro": {"difference": float(point.mean()) if np.isfinite(point).all() else None,
                      **interval(macro_boot)}}
    return {"method": "paired percentile bootstrap of image pairs; QA-weighted within each type; equal-weight 8-type macro",
            "seed": seed, "repetitions": repetitions, "pair_clusters": len(pairs), "comparisons": result}
