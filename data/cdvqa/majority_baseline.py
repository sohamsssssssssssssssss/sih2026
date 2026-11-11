"""CPU majority-answer baseline per question type, learned from the FULL Train split.

Run: python3.11 data/cdvqa/majority_baseline.py
Writes results/cdvqa-g0/majority-baseline.json.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_manifests import OUT, TYPES, load  # noqa: E402

RESULTS = HERE.parents[1] / "results/cdvqa-g0"


def majority_table():
    counts = defaultdict(Counter)
    for row in load("Train")[0]:
        counts[row["question_type"]][row["answer"].strip().casefold()] += 1
    # Ties broken by lexicographic order so the result is reproducible.
    return {t: min(counts[t], key=lambda a: (-counts[t][a], a)) for t in TYPES}


def evaluate(manifest, majority):
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    per_type = {}
    for t in TYPES:
        subset = [r for r in rows if r["question_type"] == t]
        correct = sum(r["answer"].strip().casefold() == majority[t] for r in subset)
        per_type[t] = {"n": len(subset), "correct": correct, "accuracy": correct / len(subset)}
    return {"n": len(rows), "per_type": per_type,
            "macro_accuracy": sum(v["accuracy"] for v in per_type.values()) / len(TYPES),
            "malformed_n": 0}


def main():
    majority = majority_table()
    report = {"baseline": "majority answer per question type, learned from full Train split (65,967 QA)",
              "majority_answer": majority, "eval": {}}
    for name in ("eval-test1-400.v1", "eval-test2-400.v1"):
        report["eval"][name] = evaluate(OUT / f"{name}.jsonl", majority)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "majority-baseline.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, result in report["eval"].items():
        print(name, f"macro={result['macro_accuracy']:.4f}")
        for t in TYPES:
            v = result["per_type"][t]
            print(f"  {t:20s} n={v['n']:3d} majority={majority[t]:16s} acc={v['accuracy']:.4f}")


if __name__ == "__main__":
    main()
