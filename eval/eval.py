"""Run a registered model against a named evaluation suite."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites import SUITE_NAMES, load_suite  # noqa: E402
from orchestrator.registry import get, names  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=names())
    parser.add_argument("--suite", required=True, choices=SUITE_NAMES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    model = get(args.model)
    samples = load_suite(args.suite)
    results = []
    correct = 0
    for sample in samples:
        prediction = model.infer(sample["image_paths"], sample["question"])
        is_correct = prediction["answer"].strip().lower() == sample["expected_answer"].strip().lower()
        correct += int(is_correct)
        results.append({**sample, "prediction": prediction, "correct": is_correct})

    report = {
        "model": args.model,
        "suite": args.suite,
        "accuracy": correct / len(samples) if samples else 0.0,
        "n_samples": len(samples),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
