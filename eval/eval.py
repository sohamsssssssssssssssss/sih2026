"""Run a registered model against a named evaluation suite."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites import SUITE_NAMES, load_suite  # noqa: E402
from eval.ladder import QUESTIONS, RUNGS  # noqa: E402
from orchestrator.registry import get, names  # noqa: E402


def load_ladder_samples(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ladder manifest not found at {manifest_path}; run python3 eval/ladder.py first"
        )
    samples = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for question in QUESTIONS:
            samples.append(
                {
                    "image_paths": [str(ROOT / record["image_path"])],
                    "question": question,
                    "expected_answer": record["answers"][question],
                    "gsd": float(record["gsd"]),
                    "tile_id": record["id"],
                }
            )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=names())
    parser.add_argument("--suite", required=True, choices=SUITE_NAMES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    model = get(args.model)
    samples = (
        load_ladder_samples(ROOT / "data" / "ladder" / "manifest.jsonl")
        if args.suite == "ladder"
        else load_suite(args.suite, limit=args.limit, full=args.full)
    )
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
    if args.suite == "ladder":
        per_rung = {}
        for rung in RUNGS:
            rung_results = [result for result in results if result["gsd"] == rung]
            rung_correct = sum(int(result["correct"]) for result in rung_results)
            per_rung[f"{rung:.1f}"] = {
                "accuracy": rung_correct / len(rung_results) if rung_results else 0.0,
                "n": len(rung_results),
            }
        report["per_rung"] = per_rung
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
