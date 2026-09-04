"""Run a registered model against a named evaluation suite.

SAMPLE-SIZE WARNING — read before comparing two runs.

The RSVQA-LR test split is 6957 binary (yes/no) questions to 3047 open-ended,
so --limit draws roughly 70/30 in favour of binary. Open-ended questions are
the minority stratum and small runs land on very few of them: --limit 200
yields only ~61 open-ended questions.

Measured on the frozen Qwen2.5-VL-3B baseline, same config, same matcher:

    metric               --limit 200    full split (n=10004)
    binary_accuracy         0.699           0.6655
    open_accuracy           0.4227          0.1651

Both columns were scored under the pre-daab09b matcher, so the comparison is
like-for-like; open_accuracy is invariant under that change anyway.

binary_accuracy moved 3 points. open_accuracy moved by a factor of 2.5, and
the small-sample figure was optimistic in the direction that flatters us.

The rule: binary_accuracy is stable under small --limit runs. open_accuracy
is NOT. Do not use --limit below ~2000 to compare open-ended performance
between models or checkpoints — at that limit the run draws ~609 open-ended
questions. Headline numbers use --full.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites import SUITE_NAMES, load_suite  # noqa: E402
from eval.ladder import QUESTIONS, RUNGS  # noqa: E402
from orchestrator.registry import get, names  # noqa: E402


def answer_matches(raw: str, expected: str) -> bool:
    raw_l = raw.strip().lower()
    exp_l = expected.strip().lower()
    if raw_l == exp_l:
        return True
    if raw_l == "1" and exp_l == "yes":
        return True
    if re.search(rf"\b{re.escape(exp_l)}\b", raw_l):
        return True
    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    raw_norm = raw_l
    for word, number in word_to_num.items():
        raw_norm = re.sub(rf"\b{word}\b", number, raw_norm)
    if re.search(rf"\b{re.escape(exp_l)}\b", raw_norm):
        return True
    return False


RESULTS_DIR = ROOT / "results"
SUITE_SPLITS = {
    "rsvqa": "official RSVQA-LR test split",
    "ladder": "generated ladder manifest",
}

BINARY_ANSWERS = ("yes", "no")
DEGENERATE_YES_RATE = 0.85
# Below this many open-ended questions, open_accuracy is noise — see the
# module docstring for the measured 2.5x swing that motivates the number.
MIN_OPEN_SAMPLES = 500
# A rung needs at least this many binary questions before its yes-rate is
# meaningful enough to call degenerate. The real ladder has 200 per rung.
MIN_BINARY_FOR_RUNG_GUARD = 30


def _is_binary_gold(expected: str) -> bool:
    return expected.strip().lower() in BINARY_ANSWERS


def summarise(results: list[dict]) -> dict:
    """Aggregate per-sample results, reporting binary and open-ended separately.

    RSVQA-LR is heavily yes/no weighted, so a model that answers "yes" to
    everything scores respectably on the aggregate and looks like a working
    baseline. The split, plus pred_yes_rate_on_binary, is how that gets
    caught. Whether a prediction counts as "yes" is decided by
    answer_matches, so the scorer and the guard can never disagree.
    """
    n = len(results)
    binary = [result for result in results if _is_binary_gold(result["expected_answer"])]
    openended = [
        result for result in results if not _is_binary_gold(result["expected_answer"])
    ]
    yes_predictions = sum(
        int(answer_matches(result["prediction"]["answer"], "yes")) for result in binary
    )
    return {
        "n": n,
        "accuracy": sum(int(result["correct"]) for result in results) / n if n else 0.0,
        "binary_n": len(binary),
        "binary_accuracy": (
            sum(int(result["correct"]) for result in binary) / len(binary)
            if binary
            else 0.0
        ),
        "open_n": len(openended),
        "open_accuracy": (
            sum(int(result["correct"]) for result in openended) / len(openended)
            if openended
            else 0.0
        ),
        "pred_yes_rate_on_binary": (
            yes_predictions / len(binary) if binary else 0.0
        ),
    }


def degenerate(summary: dict) -> str | None:
    """Return a warning string if the run is degenerate, else None.

    A run that trips this is not a weak baseline, it is a non-measurement:
    the accuracy reflects how often the model says "yes", not what it knows.
    """
    if summary["pred_yes_rate_on_binary"] > DEGENERATE_YES_RATE:
        return (
            f"Model answered 'yes' to {summary['pred_yes_rate_on_binary']:.0%} of "
            f"{summary['binary_n']} binary questions "
            f"(threshold {DEGENERATE_YES_RATE:.0%}). This accuracy measures "
            "verbosity, not knowledge — fix prompting before recording it."
        )
    return None


def git_sha() -> str:
    """Short SHA of the tree that produced a result, or "nogit" if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def gpu_name() -> str:
    """Name of the accelerator that produced a result, or "cpu"."""
    try:
        import torch
    except ImportError:
        return "unknown"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


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
    # Gold answers are known before any inference, so this is an exact count
    # rather than a projection. Printed up front: a run too small to compare
    # open-ended performance should be abandoned before it burns GPU time.
    open_n = sum(
        1 for sample in samples if not _is_binary_gold(sample["expected_answer"])
    )
    sampling_warning = None
    if open_n < MIN_OPEN_SAMPLES:
        sampling_warning = (
            f"This run has only {open_n} open-ended questions "
            f"(threshold {MIN_OPEN_SAMPLES}). binary_accuracy will be usable but "
            "open_accuracy will not — it is the headline metric and it is "
            "unstable at this sample size. Raise --limit (~2000 on RSVQA-LR) "
            "or use --full before comparing open-ended performance."
        )
        print(f"WARNING: {sampling_warning}")

    results = []
    correct = 0
    for sample in samples:
        prediction = model.infer(sample["image_paths"], sample["question"])
        is_correct = answer_matches(prediction["answer"], sample["expected_answer"])
        correct += int(is_correct)
        results.append({**sample, "prediction": prediction, "correct": is_correct})

    summary = summarise(results)
    warning = degenerate(summary)
    report = {
        "model": args.model,
        "suite": args.suite,
        "split": SUITE_SPLITS.get(args.suite, "placeholder"),
        "config": {
            "model": args.model,
            "suite": args.suite,
            "limit": args.limit,
            "full": args.full,
        },
        "git_sha": git_sha(),
        "gpu": gpu_name(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": summary["accuracy"],
        "n_samples": summary["n"],
        "summary": summary,
        "warning": warning,
        "sampling_warning": sampling_warning,
        "results": results,
    }
    if args.suite == "ladder":
        # Per-rung stratification matters more here than anywhere else. A model
        # that collapses to always-"yes" at coarse GSD scores the binary base
        # rate, which reads on an aggregate curve as accuracy RECOVERING at
        # lower resolution — physically impossible, and the artefact that makes
        # a ladder plot lie. Running the same guard per rung catches it; running
        # it only over the whole run does not, because the good rungs dilute the
        # degenerate ones.
        per_rung = {}
        for rung in RUNGS:
            rung_results = [result for result in results if result["gsd"] == rung]
            if not rung_results:
                per_rung[f"{rung:.1f}"] = {"accuracy": 0.0, "n": 0}
                continue
            rung_summary = summarise(rung_results)
            # A yes-rate over a handful of questions is noise, not evidence of
            # collapse; require a floor before calling a rung degenerate.
            rung_warning = (
                degenerate(rung_summary)
                if rung_summary["binary_n"] >= MIN_BINARY_FOR_RUNG_GUARD
                else None
            )
            per_rung[f"{rung:.1f}"] = {**rung_summary, "warning": rung_warning}
        report["per_rung"] = per_rung
        degenerate_rungs = [g for g, v in per_rung.items() if v.get("warning")]
        report["degenerate_rungs"] = degenerate_rungs
        if degenerate_rungs:
            print(
                "WARNING: rungs "
                + ", ".join(f"{g}m" for g in degenerate_rungs)
                + " are degenerate (always-yes on binary questions). Their "
                "accuracy reflects the answer prior, not resolution. Do not "
                "read the aggregate curve across these points."
            )
    payload = json.dumps(report, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    # A number that exists only under /kaggle/working dies with the session.
    # The in-repo copy is the audit trail; it is what gets committed.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace("-", "").replace(":", "").split(".")[0]
    archived = RESULTS_DIR / f"{args.model}__{args.suite}__{stamp}Z.json"
    archived.write_text(payload, encoding="utf-8")
    print(f"[saved] {args.out}")
    print(f"[saved] {archived.relative_to(ROOT)}  <- commit this")

    if warning:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
