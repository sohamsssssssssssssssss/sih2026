"""Small suite loader stubs with a stable sample contract.

Before wiring up or switching a suite, read RECON.md in this directory:
it records per-suite splits, licences (two are non-commercial), and what
is currently blocking each unwired suite.
"""

from eval.suites.rsvqa import load_rsvqa_lr

SUITE_NAMES = ("rsvqa", "vrsbench", "cdvqa", "ladder", "proxy")


def load_suite(name: str, limit: int = 200, full: bool = False) -> list[dict]:
    if name not in SUITE_NAMES:
        raise ValueError(f"Unknown suite: {name}")
    if name == "rsvqa":
        return load_rsvqa_lr(limit=limit, full=full)
    question = f"placeholder question for {name}"
    return [
        {
            "image_paths": [f"data/{name}_placeholder.png"],
            "question": question,
            "expected_answer": f"mock answer for: {question}",
        }
    ]
