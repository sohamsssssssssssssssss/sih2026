"""Small suite loader stubs with a stable sample contract."""

SUITE_NAMES = ("rsvqa", "vrsbench", "cdvqa", "ladder", "proxy")


def load_suite(name: str) -> list[dict]:
    if name not in SUITE_NAMES:
        raise ValueError(f"Unknown suite: {name}")
    question = f"placeholder question for {name}"
    return [
        {
            "image_paths": [f"data/{name}_placeholder.png"],
            "question": question,
            "expected_answer": f"mock answer for: {question}",
        }
    ]
