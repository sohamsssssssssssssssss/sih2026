"""Local contract check that never loads Qwen model weights."""

import sys
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.suites.rsvqa import load_rsvqa_lr  # noqa: E402
from models.qwen_vl import QwenVLModel  # noqa: E402
from orchestrator.registry import get  # noqa: E402


def main() -> int:
    samples = load_rsvqa_lr(limit=2)
    assert len(samples) == 2
    assert all(set(sample) == {"image_paths", "question", "expected_answer"} for sample in samples)
    assert all(isinstance(sample["question"], str) for sample in samples)
    assert all(isinstance(sample["expected_answer"], str) for sample in samples)
    with Image.open(samples[0]["image_paths"][0]) as image:
        assert image.size == (256, 256)
    model = QwenVLModel()
    assert get("qwen2.5vl-3b") is not None
    with patch.object(model, "_generate_answer", return_value="urban") as mocked_forward:
        prediction = model.infer(samples[0]["image_paths"], samples[0]["question"])
    mocked_forward.assert_called_once()
    assert mocked_forward.call_args.args[1].endswith(
        " Answer with a single word or number only. No explanation."
    )
    assert prediction == {"answer": "urban", "confidence": 1.0, "evidence": []}
    assert model._model is None, "dry run unexpectedly loaded model weights"
    print("RSVQA LOCAL DRY RUN PASSED: real 256x256 sample, registry, mocked inference contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
