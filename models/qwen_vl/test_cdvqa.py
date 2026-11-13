"""CDVQA contract: question typing on real CDVQA strings, vocabulary checks, prompt identity with the package."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from models.qwen_vl import cdvqa

MANIFESTS = cdvqa.ROOT / "data/manifests/cdvqa"
EVAL = ("eval-test1-400.v1.jsonl", "eval-test2-400.v1.jsonl")


def rows(*names):
    return [json.loads(line) for name in names for line in (MANIFESTS / name).read_text().splitlines() if line.strip()]


def test_classifier_types_every_frozen_eval_and_train_question():
    sample = rows(*EVAL, "train-8k.v2.jsonl")
    assert len(sample) == 8800
    wrong = [(r["question"], r["question_type"]) for r in sample if cdvqa.question_type(r["question"]) != r["question_type"]]
    assert wrong == []


@pytest.mark.parametrize("question", [
    "What changed between these two images?", "How many buildings were built?",
    "Did the forest burn?", "Have the areas of roads changed?", "Describe the scene.",
])
def test_non_template_questions_are_unsupported(question):
    assert cdvqa.question_type(question) is None


@pytest.mark.parametrize("raw,qtype,status", [
    ("yes", "change_or_not", "ok"), (" Yes ", "decrease_or_not", "ok"),
    ("NVG_surface", "change_to_what", "ok"), ("0_to_10", "change_ratio", "ok"),
    ("80_to_90", "change_ratio_types", "malformed"),  # outside that type's Train vocabulary
    ("probably yes", "change_or_not", "malformed"), ("", "largest_change", "malformed"),
    ("10%", "change_ratio", "malformed"),
])
def test_answers_are_checked_against_the_type_vocabulary(raw, qtype, status):
    assert cdvqa.validate(raw, qtype) == status


def test_pair_refusals(tmp_path):
    def image(name, size):
        path = tmp_path / name
        Image.new("RGB", size).save(path)
        return str(path)
    a, b, wide = image("a.png", (100, 100)), image("b.png", (120, 90)), image("c.png", (250, 100))
    assert cdvqa.pair_refusal([a, b]) is None
    assert "exactly two" in cdvqa.pair_refusal([a])
    assert "sizes differ" in cdvqa.pair_refusal([a, wide])
    assert "Ground sample" in cdvqa.pair_refusal([a, b], (0.5, 2.0))
    assert cdvqa.pair_refusal([a, b], (0.5, None)) is None


PACKAGE_PROMPTS = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from eval.cdvqa_pairs import load_pair_manifest
from training.remote_sensing import messages
root = Path(sys.argv[1])
out = []
for name in sys.argv[3:]:
    examples, _ = load_pair_manifest(root / "data/manifests/cdvqa" / name, sys.argv[2],
                                     root / "data/manifests/cdvqa/provenance.json")
    out += [[e.instruction, str(e.image_path), str(e.after_image_path), json.dumps(messages(e, 392, False))]
            for e in examples]
print(json.dumps(out))
"""


def test_app_prompt_is_byte_identical_to_the_package_eval_prompt(tmp_path):
    for row in rows(*EVAL):
        for key in ("before_image", "after_image"):
            (tmp_path / row[key]).parent.mkdir(exist_ok=True)
            (tmp_path / row[key]).touch()
    package = cdvqa.ROOT / cdvqa.config()["package"]
    # Separate interpreter: the package's own loader + messages(), with no app code imported.
    output = subprocess.run([sys.executable, "-c", PACKAGE_PROMPTS, str(package), str(tmp_path), *EVAL],
                            capture_output=True, text=True, check=True, cwd=tmp_path).stdout
    expected = json.loads(output)
    assert len(expected) == 800
    for question, before, after, prompt in expected:
        app = cdvqa.messages(before, after, question, cdvqa.question_type(question))
        assert json.dumps(app) == prompt
    text = json.loads(expected[0][3])[0]["content"][2]["text"]
    assert text.startswith(cdvqa.package().PAIR_ORDER_INSTRUCTION)
    assert text.endswith(cdvqa.package().ANSWER_INSTRUCTION)


def test_flag_and_adapter_come_from_config_then_env(monkeypatch, tmp_path):
    monkeypatch.delenv(cdvqa.FLAG_ENV, raising=False)
    monkeypatch.delenv(cdvqa.ADAPTER_ENV, raising=False)
    assert cdvqa.enabled() is False and cdvqa.adapter_path() is None
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    monkeypatch.setenv(cdvqa.ADAPTER_ENV, str(tmp_path))
    assert cdvqa.enabled() is True and cdvqa.adapter_path() == tmp_path.resolve()
    monkeypatch.setenv(cdvqa.FLAG_ENV, "0")
    assert cdvqa.enabled() is False
