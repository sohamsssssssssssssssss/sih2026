"""Runs the stage code against the SHIPPED package files (data/cdvqa/package/), on CPU.

A key missing from pilot-config.json or model_artifacts.json fails here, not on Kaggle.
Fake processor and model only; no weights, no GPU.
"""
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from eval.test_cdvqa_pilot import FakeProcessor, fake_process_vision_info
from scripts import cdvqa_pilot as pilot
from scripts.stage1_runtime import verify_checkpoint
from training.remote_sensing import TrainingExample

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "data/cdvqa/package"
PINNED_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
TRAINING_KEYS = {"split", "seed", "max_samples", "image_size", "epochs", "max_steps", "batch_size",
                 "gradient_accumulation_steps", "learning_rate", "precision", "quantization", "lora_rank",
                 "lora_alpha", "lora_dropout", "lora_target_modules", "max_sequence_length"}


def shipped_root(tmp_path):
    """A throwaway package root holding only the shipped config files and the real manifests."""
    root = tmp_path / "pkg"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(SHIPPED / "pilot-config.json", root / "pilot-config.json")
    shutil.copyfile(SHIPPED / "configs/model_artifacts.json", root / "configs/model_artifacts.json")
    return root


def test_shipped_plan_has_every_key_the_stages_read():
    plan = json.loads((SHIPPED / "pilot-config.json").read_text())
    assert TRAINING_KEYS <= set(plan["training"])
    assert plan["training"]["max_sequence_length"] == 32768
    assert plan["base_revision"] == PINNED_REVISION
    assert set(plan["dependency_pins"]) >= {"torch", "transformers", "peft", "qwen-vl-utils"}
    assert plan["budget"]["ledger_total_seconds"] == {"train": 25200, "evaluate": 21600}
    assert plan["training"]["max_steps"] == 1000 and plan["training"]["max_samples"] == 8000
    assert plan["training"]["gradient_accumulation_steps"] == 8 and plan["training"]["batch_size"] == 1


def test_shipped_model_artifacts_have_every_key_the_stages_read():
    spec = json.loads((SHIPPED / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    assert spec["revision"] == PINNED_REVISION
    assert spec["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert set(spec["weight_sha256"]) == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_shipped_provenance_has_every_key_the_stages_read():
    provenance = json.loads((REPO / "data/manifests/cdvqa/provenance.json").read_text())
    for key in ("manifest_sha256", "answer_vocabulary", "answer_choices", "majority_answer"):
        assert key in provenance, key


def test_token_check_runs_on_shipped_plan(tmp_path):
    """token_preflight's core, with the shipped training block: fails with KeyError if a key is missing."""
    plan = json.loads((SHIPPED / "pilot-config.json").read_text())["training"]
    examples = []
    for i in range(2):
        before, after = tmp_path / f"im1-{i}.png", tmp_path / f"im2-{i}.png"
        Image.new("RGB", (512, 512), (i, 0, 0)).save(before)
        Image.new("RGB", (512, 512), (0, i, 0)).save(after)
        examples.append(TrainingExample(before, "Did it change?", "yes", "CDVQA", str(i), "train", f"s{i}",
                                        after_image_path=after, choices=("yes", "no")))
    report = pilot.token_check(plan, examples, FakeProcessor(), process_vision_info=fake_process_vision_info)
    assert report["max_sequence_length"] == 32768
    assert report["examples_checked"] == 2


def test_train_config_loads_from_shipped_plan(tmp_path):
    root = shipped_root(tmp_path)
    config = pilot.train_config(root, tmp_path / "weights", tmp_path / "out", tmp_path / "images")
    config.validate()
    assert (config.max_samples, config.max_steps, config.max_sequence_length) == (8000, 1000, 32768)
    assert config.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_verify_base_reads_shipped_artifacts_and_rejects_missing_weights(tmp_path):
    root = shipped_root(tmp_path)
    empty_weights = tmp_path / "weights"
    empty_weights.mkdir()
    with pytest.raises(ValueError, match="Pinned base revision"):
        pilot.verify_base(root, empty_weights)


def test_revision_proof_from_restored_hf_metadata(tmp_path):
    """Kaggle workaround: no _commit_hash in config.json, so the pinned commit is read from restored metadata.
    Weights must still match, so this test checks only the revision side."""
    weights = tmp_path / "qwen-pinned"
    (weights / ".cache/huggingface/download").mkdir(parents=True)
    (weights / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    (weights / ".cache/huggingface/download/config.json.metadata").write_text(f"{PINNED_REVISION}\netag\n0\n")
    spec = json.loads((SHIPPED / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    proof = verify_checkpoint(weights, spec["revision"], spec["weight_sha256"])
    assert proof["revision_source"] == "hf_download_metadata"
    assert proof["revision_matches"] is True
    assert proof["passed"] is False  # weight files are absent here, so the full proof must still fail


class StubTokenizer:
    pad_token_id = 0
    padding_side = "right"


class BatchEncoding(dict):
    """Minimal stand-in for the processor's batch output: attribute access plus .to(device)."""

    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class StubProcessor:
    """Records every chat-template text so train and eval prompts can be compared exactly."""

    tokenizer = StubTokenizer()
    chat_template = "stub"

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        content = conversation[0]["content"]
        text = " | ".join(item.get("image", "") if item["type"] == "image" else item["text"] for item in content)
        text = f"{'prompt' if add_generation_prompt else 'full'}::{text}"
        self.prompts.append(text)
        return text

    def __call__(self, text=None, images=None, videos=None, return_tensors=None, **kwargs):
        import torch
        return BatchEncoding(input_ids=torch.tensor([[1, 2, 3]] * len(text)))

    def batch_decode(self, tokens, **kwargs):
        return [" Yes \n"] * len(tokens)  # raw decode with whitespace: normalisation must strip it


class StubModel:
    device = "cpu"
    generation_config = None

    def __init__(self):
        self.generate_calls = 0

    def eval(self):
        return self

    def parameters(self):
        return []

    def generate(self, input_ids=None, do_sample=None, max_new_tokens=None, **kwargs):
        import torch
        self.generate_calls += 1
        assert do_sample is False and max_new_tokens == 32  # B13 decoding settings
        return torch.tensor([[1, 2, 3, 7, 8]])  # prompt tokens followed by the new tokens


def stub_vision_info(conversation):
    return [item["image"] for item in conversation[0]["content"] if item["type"] == "image"], None


def pair_fixture(tmp_path):
    """A package root with the shipped configs, real provenance, and a tiny two-image eval manifest."""
    root = shipped_root(tmp_path)
    (root / "data/manifests/cdvqa").mkdir(parents=True)
    for name in ("provenance.json", "train-8k.v2.jsonl", "eval-test1-400.v1.jsonl"):
        shutil.copyfile(REPO / "data/manifests/cdvqa" / name, root / "data/manifests/cdvqa" / name)
    (root / "package-manifest.json").write_text(json.dumps({"payload_sha256": {}}))
    images = tmp_path / "images"
    rows = []
    for i in range(2):
        for side in ("im1", "im2"):
            (images / side).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), (i * 40, 0, 0)).save(images / side / f"{i}.png")
        rows.append({"id": f"cdvqa-test1-{i}", "split": "test1", "question_type": "change_or_not",
                     "question": f"Did area {i} change?", "answer": "yes", "before_image": f"im1/{i}.png",
                     "after_image": f"im2/{i}.png", "pair_id": str(i), "question_id": i})
    (root / pilot.EVAL_MANIFESTS["test1"]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, images, rows


def test_evaluate_gpu_base_only_runs_end_to_end_on_stubs(tmp_path, monkeypatch):
    """Regression: the real evaluate_gpu must run every question. Before the fix this raised
    NameError: name 'answer_pair' is not defined. Only model, processor, vision info and the base
    verification are stubbed."""
    import qwen_vl_utils
    import transformers
    root, images, rows = pair_fixture(tmp_path)
    processor, model = StubProcessor(), StubModel()
    # Patch the loaders on the real classes: evaluate_gpu imports them inside the function.
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: processor))
    monkeypatch.setattr(transformers.Qwen2_5_VLForConditionalGeneration, "from_pretrained",
                        staticmethod(lambda *a, **k: model))
    monkeypatch.setattr(qwen_vl_utils, "process_vision_info", stub_vision_info)
    monkeypatch.setattr(pilot, "verify_base", lambda root, weights: {
        "revision_source": "hf_download_metadata", "measured_revision": PINNED_REVISION, "passed": True})
    output = tmp_path / "out"
    result = pilot.evaluate_gpu(root, tmp_path / "weights", None, images, "test1", output)
    assert model.generate_calls == len(rows)
    records = result["conditions"]["base_real"]["records"]
    assert [r["prediction"] for r in records] == ["Yes"] * len(rows)  # decoded, stripped, not normalised away
    assert all(r["correct"] for r in records) and all(not r["malformed"] for r in records)
    assert result["mode"] == "base_only" and result["paired_differences"] is None
    assert (output / "result.json").is_file()


def test_eval_prompt_equals_training_collator_prompt(tmp_path):
    """Train/eval parity: for the same pair example, the eval answerer and the training collator must
    produce the same chat-template prompt text (same messages, same order, same wording)."""
    from training.remote_sensing import QwenVQACollator
    root, images, rows = pair_fixture(tmp_path)
    examples = pilot.pair_examples(root / pilot.EVAL_MANIFESTS["test1"], images, root)
    eval_processor = StubProcessor()
    pilot.answer_pair(StubModel(), eval_processor, stub_vision_info, examples[0], 392)
    train_processor = StubProcessor()
    collator = QwenVQACollator(train_processor, 392, process_vision_info=stub_vision_info, max_sequence_length=32768)
    collator._encode([examples[0]], include_target=False)
    assert eval_processor.prompts == train_processor.prompts
    text = eval_processor.prompts[0]
    assert "first image is the earlier image" in text and "Allowed answers: no, yes." in text
    assert text.index(str(images / "im1" / "0.png")) < text.index(str(images / "im2" / "0.png")) < text.index("Did area 0")


@pytest.mark.parametrize("seconds,exit_code", [(7200, 0), (7201, 2)])
def test_evaluate_wall_clock_cap(seconds, exit_code, tmp_path, monkeypatch):
    """Exercise the actual CLI reservation/supervisor boundary, without a GPU worker."""
    root = shipped_root(tmp_path)
    ledger = tmp_path / "eval-ledger.json"
    output = tmp_path / "eval-output"
    calls = []
    monkeypatch.setattr(pilot, "ROOT", root)
    monkeypatch.setattr(pilot, "preflight", lambda *args: calls.append("preflight"))
    monkeypatch.setattr(pilot, "binding", lambda *args: "synthetic-eval-cap")
    def supervise(directory, argv, max_seconds, **kwargs):
        assert max_seconds == 7200
        assert json.loads(ledger.read_text())[0]["charged_seconds"] == 7200
        calls.append("supervise")
        return {"exit_code": 0, "status": "synthetic CPU check"}
    monkeypatch.setattr(pilot, "supervise", supervise)
    with pytest.raises(SystemExit) as stopped:
        pilot.main(["evaluate", "--weights", str(tmp_path / "unloaded-weights"),
                    "--image-root", str(tmp_path / "unloaded-images"), "--output", str(output),
                    "--budget-ledger", str(ledger), "--max-seconds", str(seconds)])
    assert stopped.value.code == exit_code
    assert pilot.LIMITS["train"] == 3600
    assert pilot.ledger_total(pilot.load_plan(root), "train") == 25200
    if exit_code == 0:
        assert calls == ["preflight", "supervise"]
        assert json.loads((output / "session.json").read_text())["budget_seconds"] == 7200
    else:
        assert calls == [] and not output.exists() and not ledger.exists()
