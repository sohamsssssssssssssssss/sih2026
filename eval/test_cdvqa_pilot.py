"""CPU-double tests for the CDVQA pair stages: 2-step train, resume, and all eval conditions.

Uses a tiny fake Qwen-shaped model and processor, real PNG files, and the real Trainer on CPU.
No weights, no GPU, no network. The 4-bit/fp16 GPU path is intentionally not exercised here.
"""
import json
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from transformers import TrainerCallback

from eval.cdvqa_pairs import (QUESTION_TYPES, condition_example, evaluate_conditions, majority_accuracy,
                              paired_differences, score, summarise)
from scripts import cdvqa_pilot as pilot
from training.remote_sensing import QwenVQACollator

CHOICES = ("yes", "no")


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, conversation, **kwargs):
        return "full" if conversation[-1]["role"] == "assistant" else "prompt"

    def __call__(self, text, images=None, videos=None, **kwargs):
        ids = [[4, 5, 6, 7, 0]] * len(text) if text[0] == "full" else [[4, 5, 0]] * len(text)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.long)}

    def save_pretrained(self, path):
        (path / "processor.json").write_text("{}")


def fake_process_vision_info(conversation):
    return [item["image"] for item in conversation[0]["content"] if item["type"] == "image"], None


class TinyQwen(nn.Module):
    """Module names mirror the real layout: visual.patch_embed.proj and layers.N.self_attn.<target>.lora_{A,B}."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_embed = nn.Module()
        self.visual.patch_embed.proj = nn.Conv3d(3, 4, kernel_size=1)
        self.visual.patch_embed.proj.requires_grad_(False)
        self.embed = nn.Embedding(16, 32)
        self.embed.requires_grad_(False)
        block = nn.ModuleDict({t: nn.ModuleDict({"lora_A": nn.ModuleDict({"default": nn.Linear(32, 16, bias=False)}),
                                                 "lora_B": nn.ModuleDict({"default": nn.Linear(16, 32, bias=False)})})
                               for t in ("q_proj", "k_proj", "v_proj", "o_proj")})
        self.layers = nn.ModuleList([nn.ModuleDict({"self_attn": block})])

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.embed(input_ids)
        attention = self.layers[0]["self_attn"]
        for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hidden = hidden + attention[target]["lora_B"]["default"](attention[target]["lora_A"]["default"](hidden))
        return {"loss": (hidden ** 2).mean()}

    def save_pretrained(self, path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(b"tiny")


def make_root(tmp_path, rows=4):
    root = tmp_path / "package"
    manifests = root / pilot.MANIFESTS
    manifests.mkdir(parents=True)
    lines = []
    for i in range(rows):
        name = f"{i}.png"
        for side in ("im1", "im2"):
            (root / "images" / side).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(root / "images" / side / name)
        lines.append({"id": f"cdvqa-train-{i}", "split": "train",
                      "question": f"Did area {i} change?", "answer": "yes",
                      "question_type": QUESTION_TYPES[i % len(QUESTION_TYPES)], "before_image": f"im1/{name}",
                      "after_image": f"im2/{name}", "pair_id": str(i), "question_id": i})
    (manifests / "train-8k.v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    types = {t: list(CHOICES) for t in QUESTION_TYPES}
    (root / pilot.PROVENANCE).write_text(json.dumps({
        "answer_vocabulary": types, "answer_choices": types, "majority_answer": {t: "yes" for t in QUESTION_TYPES},
        "manifest_sha256": {}}))
    training = {"split": "train", "seed": 17, "max_samples": 8000, "image_size": 392, "epochs": 1.0,
                "max_steps": 1000, "batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 0.0002,
                "precision": "bf16", "quantization": "none", "lora_rank": 16, "lora_alpha": 32,
                "lora_dropout": 0.05, "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    (root / "pilot-config.json").write_text(json.dumps({"training": training, "base_revision": "rev",
                                                        "dependency_pins": {}}))
    (root / "package-manifest.json").write_text(json.dumps({"payload_sha256": {"pilot-config.json": "x"}}))
    return root


def tiny_collator(processor, image_size, max_sequence_length=32768):
    return QwenVQACollator(processor, image_size, process_vision_info=fake_process_vision_info,
                           max_sequence_length=max_sequence_length)


class WriteAdapter(TrainerCallback):
    """Test callback: Trainer saves plain modules without PEFT's adapter files, so write them on save.

    Registered before the pilot's Budget callback, so the hash inventory includes these files.
    """

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"tiny")
        return control


def run_train(root, output, *, total_steps=2, save_every=1, resume_from=None):
    return pilot.train_pairs(root, root / "weights", output, root / "images", began=None,
                             resume_from=resume_from, total_steps=total_steps, save_every=save_every,
                             expected_examples=4, extra_callbacks=(WriteAdapter(),),
                             component_loader=lambda cfg: (TinyQwen(), FakeProcessor()),
                             collator_factory=tiny_collator)


def test_two_step_pair_training_on_cpu(tmp_path):
    root = make_root(tmp_path)
    output = tmp_path / "run-a"
    output.mkdir()
    report = run_train(root, output)
    assert report["status"] == "training_complete"
    assert report["global_step"] == 2
    assert (output / "checkpoints" / "checkpoint-2" / "pilot-checkpoint.json").is_file()
    assert json.loads((output / "training-report.json").read_text())["status"] == "training_complete"
    trainable = json.loads((output / "trainable-parameters.json").read_text())
    assert len(trainable) == 8 and all("lora_" in n for n in trainable)


def test_resume_from_verified_checkpoint_and_tamper_rejected(tmp_path):
    root = make_root(tmp_path)
    first = tmp_path / "run-a"
    first.mkdir()
    run_train(root, first)
    checkpoint = first / "checkpoints" / "checkpoint-1"
    identity = pilot.binding(root)
    evidence = pilot.verify_resume(checkpoint, identity, 2)
    assert evidence["step"] == 1

    tampered = tmp_path / "tampered"
    (tampered / "checkpoints").mkdir(parents=True)
    shutil.copytree(checkpoint, tampered / "checkpoint-1")
    (tampered / "checkpoint-1" / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.verify_resume(tampered / "checkpoint-1", identity, 2)
    with pytest.raises(ValueError, match="identity"):
        pilot.verify_resume(checkpoint, "0" * 64, 2)
    with pytest.raises(ValueError, match="schedule"):
        pilot.verify_resume(checkpoint, identity, 5)

    second = tmp_path / "run-b"
    second.mkdir()
    resumed = run_train(root, second, resume_from=checkpoint)
    assert resumed["status"] == "training_complete" and resumed["global_step"] == 2


def test_train_rejects_wrong_selection_size(tmp_path):
    root = make_root(tmp_path, rows=3)
    output = tmp_path / "run-c"
    output.mkdir()
    with pytest.raises(ValueError, match="wrong size"):
        run_train(root, output)


def test_pair_conditions_build_prompt_order_and_scores(tmp_path):
    root = make_root(tmp_path, rows=2)
    rows = [json.loads(line) for line in (root / pilot.TRAIN_MANIFEST).read_text().splitlines()]
    examples = pilot.pair_examples(root / pilot.TRAIN_MANIFEST, root / "images", root)
    seen = {}

    def answer(example):
        seen.setdefault(example.sample_id, []).append((example.image_path.name, example.after_image_path.parent.name))
        return "maybe"

    vocab = {t: frozenset(CHOICES) for t in QUESTION_TYPES}
    conditions = {"real": "real", "identical": "identical_pair", "swapped": "swapped_order"}
    records = evaluate_conditions(rows, examples, answer, vocab, conditions)
    assert set(records) == {"real", "identical", "swapped"}
    assert all(r["malformed"] and not r["correct"] for name in records for r in records[name])
    assert seen["cdvqa-train-0"][0] == ("0.png", "im2")
    assert condition_example(examples[0], "identical_pair").after_image_path == examples[0].image_path
    swapped = condition_example(examples[0], "swapped_order")
    assert (swapped.image_path, swapped.after_image_path) == (examples[0].after_image_path, examples[0].image_path)


def synthetic_records(pairs=20, seed=0):
    import random
    rng = random.Random(seed)
    result = {}
    for name, rate in (("base_real", 0.5), ("adapter_real", 0.8), ("adapter_identical", 0.6), ("adapter_swapped", 0.7)):
        records = []
        for p in range(pairs):
            for t in QUESTION_TYPES:
                for k in range(2):
                    records.append({"sample_id": f"s{p}-{t}-{k}", "pair_id": str(p), "question_type": t,
                                    "expected": "yes", "correct": rng.random() < rate, "malformed": False})
        result[name] = records
    return result


def test_paired_bootstrap_and_summaries_are_reproducible():
    conditions = synthetic_records()
    first = paired_differences(conditions, seed=17, repetitions=200)
    second = paired_differences(conditions, seed=17, repetitions=200)
    assert first == second
    assert set(first["comparisons"]) == {"adapter_real_minus_base_real", "adapter_real_minus_adapter_identical",
                                         "adapter_real_minus_adapter_swapped"}
    assert first["pair_clusters"] == 20
    macro = first["comparisons"]["adapter_real_minus_base_real"]["macro"]
    assert macro["ci95"][0] <= macro["difference"] <= macro["ci95"][1]
    summary = summarise(conditions["adapter_real"])
    assert set(summary["per_type"]) == set(QUESTION_TYPES) and summary["n"] == 320


def test_majority_baseline_is_constant_and_malformed_free():
    rows = [{"question_type": t, "answer": "yes" if i % 2 else "no"} for i, t in enumerate(QUESTION_TYPES * 4)]
    result = majority_accuracy(rows, {t: "no" for t in QUESTION_TYPES})
    assert result["malformed_n"] == 0
    assert result["macro_accuracy"] == pytest.approx(
        sum(v["accuracy"] for v in result["per_type"].values()) / len(QUESTION_TYPES))
    assert score("no", "no", "change_or_not", {t: frozenset(CHOICES) for t in QUESTION_TYPES})["correct"]


def test_evaluate_stage_all_conditions_on_tiny_manifest(tmp_path):
    """Evaluate path with a fake answerer: four conditions, prompt order checked, B13-shaped result."""
    root = make_root(tmp_path, rows=16)
    rows = [json.loads(line) for line in (root / pilot.TRAIN_MANIFEST).read_text().splitlines()]
    examples = pilot.pair_examples(root / pilot.TRAIN_MANIFEST, root / "images", root)
    provenance = json.loads((root / pilot.PROVENANCE).read_text())
    vocab = {t: frozenset(CHOICES) for t in QUESTION_TYPES}
    orders = {}

    def answerer_for(name):
        def answer(example):
            # Record which folder fills the first and second prompt slots: (first, second).
            orders.setdefault(name, []).append((example.image_path.parent.name, example.after_image_path.parent.name))
            return "yes" if example.sample_id.rsplit("-", 1)[1] in {"0", "2", "10", "12"} else "bogus"
        return answer

    names = {"base_real": "real", "adapter_real": "real", "adapter_identical": "identical_pair",
             "adapter_swapped": "swapped_order"}
    conditions = {name: evaluate_conditions(rows, examples, answerer_for(name), vocab, {name: kind})[name]
                  for name, kind in names.items()}
    assert orders["base_real"][0] == ("im1", "im2")
    assert orders["adapter_real"][0] == ("im1", "im2")
    assert orders["adapter_identical"][0] == ("im1", "im1")
    assert orders["adapter_swapped"][0] == ("im2", "im1")
    kwargs = dict(eval_name="test1", eval_manifest_sha256="0" * 64, rows=rows, provenance=provenance,
                  base_meta={"base_model_id": "m", "base_revision": "r", "base_verification": {}},
                  adapter_meta={"adapter_model.safetensors": "0" * 64}, run_identity="id", bootstrap=50)
    result = pilot.build_pair_result(mode="adapter", conditions=conditions, **kwargs)
    assert set(result["conditions"]) == set(names)
    assert result["conditions"]["adapter_real"]["summary"]["malformed_n"] == 12
    assert set(result["paired_differences"]["comparisons"]) == {
        "adapter_real_minus_base_real", "adapter_real_minus_adapter_identical", "adapter_real_minus_adapter_swapped"}
    assert result["majority_baseline"]["malformed_n"] == 0
    assert set(result["prompt"]["allowed_answers"]) == set(QUESTION_TYPES)
    base_only = pilot.build_pair_result(mode="base_only", conditions={"base_real": conditions["base_real"]}, **kwargs)
    assert base_only["paired_differences"] is None and base_only["mode"] == "base_only"


def test_ledger_total_defaults_and_overrides():
    b13_default = {"training": {}}
    assert pilot.ledger_total(b13_default, "train") == 7200
    assert pilot.ledger_total(b13_default, "evaluate") == 7200
    cdvqa = {"budget": {"ledger_total_seconds": {"train": 25200, "evaluate": 7200}}}
    assert pilot.ledger_total(cdvqa, "train") == 25200
    assert pilot.ledger_total(cdvqa, "evaluate") == 7200
    assert pilot.ledger_total(cdvqa, "train", override=9000) == 9000


def test_reservation_guard_refuses_over_cap():
    entries = [{"charged_seconds": 3600}] * 6
    pilot.check_reservation(entries, 3600, 25200)  # 7 x 3600 = 25200 fits exactly
    with pytest.raises(ValueError, match="exhausted"):
        pilot.check_reservation(entries + [{"charged_seconds": 3600}], 1, 25200)
    with pytest.raises(ValueError, match="exhausted"):
        pilot.check_reservation([{"charged_seconds": 3600}] * 2, 3600, 7200)
