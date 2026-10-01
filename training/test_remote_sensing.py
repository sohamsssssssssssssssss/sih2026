import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.prepare_rsvqa_training_manifest import build_manifest
from scripts.train_remote_sensing_adapter import parse_args, run
from training.remote_sensing import (
    ANSWER_INSTRUCTION,
    QwenVQACollator,
    TrainingConfig,
    load_manifest,
    load_training_components,
    messages,
    select_examples,
    sha256_file,
)


def image(path: Path, color=(10, 20, 30)) -> Path:
    Image.new("RGB", (4, 4), color).save(path)
    return path


def record(path: Path, sample_id: str, split: str = "train") -> dict:
    return {
        "image": path.name,
        "instruction": "What is visible?",
        "response": "forest",
        "dataset": "fixture-rs-vqa",
        "source": "generated test fixture",
        "split": split,
        "sample_id": sample_id,
    }


def manifest(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def test_valid_manifest_builds_complete_examples(tmp_path: Path) -> None:
    raster = image(tmp_path / "scene.png")
    loaded = load_manifest(manifest(tmp_path / "samples.jsonl", [record(raster, "one")]))

    assert loaded[0].image_path == raster.resolve()
    assert loaded[0].instruction == "What is visible?"
    assert loaded[0].response == "forest"
    assert loaded[0].dataset == "fixture-rs-vqa"
    assert loaded[0].source == "generated test fixture"


def test_rsvqa_builder_preserves_official_splits_without_downloading(tmp_path: Path) -> None:
    dataset = tmp_path / "rsvqa"
    dataset.mkdir()
    for index, file_split in enumerate(("train", "val", "test"), 1):
        image(dataset / f"{index}.tif")
        (dataset / f"LR_split_{file_split}_images.json").write_text(
            json.dumps({"images": [{"id": index, "active": True}]}), encoding="utf-8"
        )
        (dataset / f"LR_split_{file_split}_questions.json").write_text(
            json.dumps(
                {
                    "questions": [
                        {"id": index * 10, "img_id": index, "question": "What?", "active": True},
                        *([{"id": 999, "active": False}] if file_split == "train" else []),
                    ]
                }
            ),
            encoding="utf-8",
        )
        (dataset / f"LR_split_{file_split}_answers.json").write_text(
            json.dumps(
                {"answers": [{"question_id": index * 10, "answer": "forest", "active": True}]}
            ),
            encoding="utf-8",
        )
    output = tmp_path / "rsvqa.jsonl"

    assert build_manifest(dataset, output) == 3
    loaded = load_manifest(output, dataset)
    assert {item.split for item in loaded} == {"train", "validation", "test"}
    assert {item.source for item in loaded} == {"RSVQA-LR, Zenodo record 6344334"}
    assert {item.sample_id for item in loaded} == {"train-10", "validation-20", "test-30"}

    (dataset / "LR_split_train_questions.json").write_text(
        json.dumps({"questions": [{"id": 10, "question": "What?", "active": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="img_id"):
        build_manifest(dataset, output)


@pytest.mark.parametrize("field", ["instruction", "response", "dataset", "source", "split", "sample_id"])
def test_manifest_rejects_missing_or_empty_fields(tmp_path: Path, field: str) -> None:
    item = record(image(tmp_path / "scene.png"), "one")
    item[field] = ""
    with pytest.raises(ValueError):
        load_manifest(manifest(tmp_path / "samples.jsonl", [item]))


def test_manifest_rejects_malformed_json_missing_and_corrupt_images(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_manifest(malformed)

    with pytest.raises(FileNotFoundError, match="image does not exist"):
        load_manifest(manifest(tmp_path / "missing.jsonl", [record(tmp_path / "none.png", "one")]))

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt or unsupported"):
        load_manifest(manifest(tmp_path / "corrupt.jsonl", [record(corrupt, "one")]))


def test_manifest_rejects_image_leakage_across_splits(tmp_path: Path) -> None:
    raster = image(tmp_path / "scene.png")
    records = [record(raster, "train-one"), record(raster, "validation-one", "validation")]
    with pytest.raises(ValueError, match="cross train/evaluation splits"):
        load_manifest(manifest(tmp_path / "samples.jsonl", records))


def test_manifest_rejects_image_leakage_even_if_dataset_label_changes(tmp_path: Path) -> None:
    raster = image(tmp_path / "scene.png")
    train = record(raster, "train-one")
    validation = record(raster, "validation-one", "validation")
    validation["dataset"] = "different-label"
    with pytest.raises(ValueError, match="cross train/evaluation splits"):
        load_manifest(manifest(tmp_path / "samples.jsonl", [train, validation]))


def test_selection_is_split_safe_bounded_and_deterministic(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        records.append(record(image(tmp_path / f"train-{index}.png"), str(index)))
    records.append(record(image(tmp_path / "validation.png"), "v", "validation"))
    loaded = load_manifest(manifest(tmp_path / "samples.jsonl", records))

    first = select_examples(loaded, "train", 4, seed=9)
    second = select_examples(reversed(loaded), "train", 4, seed=9)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len(first) == 4
    assert {item.split for item in first} == {"train"}


def test_prompt_contains_image_question_and_target(tmp_path: Path) -> None:
    example = load_manifest(
        manifest(tmp_path / "samples.jsonl", [record(image(tmp_path / "scene.png"), "one")])
    )[0]
    conversation = messages(example, 224, include_target=True)

    assert conversation[0]["content"][0]["resized_height"] == 224
    assert conversation[0]["content"][1]["text"] == f"What is visible? {ANSWER_INSTRUCTION}"
    assert conversation[1]["content"][0]["text"] == "forest"


class FakeBatch(dict):
    pass


class FakeProcessor:
    class Tokenizer:
        pad_token_id = 0

    tokenizer = Tokenizer()

    def apply_chat_template(self, conversation, **_):
        return "full target" if conversation[-1]["role"] == "assistant" else "prompt"

    def __call__(self, *, text, **_):
        width = 4 if text[0] == "full target" else 2
        return FakeBatch(input_ids=torch.tensor([[1] * width + [0]]))


def test_collator_masks_prompt_and_padding_but_keeps_target(tmp_path: Path) -> None:
    example = load_manifest(
        manifest(tmp_path / "samples.jsonl", [record(image(tmp_path / "scene.png"), "one")])
    )[0]
    collator = QwenVQACollator(
        FakeProcessor(), 224, process_vision_info=lambda _: ([object()], None)
    )
    batch = collator([example])

    assert batch["labels"].tolist() == [[-100, -100, 1, 1, -100]]


def test_configuration_serialization(tmp_path: Path) -> None:
    config = TrainingConfig(
        model_path=tmp_path / "model",
        dataset_manifest=tmp_path / "samples.jsonl",
        image_root=None,
        output_dir=tmp_path / "out",
        lora_rank=8,
    )
    serialized = config.serializable()
    assert serialized["lora_rank"] == 8
    assert serialized["model_path"].endswith("model")
    assert serialized["lora_target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_model_and_processor_loading_is_local_only(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls = []

    class LoadedModel(FakeModel):
        config = type("Config", (), {"use_cache": True})()

        def gradient_checkpointing_enable(self):
            return None

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return LoadedModel()

    class ProcessorLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", path, kwargs))
            return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = ProcessorLoader
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    transformers.Qwen2_5_VLForConditionalGeneration = ModelLoader
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: kwargs
    peft.get_peft_model = lambda model, _: model
    peft.prepare_model_for_kbit_training = lambda model: model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    config = TrainingConfig(
        model_path=model_dir,
        dataset_manifest=tmp_path / "unused.jsonl",
        image_root=None,
        output_dir=tmp_path / "out",
        quantization="none",
    )

    load_training_components(config)

    assert [item[0] for item in calls] == ["model", "processor"]
    assert all(item[2]["local_files_only"] is True for item in calls)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, **_):
        return type("Output", (), {"loss": self.weight.sum()})()


def test_dry_run_is_offline_and_performs_no_optimizer_step(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = manifest(
        tmp_path / "samples.jsonl",
        [record(image(tmp_path / "scene.png"), "one")],
    )
    config = TrainingConfig(
        model_path=model_dir,
        dataset_manifest=dataset,
        image_root=None,
        output_dir=tmp_path / "out",
        max_samples=1,
    )
    model = FakeModel()
    initial = model.weight.detach().clone()

    report = run(
        config,
        dry_run=True,
        component_loader=lambda _: (model, object()),
        collator_factory=lambda *_: (lambda _: {"input_ids": torch.tensor([[1]])}),
    )

    assert report["status"] == "dry_run_passed"
    assert report["dataset"]["training_sample_count"] == 1
    assert report["base_model"]["revision"] is None
    assert model.weight.detach().equal(initial)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert (tmp_path / "out" / "training-report.json").is_file()


def test_training_entrypoint_rejects_evaluation_split(tmp_path: Path) -> None:
    config = TrainingConfig(
        model_path=tmp_path / "model",
        dataset_manifest=tmp_path / "samples.jsonl",
        image_root=None,
        output_dir=tmp_path / "out",
        split="validation",
    )
    with pytest.raises(ValueError, match="only permits the train split"):
        run(config, dry_run=True)


def test_training_rejects_unexpected_manifest_before_model_load(tmp_path: Path) -> None:
    dataset = manifest(
        tmp_path / "samples.jsonl",
        [record(image(tmp_path / "scene.png"), "one")],
    )
    config = TrainingConfig(
        model_path=tmp_path / "model",
        dataset_manifest=dataset,
        image_root=None,
        output_dir=tmp_path / "out",
        expected_manifest_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="Dataset manifest SHA-256 mismatch"):
        run(config, component_loader=lambda _: pytest.fail("model loaded before manifest check"))
    assert not config.output_dir.exists()

    config, _ = parse_args([
        "--model-path", str(tmp_path / "model"),
        "--dataset-manifest", str(dataset),
        "--output-dir", str(tmp_path / "out"),
        "--expected-manifest-sha256", sha256_file(dataset),
    ])
    assert config.expected_manifest_sha256 == sha256_file(dataset)
    (tmp_path / "model").mkdir()
    assert run(
        config,
        dry_run=True,
        component_loader=lambda _: (FakeModel(), object()),
        collator_factory=lambda *_: (lambda _: {"input_ids": torch.tensor([[1]])}),
    )["status"] == "dry_run_passed"


def test_training_rejects_invalid_expected_digest(tmp_path: Path) -> None:
    config = TrainingConfig(
        model_path=tmp_path / "model",
        dataset_manifest=tmp_path / "samples.jsonl",
        image_root=None,
        output_dir=tmp_path / "out",
        expected_manifest_sha256="invalid",
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        run(config, dry_run=True)
