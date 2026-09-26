"""Offline manifest, collation, and QLoRA setup for remote-sensing VQA."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, UnidentifiedImageError

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SPLITS = frozenset({"train", "validation", "test"})
ANSWER_INSTRUCTION = "Answer with a single word or number only. No explanation."


@dataclass(frozen=True)
class TrainingExample:
    image_path: Path
    instruction: str
    response: str
    dataset: str
    source: str
    split: str
    sample_id: str


@dataclass(frozen=True)
class TrainingConfig:
    model_path: Path
    dataset_manifest: Path
    image_root: Path | None
    output_dir: Path
    split: str = "train"
    seed: int = 17
    max_samples: int = 512
    image_size: int = 392
    epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    precision: str = "fp16"
    quantization: str = "4bit"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    model_revision: str | None = None
    resume_from_checkpoint: Path | None = None

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if self.max_samples < 1 or self.image_size < 28 or self.image_size % 28:
            raise ValueError("max_samples must be positive and image_size a positive multiple of 28")
        if self.epochs <= 0 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch size, and accumulation must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.learning_rate <= 0 or self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("learning rate and LoRA rank/alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.precision not in {"fp16", "bf16"}:
            raise ValueError("precision must be fp16 or bf16")
        if self.quantization not in {"4bit", "none"}:
            raise ValueError("quantization must be 4bit or none")
        if not self.lora_target_modules:
            raise ValueError("at least one LoRA target module is required")

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("model_path", "dataset_manifest", "image_root", "output_dir", "resume_from_checkpoint"):
            value[key] = str(value[key]) if value[key] is not None else None
        value["lora_target_modules"] = list(self.lora_target_modules)
        return value


def _text(record: dict[str, Any], field: str, line: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: {field} must be non-empty text")
    return value.strip()


def load_manifest(manifest: Path, image_root: Path | None = None) -> list[TrainingExample]:
    """Load and fully validate an offline JSONL training/evaluation manifest."""
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest}")
    root = (image_root or manifest.parent).resolve()
    examples: list[TrainingExample] = []
    identities: set[tuple[str, str]] = set()
    image_splits: dict[Path, set[str]] = {}
    validated_images: set[Path] = set()
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: record must be an object")
        image_value = _text(record, "image", line_number)
        image_path = Path(image_value)
        image_path = (image_path if image_path.is_absolute() else root / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"line {line_number}: image does not exist: {image_path}")
        if image_path not in validated_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"line {line_number}: corrupt or unsupported image: {image_path}") from exc
            validated_images.add(image_path)
        split = _text(record, "split", line_number)
        if split not in SPLITS:
            raise ValueError(f"line {line_number}: split must be one of {sorted(SPLITS)}")
        dataset = _text(record, "dataset", line_number)
        sample_id = _text(record, "sample_id", line_number)
        identity = (dataset, sample_id)
        if identity in identities:
            raise ValueError(f"line {line_number}: duplicate dataset/sample identity: {identity}")
        identities.add(identity)
        image_splits.setdefault(image_path, set()).add(split)
        examples.append(
            TrainingExample(
                image_path=image_path,
                instruction=_text(record, "instruction", line_number),
                response=_text(record, "response", line_number),
                dataset=dataset,
                source=_text(record, "source", line_number),
                split=split,
                sample_id=sample_id,
            )
        )
    if not examples:
        raise ValueError("Dataset manifest contains no examples")
    leaked = [key for key, splits in image_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Images cross train/evaluation splits: {leaked[:3]}")
    return examples


def select_examples(
    examples: Iterable[TrainingExample], split: str, max_samples: int, seed: int
) -> list[TrainingExample]:
    if split not in SPLITS or max_samples < 1:
        raise ValueError("invalid split or max_samples")
    selected = [example for example in examples if example.split == split]
    selected.sort(
        key=lambda example: hashlib.sha256(
            f"{seed}\0{example.dataset}\0{example.sample_id}".encode()
        ).digest()
    )
    if not selected:
        raise ValueError(f"No examples exist for split: {split}")
    return selected[:max_samples]


def messages(example: TrainingExample, image_size: int, include_target: bool) -> list[dict]:
    content = [
        {
            "type": "image",
            "image": example.image_path.as_uri(),
            "resized_height": image_size,
            "resized_width": image_size,
        },
        {"type": "text", "text": f"{example.instruction} {ANSWER_INSTRUCTION}"},
    ]
    result = [{"role": "user", "content": content}]
    if include_target:
        result.append(
            {"role": "assistant", "content": [{"type": "text", "text": example.response}]}
        )
    return result


class QwenVQACollator:
    """Build Qwen multimodal batches and mask all non-target prompt tokens."""

    def __init__(
        self,
        processor: Any,
        image_size: int,
        process_vision_info: Callable | None = None,
    ) -> None:
        if process_vision_info is None:
            from qwen_vl_utils import process_vision_info
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.image_size = image_size
        self.process_vision_info = process_vision_info

    def _encode(self, batch: list[TrainingExample], include_target: bool) -> Any:
        conversations = [messages(item, self.image_size, include_target) for item in batch]
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=not include_target,
            )
            for conversation in conversations
        ]
        images, videos = zip(
            *(self.process_vision_info(conversation) for conversation in conversations)
        )
        image_inputs = [item for group in images if group for item in group]
        video_inputs = [item for group in videos if group for item in group]
        return self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, batch: list[TrainingExample]) -> dict[str, Any]:
        encoded = self._encode(batch, include_target=True)
        prompt = self._encode(batch, include_target=False)
        labels = encoded["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row, prompt_ids in enumerate(prompt["input_ids"]):
            prompt_length = int((prompt_ids != self.processor.tokenizer.pad_token_id).sum())
            labels[row, :prompt_length] = -100
        encoded["labels"] = labels
        return encoded


def load_training_components(config: TrainingConfig) -> tuple[Any, Any]:
    """Load an offline Qwen checkpoint and attach trainable LoRA parameters."""
    config.validate()
    if not config.model_path.is_dir():
        raise FileNotFoundError(f"Local base model directory does not exist: {config.model_path}")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    quantization_config = None
    if config.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(config.model_path),
        local_files_only=True,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.lora_target_modules),
        ),
    )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("LoRA setup produced no trainable parameters")
    processor = AutoProcessor.from_pretrained(str(config.model_path), local_files_only=True)
    return model, processor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
