"""Lazy, inference-only Qwen2.5-VL wrapper."""

import os
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from models.artifacts import validate_artifact
from models.base import Model, ModelReadiness

# Opt-in only. Apple MPS lets this model run on a developer Mac, which is the
# difference between iterating locally and not being able to execute the live path
# at all. It is NOT a substitute for the rented GPU: MPS uses different kernels and a
# different accumulation order, so a number produced here is not comparable with the
# committed CUDA baselines. Requiring an explicit environment variable keeps that a
# deliberate act rather than something a laptop starts doing silently.
ALLOW_MPS_ENV = "SATQUERY_ALLOW_MPS"

NO_ACCELERATOR_ERROR = (
    "Qwen2.5-VL inference requires a CUDA GPU for this baseline; use the Kaggle T4 "
    f"runner. For local iteration only, set {ALLOW_MPS_ENV}=1 to permit Apple MPS — "
    "its output is not a baseline number and must never be reported as one."
)


class QwenVLModel(Model):
    """Frozen Qwen2.5-VL-3B-Instruct model for zero-shot VQA."""

    name = "qwen2.5vl-3b"
    version = "Qwen/Qwen2.5-VL-3B-Instruct"
    isolated = True

    def __init__(self, model_id: str = version, max_new_tokens: int = 50) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._process_vision_info: Any | None = None
        # Set by _load(). "cuda" for reportable runs, "mps" for local iteration only.
        # Read this before quoting any number this model produced.
        self.device: str | None = None

    @staticmethod
    def _resolve_device(torch: Any) -> str:
        """Pick the accelerator, or fail closed.

        CUDA is the only device whose numbers are comparable with the committed
        baselines. MPS is permitted solely as an explicit, opt-in escape hatch for
        local iteration. Absent both, this raises rather than silently falling back
        to CPU: a 3B model on CPU would appear to work while taking minutes per
        question, which is a worse failure than an honest refusal.
        """
        if torch.cuda.is_available():
            return "cuda"
        if os.environ.get(ALLOW_MPS_ENV) == "1" and torch.backends.mps.is_available():
            warnings.warn(
                f"Running Qwen2.5-VL on Apple MPS because {ALLOW_MPS_ENV}=1. This is "
                "for local iteration only — the result is NOT comparable with the "
                "committed CUDA baselines and must not be reported as a measurement.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "mps"
        raise RuntimeError(NO_ACCELERATOR_ERROR)

    def readiness(self) -> ModelReadiness:
        for dependency in (
            "torch",
            "transformers",
            "qwen_vl_utils",
            "accelerate",
            "huggingface_hub",
        ):
            if find_spec(dependency) is None:
                return ModelReadiness(
                    False,
                    "DEPENDENCY_UNAVAILABLE",
                    f"Required dependency {dependency} is unavailable.",
                )
        import torch

        try:
            self._resolve_device(torch)
        except RuntimeError as exc:
            return ModelReadiness(False, "CUDA_UNAVAILABLE", str(exc))
        artifact = validate_artifact(self.name, model_id=self.model_id)
        if not artifact.available:
            return ModelReadiness(False, artifact.reason_code, artifact.detail)
        return ModelReadiness(True)

    def _load(self) -> None:
        """Load weights only on the first real inference call."""
        if self._model is not None:
            return
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen inference requires transformers>=4.49, qwen-vl-utils, accelerate, and torch"
            ) from exc
        device = self._resolve_device(torch)
        artifact = validate_artifact(self.name, model_id=self.model_id)
        if not artifact.available or artifact.path is None:
            raise RuntimeError(artifact.detail or "Qwen artifact unavailable")
        model_path = str(artifact.path)
        if device == "cuda":
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True,
            )
        else:
            # device_map="auto" dispatches through accelerate, which does not place
            # reliably on Apple Silicon; move the whole model explicitly instead.
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to(device)
        self.device = device
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self._process_vision_info = process_vision_info

    def _generate_answer(self, image_paths: list[str], question: str) -> str:
        self._load()
        import torch

        assert (
            self._model is not None
            and self._processor is not None
            and self._process_vision_info is not None
        )
        content = [
            {"type": "image", "image": Path(image_path).resolve().as_uri()}
            for image_path in image_paths
        ]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def infer(self, image_paths: list[str], question: str) -> dict:
        if not image_paths:
            raise ValueError("Qwen VQA requires at least one image path")
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Image paths do not exist: {missing}")
        question = question + " Answer with a single word or number only. No explanation."
        answer = self._generate_answer(image_paths, question).strip()
        return {
            "answer": answer,
            "confidence": None,
            "evidence": [],
        }
