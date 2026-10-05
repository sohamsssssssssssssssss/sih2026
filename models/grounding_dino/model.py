"""Lazy Grounding DINO inference wrapper."""

import math
import os
from pathlib import Path
from importlib.util import find_spec
from typing import Any, Callable

from models.base import Model, ModelReadiness
from models.artifacts import validate_artifact

# The packaged SwinT OGC config loads this text encoder with from_pretrained,
# so under HF_HUB_OFFLINE it must already be in the local Hugging Face cache.
TEXT_ENCODER_ID = "bert-base-uncased"
TEXT_ENCODER_FILES = ("config.json", "vocab.txt")
TEXT_ENCODER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _text_encoder_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(TEXT_ENCODER_ID, filename), str)

    return all(cached(name) for name in TEXT_ENCODER_FILES) and any(
        cached(name) for name in TEXT_ENCODER_WEIGHTS
    )


class GroundingDINOModel(Model):
    """Ground text in one image with the official Swin-T checkpoint.

    The official implementation supports CPU inference, but this provider is
    deliberately CUDA-only so a missing GPU fails clearly instead of starting
    an unexpectedly slow CPU job.
    """

    name = "grounding-dino-swint"
    version = "ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth"
    isolated = True

    def __init__(
        self,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        config_path: str | Path | None = None,
    ) -> None:
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model: Any | None = None
        self._load_image: Callable[..., Any] | None = None
        self._predict_fn: Callable[..., Any] | None = None
        self._configured_config_path = Path(config_path) if config_path else None
        self._resolved_config_path: Path | None = None

    def _resolve_config(self) -> Path:
        """Resolve once so readiness and inference use the identical local file."""
        if self._resolved_config_path is None:
            configured = self._configured_config_path
            if configured is None and (environment_path := os.environ.get("SATQUERY_GROUNDING_CONFIG")):
                configured = Path(environment_path)
            if configured is None:
                import groundingdino

                configured = (
                    Path(groundingdino.__file__).resolve().parent
                    / "config"
                    / "GroundingDINO_SwinT_OGC.py"
                )
            self._resolved_config_path = configured.expanduser().resolve()
        if not self._resolved_config_path.is_file():
            raise FileNotFoundError(
                f"Grounding DINO configuration is unavailable: {self._resolved_config_path}"
            )
        return self._resolved_config_path

    def readiness(self) -> ModelReadiness:
        for dependency in ("groundingdino", "torch", "huggingface_hub"):
            if find_spec(dependency) is None:
                return ModelReadiness(False, "DEPENDENCY_UNAVAILABLE", f"Required dependency {dependency} is unavailable.")
        import torch
        if not torch.cuda.is_available():
            return ModelReadiness(False, "CUDA_UNAVAILABLE", "A CUDA GPU is required for Grounding DINO.")
        try:
            self._resolve_config()
        except (ImportError, OSError):
            return ModelReadiness(
                False,
                "NOT_CONFIGURED",
                "Grounding DINO configuration is unavailable; set "
                "SATQUERY_GROUNDING_CONFIG or install the packaged configuration.",
            )
        artifact = validate_artifact(self.name)
        if not artifact.available:
            return ModelReadiness(False, artifact.reason_code, artifact.detail)
        if not _text_encoder_cached():
            return ModelReadiness(
                False,
                "ARTIFACT_UNAVAILABLE",
                f"Grounding DINO text encoder {TEXT_ENCODER_ID} is not in the local "
                "Hugging Face cache; provision it before enabling grounding.",
            )
        return ModelReadiness(True)

    def _load(self) -> None:
        """Load the package and checkpoint only on first real inference."""
        if self._model is not None:
            return
        try:
            import torch
            from groundingdino.util.inference import load_image, load_model, predict
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO inference requires groundingdino, torch, and huggingface-hub"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Grounding DINO inference requires a CUDA GPU; CPU fallback is disabled"
            )

        try:
            config_path = self._resolve_config()
        except (ImportError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc
        artifact = validate_artifact(self.name)
        if not artifact.available or artifact.path is None:
            raise RuntimeError(artifact.detail or "Grounding DINO artifact unavailable")
        try:
            model = load_model(
                str(config_path), str(artifact.path), device="cuda"
            )
        except Exception as exc:
            detail = " ".join(str(exc).split())[:240] or "no detail"
            raise RuntimeError(
                "Grounding DINO Swin-T checkpoint could not be loaded: "
                f"{type(exc).__name__}: {detail}"
            ) from exc
        try:
            model.to("cuda")
            parameters = iter(model.parameters())
            first = next(parameters)
            if first.device.type != "cuda" or any(
                parameter.device.type != "cuda" for parameter in parameters
            ):
                raise RuntimeError("model parameters are not on CUDA")
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO model could not be placed on CUDA"
            ) from exc
        self._model = model
        self._load_image = load_image
        self._predict_fn = predict

    def _predict(
        self, image_path: str, question: str
    ) -> tuple[list[list[float]], list[float], list[str]]:
        self._load()
        assert (
            self._model is not None
            and self._load_image is not None
            and self._predict_fn is not None
        )
        _, image = self._load_image(image_path)
        boxes, scores, labels = self._predict_fn(
            model=self._model,
            image=image,
            caption=question,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device="cuda",
        )
        return (
            boxes.detach().cpu().tolist(),
            scores.detach().cpu().tolist(),
            list(labels),
        )

    def infer(self, image_paths: list[str], question: str) -> dict:
        if len(image_paths) != 1:
            raise ValueError("Grounding DINO requires exactly one image path")
        image_path = Path(image_paths[0])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        query = question.strip()
        if not query:
            raise ValueError("Grounding DINO requires a non-empty text query")

        boxes, scores, labels = self._predict(str(image_path), query)
        evidence = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            if len(box) != 4:
                raise ValueError("Grounding DINO returned a malformed box")
            center_x, center_y, width, height = map(float, box)
            score = float(score)
            if not all(math.isfinite(value) for value in (center_x, center_y, width, height, score)):
                raise ValueError("Grounding DINO returned non-finite evidence")
            if not 0.0 <= score <= 1.0 or width < 0.0 or height < 0.0:
                raise ValueError("Grounding DINO returned out-of-range evidence")
            coordinates = [
                max(0.0, min(1.0, center_x - width / 2)),
                max(0.0, min(1.0, center_y - height / 2)),
                max(0.0, min(1.0, center_x + width / 2)),
                max(0.0, min(1.0, center_y + height / 2)),
            ]
            evidence.append(
                {
                    "type": "bounding_box",
                    "label": str(label),
                    "coordinates": coordinates,
                    "coordinate_space": "normalized_xyxy",
                    "confidence": score,
                    "source_scene_id": None,
                }
            )

        count = len(evidence)
        answer = (
            f"No match found for '{query}'."
            if count == 0
            else f"Found {count} match{'es' if count != 1 else ''} for '{query}'."
        )
        return {"answer": answer, "evidence": evidence}
