"""Read-only artifact adapters and the thin live/cached inference boundary."""

import json
import os
import re
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

# Keep model resolution offline before importing the model registry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from orchestrator.capabilities import (  # noqa: E402
    CapabilityUnavailable,
    KNOWN_CAPABILITIES,
    SINGLE_IMAGE_VQA,
    UnknownCapability,
)
from orchestrator.registry import get  # noqa: E402
from orchestrator.router import (  # noqa: E402
    InvalidModelOutput,
    ModelExecutionTimeout,
    TracePersistenceError,
    route,
)
from orchestrator.trace import TraceIntegrityError, append_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "qwen2.5vl-3b"
RESULTS_RELATIVE_PATH = "results/qwen2.5vl-3b__ladder__rescored__20260904.json"
RESULTS_PATH = ROOT / RESULTS_RELATIVE_PATH
SAR_ANNOTATION_PATH = ROOT / "data" / "sar_gate" / "annotation_template.md"
SAR_RENDER_DIR = ROOT / "data" / "sar_gate" / "rendered"
GOLDEN_SCENE_ID = "loveda_LoveDA_images_png_0_gsd0.3"
GOLDEN_QUESTION = "Is there a building in this image?"
GOLDEN_CAPABILITY = SINGLE_IMAGE_VQA
INGESTED_SCENE_DIR = ROOT / "data" / "runtime" / "scenes"
INGESTED_SCENE_ID = re.compile(r"scene_[0-9a-f]{32}")
MODEL_EXECUTION_TIMEOUT_SECONDS = 120.0


class ArtifactError(RuntimeError):
    """Raised when a committed demo artifact cannot be read safely."""


class AnalysisUnavailable(RuntimeError):
    """Raised when neither live inference nor an exact cached result is available."""


class InvalidImageUpload(ValueError):
    """Raised when uploaded bytes are not a supported safe image."""


class SceneStorageError(RuntimeError):
    """Raised when a validated scene cannot be stored."""


class ModelUnavailable(RuntimeError):
    """Raised when required model runtime capabilities are unavailable."""


class ModelExecutionError(RuntimeError):
    """Raised when an available model fails during execution."""


@lru_cache(maxsize=1)
def load_results() -> dict[str, Any]:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Required resolution artifact is unavailable: {exc}") from exc


def normalize_scene_id(scene_id: str) -> str:
    path = Path(scene_id)
    value = str(path.with_suffix("")) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else scene_id
    return value.replace("loveda_Train_Rural_images_png_", "loveda_LoveDA_images_png_")


def find_cached_result(
    scene_id: str, question: str, capability: str
) -> dict[str, Any] | None:
    if capability != GOLDEN_CAPABILITY:
        return None
    if scene_id != GOLDEN_SCENE_ID or question != GOLDEN_QUESTION:
        return None
    return next(
        (
            row
            for row in load_results().get("results", [])
            if row.get("tile_id") == GOLDEN_SCENE_ID
            and row.get("question") == GOLDEN_QUESTION
        ),
        None,
    )


def local_scene_image(scene_id: str) -> Path | None:
    if INGESTED_SCENE_ID.fullmatch(scene_id):
        candidate = INGESTED_SCENE_DIR / f"{scene_id}.png"
        return candidate if candidate.is_file() else None
    if "/" in scene_id or "\\" in scene_id or ".." in scene_id:
        return None
    normalized = normalize_scene_id(scene_id)
    if "_gsd" not in normalized:
        return None
    local_id = normalized.replace("loveda_LoveDA_images_png_", "loveda_Train_Rural_images_png_")
    gsd = normalized.rsplit("_gsd", 1)[-1]
    candidate = ROOT / "data" / "ladder" / gsd / f"{local_id}.png"
    return candidate if candidate.is_file() else None


def ingest_scene(data: bytes, filename: str) -> dict[str, Any]:
    if not data:
        raise InvalidImageUpload("The uploaded image is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                detected_format = source.format
                if detected_format not in {"PNG", "JPEG"}:
                    raise InvalidImageUpload("Only PNG and JPEG images are supported.")
                source.verify()
            with Image.open(BytesIO(data)) as source:
                source.load()
                width, height = source.size
                if source.mode in {"L", "LA", "RGB", "RGBA"}:
                    canonical = source.copy()
                else:
                    mode = "RGBA" if "transparency" in source.info else "RGB"
                    canonical = source.convert(mode)
    except InvalidImageUpload:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise InvalidImageUpload("The uploaded file is not a safe, valid image.") from exc

    scene_id = f"scene_{uuid4().hex}"
    target = INGESTED_SCENE_DIR / f"{scene_id}.png"
    temporary = INGESTED_SCENE_DIR / f".{scene_id}.tmp"
    try:
        INGESTED_SCENE_DIR.mkdir(parents=True, exist_ok=True)
        canonical.save(temporary, format="PNG")
        temporary.replace(target)
    except OSError as exc:
        raise SceneStorageError("The uploaded image could not be stored.") from exc
    finally:
        canonical.close()
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

    safe_filename = Path(filename.replace("\\", "/")).name or "upload"
    return {
        "scene_id": scene_id,
        "filename": safe_filename,
        "format": detected_format,
        "width": width,
        "height": height,
        "sensor": None,
        "gsd": None,
        "location": None,
        "acquisition_date": None,
    }


def _cached_response(cached: dict[str, Any], sensor: str | None, reason: str) -> dict[str, Any]:
    prediction = cached.get("prediction")
    answer = prediction.get("answer") if isinstance(prediction, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ArtifactError("The committed cached result is invalid.")
    model = get(MODEL_NAME)
    try:
        trace = append_record(
            {
                "model_name": MODEL_NAME,
                "model_version": model.version,
                "params": {
                    "capability": GOLDEN_CAPABILITY,
                    "execution_mode": "cached_result",
                    "results_artifact": RESULTS_RELATIVE_PATH,
                    "scene_id": cached["tile_id"],
                    "sensor": sensor,
                },
                "input_summary": {
                    "image_paths": cached.get("image_paths", []),
                    "question": cached["question"],
                    "n_images": len(cached.get("image_paths", [])),
                },
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (TraceIntegrityError, OSError) as exc:
        raise TracePersistenceError from exc
    return {
        "answer": answer.strip(),
        "execution_mode": "cached_result",
        "results_artifact": RESULTS_RELATIVE_PATH,
        "model": {"name": MODEL_NAME, "version": model.version},
        "trace": trace,
        "notice": f"Live inference unavailable ({reason}); showing the exact committed result for this scene and question.",
    }


def _live_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise InvalidModelOutput
    answer = result.get("answer")
    trace = result.get("trace")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(trace, Mapping):
        raise InvalidModelOutput
    params = trace.get("params")
    if not isinstance(params, Mapping) or params.get("execution_mode") != "live":
        raise InvalidModelOutput
    model_name = trace.get("model_name")
    model_version = trace.get("model_version")
    if not isinstance(model_name, str) or not isinstance(model_version, str):
        raise InvalidModelOutput
    return {
        "answer": answer.strip(),
        "execution_mode": "live",
        "results_artifact": None,
        "model": {"name": model_name, "version": model_version},
        "trace": dict(trace),
        "notice": "Live Qwen2.5-VL-3B inference completed.",
    }


def analyze_scene(
    scene_id: str,
    question: str,
    sensor: str | None,
    capability: str = GOLDEN_CAPABILITY,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise AnalysisUnavailable("A non-empty question is required. No answer was generated.")
    if capability != GOLDEN_CAPABILITY:
        if capability not in KNOWN_CAPABILITIES:
            raise UnknownCapability(f"Unknown capability: {capability}")
        raise CapabilityUnavailable(
            f"No provider is registered for capability: {capability}"
        )
    cached = find_cached_result(scene_id, question, capability)
    image_path = local_scene_image(scene_id)
    if image_path is None:
        if cached is None:
            raise AnalysisUnavailable(
                "No local scene pixels or exact committed result match this scene and question. No answer was generated."
            )
        return _cached_response(cached, sensor, "local scene pixels unavailable")

    try:
        result = route(
            capability=capability,
            image_paths=[str(image_path)],
            question=question,
            params={
                "scene_id": normalize_scene_id(scene_id),
                "sensor": sensor,
                "execution_mode": "live",
            },
            timeout_seconds=MODEL_EXECUTION_TIMEOUT_SECONDS,
        )
        return _live_response(result)
    except TracePersistenceError:
        raise
    except InvalidModelOutput:
        raise
    except (CapabilityUnavailable, UnknownCapability):
        raise
    except ModelExecutionTimeout as exc:
        if cached is not None:
            return _cached_response(cached, sensor, "model execution timed out")
        raise ModelUnavailable from exc
    except Exception as exc:
        unavailable = "CUDA GPU" in str(exc) or "requires transformers" in str(exc)
        if cached is None:
            error = ModelUnavailable if unavailable else ModelExecutionError
            raise error from exc
        reason = "no CUDA GPU" if unavailable else "model execution failed"
        return _cached_response(cached, sensor, reason)


def capabilities_overview() -> dict[str, Any]:
    """Truthful availability snapshot backed by the provider registry."""
    from orchestrator.capabilities import capabilities_status

    return {"capabilities": capabilities_status()}


def resolution_report() -> dict[str, Any]:
    report = load_results()
    return {
        "model": report["model"],
        "n_samples": report["n_samples"],
        "timestamp": report["timestamp"],
        "provenance": report.get("provenance"),
        "per_rung": report["per_rung"],
        "degenerate_rungs": report["degenerate_rungs"],
    }


def _slug(value: str) -> str:
    return "-".join(value.lower().replace("–", "-").split())


def sar_annotation(scene: str) -> dict[str, Any]:
    try:
        document = SAR_ANNOTATION_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"SAR analyst annotation is unavailable: {exc}") from exc
    sections: dict[str, tuple[str, str]] = {}
    for block in document.split("\n## ")[1:]:
        title, _, body = block.partition("\n")
        sections[_slug(title)] = (title.strip(), body.strip())
    key = {"mumbai": "mumbai-coastal"}.get(_slug(scene), _slug(scene))
    if key not in sections:
        raise ArtifactError(f"No analyst annotation exists for SAR scene '{scene}'.")
    title, body = sections[key]
    body_lines = [line for line in body.splitlines() if not line.startswith("![")]
    cleaned = "\n".join(body_lines).strip()
    headings = [
        ("water", "Water areas:"),
        ("built_up", "Urban/built-up:"),
        ("vegetation", "Vegetation:"),
        ("terrain", "Terrain artifacts (layover/foreshortening/shadow):"),
    ]
    summaries: dict[str, str] = {}
    for index, (key_name, heading) in enumerate(headings):
        if heading not in cleaned:
            summaries[key_name] = "No analyst summary recorded."
            continue
        tail = cleaned.split(heading, 1)[1]
        next_markers = [next_heading for _, next_heading in headings[index + 1 :]] + ["Why it looks this way:"]
        segment = tail
        for marker in next_markers:
            if marker in segment:
                segment = segment.split(marker, 1)[0]
                break
        text = " ".join(line.strip().lstrip("- ") for line in segment.splitlines() if line.strip())
        summaries[key_name] = text[:420].rstrip() + ("…" if len(text) > 420 else "")
    render_path = SAR_RENDER_DIR / f"{key.replace('-', '_')}.png"
    return {
        "scene": key,
        "title": title,
        "human_validation": True,
        "render_available": render_path.is_file(),
        "summaries": summaries,
        "annotation": cleaned,
    }


def sar_render_path(scene: str) -> Path | None:
    key = {"mumbai": "mumbai-coastal"}.get(_slug(scene), _slug(scene))
    candidate = SAR_RENDER_DIR / f"{key.replace('-', '_')}.png"
    return candidate if candidate.is_file() else None
