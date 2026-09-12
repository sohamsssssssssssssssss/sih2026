"""Read-only artifact adapters and the thin live/cached inference boundary."""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# Keep model resolution offline before importing the model registry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from orchestrator.registry import get  # noqa: E402
from orchestrator.router import route  # noqa: E402
from orchestrator.trace import append_record  # noqa: E402
from demo_gui.golden_assets import local_golden_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "qwen2.5vl-3b"
RESULTS_RELATIVE_PATH = "results/qwen2.5vl-3b__ladder__rescored__20260904.json"
RESULTS_PATH = ROOT / RESULTS_RELATIVE_PATH
SAR_ANNOTATION_PATH = ROOT / "data" / "sar_gate" / "annotation_template.md"
SAR_RENDER_DIR = ROOT / "data" / "sar_gate" / "rendered"
GOLDEN_SCENE_ID = "loveda_LoveDA_images_png_0_gsd0.3"
GOLDEN_QUESTION = "Is there a building in this image?"


class ArtifactError(RuntimeError):
    """Raised when a committed demo artifact cannot be read safely."""


class AnalysisUnavailable(RuntimeError):
    """Raised when neither live inference nor an exact cached result is available."""


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


def find_cached_result(scene_id: str, question: str) -> dict[str, Any] | None:
    normalized = normalize_scene_id(scene_id)
    return next(
        (
            row
            for row in load_results().get("results", [])
            if row.get("tile_id") == normalized and row.get("question") == question
        ),
        None,
    )


def local_scene_image(scene_id: str) -> Path | None:
    normalized = normalize_scene_id(scene_id)
    if normalized != GOLDEN_SCENE_ID:
        return None
    return local_golden_image(normalized, ROOT)


def _cached_response(cached: dict[str, Any], sensor: str, reason: str) -> dict[str, Any]:
    model = get(MODEL_NAME)
    trace = append_record(
        {
            "model_name": MODEL_NAME,
            "model_version": model.version,
            "params": {
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
    return {
        "answer": cached["prediction"]["answer"],
        "execution_mode": "cached_result",
        "results_artifact": RESULTS_RELATIVE_PATH,
        "model": {"name": MODEL_NAME, "version": model.version},
        "trace": trace,
        "notice": f"Live inference unavailable ({reason}); showing the exact committed result for this scene and question.",
    }


def analyze_scene(scene_id: str, question: str, sensor: str) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise AnalysisUnavailable("A non-empty question is required. No answer was generated.")
    cached = find_cached_result(scene_id, question)
    image_path = local_scene_image(scene_id)
    if image_path is None:
        if cached is None:
            raise AnalysisUnavailable(
                "No local scene pixels or exact committed result match this scene and question. No answer was generated."
            )
        return _cached_response(cached, sensor, "local scene pixels unavailable")

    try:
        result = route(
            model_name=MODEL_NAME,
            image_paths=[str(image_path)],
            question=question,
            params={
                "scene_id": normalize_scene_id(scene_id),
                "sensor": sensor,
                "execution_mode": "live",
            },
        )
    except (RuntimeError, OSError) as exc:
        if cached is None:
            raise AnalysisUnavailable(
                f"Live inference is unavailable ({exc}) and no exact committed result matches. No answer was generated."
            ) from exc
        reason = "no CUDA GPU" if "CUDA GPU" in str(exc) else str(exc)
        return _cached_response(cached, sensor, reason)

    params = result.get("trace", {}).get("params", {})
    if params.get("execution_mode") != "live":
        raise AnalysisUnavailable("Live response is missing valid execution-mode provenance. No answer was returned.")
    return {
        "answer": result["answer"],
        "execution_mode": "live",
        "results_artifact": None,
        "model": {
            "name": result["trace"]["model_name"],
            "version": result["trace"]["model_version"],
        },
        "trace": result["trace"],
        "notice": "Live Qwen2.5-VL-3B inference completed.",
    }


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
