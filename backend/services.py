"""Read-only artifact adapters and the thin live/cached inference boundary."""

import json
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from backend.scene_pack import (
    ScenePackError,
    cached_scene,
    identify_scene,
    resolution_assets,
    scene_asset,
)

# Keep model resolution offline before importing the model registry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from orchestrator.capabilities import (  # noqa: E402
    CapabilityUnavailable,
    SINGLE_IMAGE_VQA,
    UnknownCapability,
)
from orchestrator.registry import get  # noqa: E402
from orchestrator.planner import (  # noqa: E402
    Plan,
    PlanRequest,
    plan_request,
)
from orchestrator.execution_plan import (  # noqa: E402
    ExecutionPlan,
    build_execution_plan,
)
from orchestrator.executor import execute_plan  # noqa: E402
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


def is_golden_eligible_plan(execution: ExecutionPlan) -> bool:
    """Golden fallback applies only to a one-step single-image VQA plan.

    Multi-step or unavailable-capability plans must never receive the exact
    committed single-image VQA result, regardless of scene and question.
    """
    return (
        len(execution.steps) == 1
        and execution.steps[0].capability == GOLDEN_CAPABILITY
        and not execution.unavailable_capabilities
    )


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
    try:
        catalog_asset = scene_asset(scene_id)
    except ScenePackError:
        catalog_asset = None
    if catalog_asset is not None:
        return catalog_asset
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
    try:
        known_scene = identify_scene(target)
    except ScenePackError:
        known_scene = None
    source = known_scene.get("source", {}) if known_scene else {}
    return {
        "scene_id": scene_id,
        "filename": safe_filename,
        "format": detected_format,
        "width": width,
        "height": height,
        "sensor": source.get("sensor"),
        "gsd": str(source["gsd"]) if source.get("gsd") is not None else None,
        "location": source.get("location"),
        "acquisition_date": source.get("acquisition_date"),
    }


def _cached_response(
    cached: dict[str, Any], sensor: str | None, reason: str, plan: Plan
) -> dict[str, Any]:
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
                    "planner_version": plan.planner_version,
                    "planner_rule": plan.rule_id,
                    "requested_capability": plan.requested_capability,
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
    response = {
        "answer": answer.strip(),
        "execution_mode": "live",
        "results_artifact": None,
        "model": {"name": model_name, "version": model_version},
        "trace": dict(trace),
        "notice": (
            "Live Qwen2.5-VL-3B inference completed."
            if model_name == MODEL_NAME
            else f"Live {model_name} inference completed."
        ),
    }
    if "evidence" in result:
        response["evidence"] = result["evidence"]
    return response


def _curated_cached_response(
    match: dict[str, Any], scene_id: str, sensor: str | None, plan: Plan
) -> dict[str, Any]:
    entry = match["entry"]
    row = match["row"]
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        raise ArtifactError("The curated cached result has no prediction.")
    answer = prediction.get("answer")
    evidence = prediction.get("evidence")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        raise ArtifactError("The curated cached result is malformed.")
    artifact = entry["artifact"]
    try:
        trace = append_record(
            {
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "params": {
                    "capability": entry["capability"],
                    "planner_version": plan.planner_version,
                    "planner_rule": plan.rule_id,
                    "requested_capability": plan.requested_capability,
                    "execution_mode": "cached_result",
                    "result_state": "cached_real",
                    "results_artifact": artifact["path"],
                    "scene_id": scene_id,
                    "sensor": sensor,
                    "source_scene_id": entry["source"]["source_id"],
                    "evaluated_expression": entry["evaluated_expression"],
                    "source_run_id": artifact.get("run_id"),
                    "source_git_sha": artifact.get("git_sha"),
                    "source_working_tree_sha256": artifact.get("working_tree_sha256"),
                },
                "input_summary": {
                    "image_paths": [entry["source"]["source_id"]],
                    "question": entry["question"],
                    "n_images": 1,
                },
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (TraceIntegrityError, OSError) as exc:
        raise TracePersistenceError from exc
    return {
        "answer": answer.strip(),
        "evidence": evidence,
        "execution_mode": "cached_result",
        "results_artifact": artifact["path"],
        "model": {
            "name": artifact["model_name"],
            "version": artifact["model_version"],
        },
        "trace": trace,
        "notice": (
            "CACHED REAL: replaying the committed measured result for evaluated "
            f"expression '{entry['evaluated_expression']}'. No live model ran."
        ),
    }


def analyze_scene(
    scene_id: str,
    question: str,
    sensor: str | None,
    capability: str | None = None,
    scene_id_2: str | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise AnalysisUnavailable("A non-empty question is required. No answer was generated.")
    plan = plan_request(
        PlanRequest(
            question=question,
            scene_ids=_scene_ids(scene_id, scene_id_2),
            sensor=sensor,
            requested_capability=capability,
        )
    )
    execution = build_execution_plan(plan)
    if plan.missing_inputs:
        if "second_scene" in plan.missing_inputs:
            raise AnalysisUnavailable("This request requires two scenes.")
        raise AnalysisUnavailable("This request requires a scene.")
    if not plan.executable:
        raise CapabilityUnavailable(
            plan.unavailable_reason or "The selected capability cannot be executed."
        )
    cached = (
        find_cached_result(scene_id, question, plan.selected_capability)
        if is_golden_eligible_plan(execution)
        else None
    )
    image_path = local_scene_image(scene_id)
    if image_path is None:
        if cached is None:
            raise AnalysisUnavailable(
                "No local scene pixels or exact committed result match this scene and question. No answer was generated."
            )
        return _cached_response(
            cached, sensor, "local scene pixels unavailable", plan
        )

    if INGESTED_SCENE_ID.fullmatch(scene_id):
        try:
            curated = cached_scene(image_path, question, plan.selected_capability)
        except ScenePackError as exc:
            raise ArtifactError("Curated cached result could not be verified.") from exc
        if curated is not None:
            return _curated_cached_response(curated, scene_id, sensor, plan)

    try:
        result = execute_plan(
            execution,
            route_fn=route,
            image_paths=[str(image_path)],
            question=question,
            base_params={
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
            return _cached_response(
                cached, sensor, "model execution timed out", plan
            )
        raise ModelUnavailable from exc
    except Exception as exc:
        unavailable = any(
            marker in str(exc)
            for marker in (
                "CUDA GPU",
                "requires transformers",
                "requires groundingdino",
                "configuration is unavailable",
                "checkpoint could not be loaded",
            )
        )
        if cached is None:
            error = ModelUnavailable if unavailable else ModelExecutionError
            raise error from exc
        reason = "no CUDA GPU" if unavailable else "model execution failed"
        return _cached_response(cached, sensor, reason, plan)


def _scene_ids(scene_id: str, scene_id_2: str | None) -> tuple[str, ...]:
    """Scene tuple for the planner.

    Pairwise capabilities (change_vqa, optical_sar) require two scenes; the
    planner reports ``second_scene`` as missing when only one is supplied.
    A caller that omits the second scene keeps the exact single-scene
    behaviour the frozen Phase 0 contract already had.
    """
    if scene_id_2 is None or not scene_id_2.strip():
        return (scene_id,)
    return (scene_id, scene_id_2)


def plan_analysis(
    scene_id: str,
    question: str,
    sensor: str | None,
    capability: str | None,
    scene_id_2: str | None = None,
) -> dict[str, Any]:
    """Truthful planning snapshot: planner decision plus structured steps.

    Planning invokes no model, writes no trace, and touches no GPU; a plan
    may be structurally valid while non-executable because providers are
    missing, and the response says exactly that.
    """
    plan = plan_request(
        PlanRequest(
            question=question,
            scene_ids=_scene_ids(scene_id, scene_id_2),
            sensor=sensor,
            requested_capability=capability,
        )
    )
    execution = build_execution_plan(plan)
    return {
        **asdict(plan),
        "execution_plan_version": execution.execution_plan_version,
        "steps": [
            {
                "step_id": step.step_id,
                "capability": step.capability,
                "depends_on": list(step.depends_on),
                "required_inputs": list(step.required_inputs),
                "provider_available": step.provider_available,
                "provider": step.provider,
            }
            for step in execution.steps
        ],
        "unavailable_capabilities": list(execution.unavailable_capabilities),
    }


def capabilities_overview() -> dict[str, Any]:
    """Truthful availability snapshot backed by the provider registry."""
    from orchestrator.capabilities import capabilities_status

    return {"capabilities": capabilities_status()}


def resolution_report() -> dict[str, Any]:
    report = load_results()
    try:
        assets = resolution_assets()
    except ScenePackError as exc:
        raise ArtifactError("Resolution scene-pack manifest is invalid.") from exc
    return {
        "model": report["model"],
        "n_samples": report["n_samples"],
        "timestamp": report["timestamp"],
        "provenance": report.get("provenance"),
        "per_rung": report["per_rung"],
        "degenerate_rungs": report["degenerate_rungs"],
        "assets": assets,
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
