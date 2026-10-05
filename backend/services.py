"""Read-only artifact adapters and the thin live/cached inference boundary."""

import hashlib
import json
import logging
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

import numpy as np
from PIL import Image, UnidentifiedImageError
from rasterio import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile

from backend.scene_pack import (
    ScenePackError,
    manifest as scene_pack_manifest,
    resolution_assets,
    scene_asset,
)
from data.dataset import SCENE_MANIFEST_VERSION, SceneManifest, validate_scene_manifest
from data.pairing import evaluate_compatibility
from models.base import ModelReadiness
from models.paths import public_path

# Keep model resolution offline before importing the model registry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from orchestrator.capabilities import (  # noqa: E402
    CapabilityUnavailable,
    CHANGE_VQA,
    OPTICAL_SAR,
    SINGLE_IMAGE_VQA,
    ProviderNotReady,
    UnknownCapability,
    capability_readiness,
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
from orchestrator.worker import WorkerCrashed  # noqa: E402
from demo_gui.golden_assets import local_golden_image  # noqa: E402

logger = logging.getLogger(__name__)

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
INGESTED_RASTER_DIR = ROOT / "data" / "runtime" / "rasters"
SCENE_MANIFEST_DIR = ROOT / "data" / "runtime" / "manifests"
INGESTED_SCENE_ID = re.compile(r"scene_[0-9a-f]{32}")
MAX_RASTER_PIXELS = 100_000_000
MAX_RASTER_BANDS = 32
MAX_PREVIEW_DIMENSION = 2048
MAX_LISTED_UPLOADS = 200
MODEL_EXECUTION_TIMEOUT_SECONDS = 120.0


LOG_EXCERPT_CHARS = 60


def log_excerpt(text: str | None, limit: int = LOG_EXCERPT_CHARS) -> str:
    """Short single-line form of user text that is safe to put in server logs."""
    value = " ".join((text or "").split())
    return value if len(value) <= limit else value[:limit] + "..."


class ArtifactError(RuntimeError):
    """Raised when a committed demo artifact cannot be read safely."""


class AnalysisUnavailable(RuntimeError):
    """Raised when neither live inference nor an exact cached result is available."""


class SceneNotFound(LookupError):
    """Raised when requested local scene pixels do not exist."""


class InvalidImageUpload(ValueError):
    """Raised when uploaded bytes are not a supported safe image."""


class SceneStorageError(RuntimeError):
    """Raised when a validated scene cannot be stored."""


class PairCompatibilityError(ValueError):
    """Raised before dispatch when a requested scene pair is ineligible."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Scene pair is not compatible with the requested workflow.")
        self.result = result


class ModelUnavailable(RuntimeError):
    """Raised when required model runtime capabilities are unavailable."""


class ModelExecutionError(RuntimeError):
    """Raised when an available model fails during execution."""


@lru_cache(maxsize=1)
def load_results() -> dict[str, Any]:
    try:
        data = RESULTS_PATH.read_bytes()
        scene = next(
            item
            for item in scene_pack_manifest()["scenes"]
            if item.get("source", {}).get("source_id") == GOLDEN_SCENE_ID
        )
        artifact = scene["artifact"]
        if (
            artifact.get("path") != RESULTS_RELATIVE_PATH
            or artifact.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError
        return json.loads(data)
    except (KeyError, OSError, StopIteration, ValueError, json.JSONDecodeError) as exc:
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
    if normalized != GOLDEN_SCENE_ID:
        return None
    return local_golden_image(normalized, ROOT)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value: object) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _preview_channel(channel: np.ma.MaskedArray) -> np.ndarray:
    values = np.ma.asarray(channel, dtype=np.float64).filled(np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(values[valid], (2, 98))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
    scaled[~valid] = 0
    return scaled.astype(np.uint8)


def _tiff_preview_and_metadata(data: bytes) -> tuple[Image.Image, dict[str, Any]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(data) as memory, memory.open() as source:
                if source.driver != "GTiff":
                    raise InvalidImageUpload("The uploaded raster is not a supported TIFF.")
                if source.width < 1 or source.height < 1 or source.count < 1:
                    raise InvalidImageUpload("The uploaded TIFF has invalid dimensions or bands.")
                if (
                    source.width * source.height > MAX_RASTER_PIXELS
                    or source.count > MAX_RASTER_BANDS
                ):
                    raise InvalidImageUpload("The uploaded TIFF exceeds raster safety limits.")
                if any(
                    np.dtype(dtype).kind not in {"u", "i", "f"}
                    for dtype in source.dtypes
                ):
                    raise InvalidImageUpload("The uploaded TIFF has an unsupported data type.")

                color = list(source.colorinterp)
                rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                if all(item in color for item in rgb):
                    bands = [color.index(item) + 1 for item in rgb]
                elif source.count >= 3:
                    bands = [1, 2, 3]
                else:
                    bands = [1]
                scale = min(1.0, MAX_PREVIEW_DIMENSION / max(source.width, source.height))
                preview_width = max(1, round(source.width * scale))
                preview_height = max(1, round(source.height * scale))
                pixels = source.read(
                    bands,
                    out_shape=(len(bands), preview_height, preview_width),
                    masked=True,
                    resampling=Resampling.nearest,
                )
                rendered = [_preview_channel(pixels[index]) for index in range(len(bands))]
                if len(rendered) == 1:
                    rendered *= 3
                preview = Image.fromarray(np.stack(rendered, axis=-1), mode="RGB")

                gcps, gcp_crs = source.gcps
                rpcs = source.rpcs
                transform = source.transform
                crs = source.crs or gcp_crs
                if rpcs:
                    georeferencing_status = "rpc"
                elif gcps:
                    georeferencing_status = "gcps"
                elif crs is None:
                    georeferencing_status = "missing_crs"
                elif transform == Affine.identity():
                    georeferencing_status = "missing_transform"
                else:
                    georeferencing_status = "affine"
                metadata = {
                    "driver": source.driver,
                    "width": source.width,
                    "height": source.height,
                    "band_count": source.count,
                    "dtypes": list(source.dtypes),
                    "crs_wkt": crs.to_wkt() if crs else None,
                    "crs_epsg": crs.to_epsg() if crs else None,
                    "transform": [
                        float(value)
                        for value in (
                            transform.a,
                            transform.b,
                            transform.c,
                            transform.d,
                            transform.e,
                            transform.f,
                        )
                    ],
                    "bounds": [float(value) for value in source.bounds],
                    "resolution": [float(abs(value)) for value in source.res],
                    "nodata": [_json_number(value) for value in source.nodatavals],
                    "georeferencing_status": georeferencing_status,
                    "pairing_ready": georeferencing_status == "affine",
                    "gcp_count": len(gcps),
                    "has_rpc": bool(rpcs),
                    "color_interpretation": [item.name for item in color],
                    "preview_bands": bands,
                }
                return preview, metadata
    except InvalidImageUpload:
        raise
    except (RasterioIOError, OSError, ValueError, TypeError) as exc:
        raise InvalidImageUpload("The uploaded file is not a safe, valid TIFF.") from exc


def _safe_filename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name or "upload"


def _declared_metadata(values: dict[str, str | None] | None) -> dict[str, Any]:
    declared: dict[str, Any] = {}
    values = values or {}
    modality = (values.get("modality") or "").strip().lower()
    if modality:
        if modality not in {"optical", "multispectral", "sar", "unknown"}:
            raise InvalidImageUpload("Unsupported declared modality.")
        declared["modality"] = modality
    for field in ("sensor", "pair_group", "benchmark_source"):
        value = (values.get(field) or "").strip()
        if value:
            if len(value) > 256:
                raise InvalidImageUpload(f"Declared {field} is too long.")
            declared[field] = value
    timestamp = (values.get("acquisition_timestamp") or "").strip()
    if timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidImageUpload("Acquisition timestamp must be valid ISO 8601.") from exc
        if parsed.tzinfo is None:
            raise InvalidImageUpload("Acquisition timestamp must include a timezone.")
        declared["acquisition_timestamp"] = parsed.isoformat()
    polarization = (values.get("polarization") or "").strip()
    if polarization:
        items = [item.strip().upper() for item in polarization.split(",") if item.strip()]
        if not items or any(not re.fullmatch(r"[A-Z0-9_-]{1,16}", item) for item in items):
            raise InvalidImageUpload("Declared polarization is malformed.")
        declared["polarizations"] = list(dict.fromkeys(items))
    return declared


def ingest_scene(
    data: bytes,
    filename: str,
    metadata: dict[str, str | None] | None = None,
    *,
    provenance: str = "user_declared_upload",
    acquisition_id: str | None = None,
) -> dict[str, Any]:
    """Store a scene and its manifest. Server-side fetchers name their own provenance."""
    if not data:
        raise InvalidImageUpload("The uploaded image is empty.")
    declared = _declared_metadata(metadata)
    raster: dict[str, Any] | None = None
    is_tiff = data[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}
    if is_tiff:
        canonical, raster = _tiff_preview_and_metadata(data)
        detected_format = "TIFF"
        width, height = raster["width"], raster["height"]
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    detected_format = source.format
                    if detected_format not in {"PNG", "JPEG"}:
                        raise InvalidImageUpload(
                            "Only PNG, JPEG, and TIFF images are supported."
                        )
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

    safe_filename = _safe_filename(filename)
    scene_id = f"scene_{uuid4().hex}"
    target = INGESTED_SCENE_DIR / f"{scene_id}.png"
    native_target = INGESTED_RASTER_DIR / f"{scene_id}.tif" if is_tiff else None
    manifest_target = SCENE_MANIFEST_DIR / f"{scene_id}.json"
    temporary = target.with_name(f".{scene_id}.png.tmp")
    native_temporary = native_target.with_name(f".{scene_id}.tif.tmp") if native_target else None
    manifest_temporary = manifest_target.with_name(f".{scene_id}.json.tmp")
    committed: list[Path] = []
    try:
        INGESTED_SCENE_DIR.mkdir(parents=True, exist_ok=True)
        SCENE_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        if native_target:
            INGESTED_RASTER_DIR.mkdir(parents=True, exist_ok=True)
            assert native_temporary is not None
            native_temporary.write_bytes(data)
        canonical.save(temporary, format="PNG")
        field_provenance = {field: provenance for field in declared}
        manifest: SceneManifest = {
            "version": SCENE_MANIFEST_VERSION,
            "scene_id": scene_id,
            "source": {
                "filename": safe_filename,
                "format": detected_format,
                "sha256": _sha256_bytes(data),
                "native_path": (
                    f"data/runtime/rasters/{scene_id}.tif" if native_target else None
                ),
            },
            "preview": {
                "path": f"data/runtime/scenes/{scene_id}.png",
                "sha256": _sha256_file(temporary),
                "width": canonical.width,
                "height": canonical.height,
                "derivation": (
                    "2nd-98th percentile display stretch"
                    if is_tiff
                    else "canonical PNG"
                ),
            },
            "raster": raster,
            "identity": {
                "sensor": declared.get("sensor"),
                "modality": declared.get("modality", "unknown"),
                "acquisition_id": acquisition_id,
                "acquisition_time": declared.get("acquisition_timestamp"),
                "polarizations": declared.get("polarizations", []),
                "benchmark_source": declared.get("benchmark_source"),
                "provenance": field_provenance,
            },
            "grouping": {
                "geographic_group": None,
                "pair_group": declared.get("pair_group"),
                "paired_scene_ids": [],
                "original_split": None,
                "label_source": None,
            },
        }
        validate_scene_manifest(manifest)
        # validate_scene_manifest ignores unknown keys, so the upload time is
        # persisted alongside the versioned schema without changing it.
        stored: dict[str, Any] = {
            **manifest,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_temporary.write_text(
            json.dumps(stored, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if native_target and native_temporary:
            native_temporary.replace(native_target)
            committed.append(native_target)
        temporary.replace(target)
        committed.append(target)
        manifest_temporary.replace(manifest_target)
        committed.append(manifest_target)
    except (OSError, ValueError) as exc:
        for path in committed:
            path.unlink(missing_ok=True)
        raise SceneStorageError("The uploaded image could not be stored.") from exc
    finally:
        canonical.close()
        for path in (temporary, native_temporary, manifest_temporary):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    return {
        "scene_id": scene_id,
        "filename": safe_filename,
        "format": detected_format,
        "width": width,
        "height": height,
        "sensor": declared.get("sensor"),
        "gsd": None,
        "location": None,
        "acquisition_date": declared.get("acquisition_timestamp"),
    }


def _uploaded_at(value: object, manifest_path: Path) -> datetime:
    """Prefer the persisted upload time; older manifests fall back to file mtime."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc)
    return datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)


def _uploaded_scene(manifest_path: Path) -> tuple[datetime, dict[str, Any]]:
    """Summarise one runtime manifest; raises ValueError/OSError when unusable."""
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Scene manifest is not an object")
    manifest = validate_scene_manifest(value)  # type: ignore[arg-type]
    scene_id = manifest["scene_id"]
    if manifest_path.name != f"{scene_id}.json":
        raise ValueError("Scene manifest filename does not match its id")
    if not (INGESTED_SCENE_DIR / f"{scene_id}.png").is_file():
        raise ValueError("Scene preview is missing")
    source, preview, raster = manifest["source"], manifest["preview"], manifest["raster"]
    identity = manifest["identity"]
    dimensions = raster if raster is not None else preview
    sensor = identity.get("sensor")
    acquisition_time = identity.get("acquisition_time")
    filename = source.get("filename")
    uploaded_at = _uploaded_at(value.get("uploaded_at"), manifest_path)
    return uploaded_at, {
        "scene_id": scene_id,
        "filename": filename if isinstance(filename, str) and filename else "upload",
        "format": source["format"],
        "width": dimensions["width"],
        "height": dimensions["height"],
        "modality": identity["modality"],
        "sensor": sensor if isinstance(sensor, str) else None,
        "acquisition_time": acquisition_time if isinstance(acquisition_time, str) else None,
        "has_native_raster": source.get("native_path") is not None
        and (INGESTED_RASTER_DIR / f"{scene_id}.tif").is_file(),
        "georeferenced": raster is not None and raster.get("pairing_ready") is True,
        "uploaded_at": uploaded_at.isoformat(),
    }


def uploaded_scene(scene_id: str) -> dict[str, Any] | None:
    """Return one stored upload's summary, or None when unknown or unusable."""
    if not INGESTED_SCENE_ID.fullmatch(scene_id):
        return None
    path = SCENE_MANIFEST_DIR / f"{scene_id}.json"
    if not path.is_file():
        return None
    try:
        return _uploaded_scene(path)[1]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Skipping unusable scene manifest %s: %s", path.name, exc)
        return None


def uploaded_scenes(limit: int | None = None) -> list[dict[str, Any]]:
    """List stored uploads newest first, skipping (and logging) unusable manifests."""
    limit = MAX_LISTED_UPLOADS if limit is None else limit
    try:
        candidates = [
            (entry.stat().st_mtime, entry)
            for entry in SCENE_MANIFEST_DIR.iterdir()
            if entry.suffix == ".json"
            and INGESTED_SCENE_ID.fullmatch(entry.stem)
            and entry.is_file()
        ]
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("Scene manifest directory could not be read", exc_info=True)
        return []
    # Manifests are written at upload time, so mtime bounds the work to the most
    # recent files; the final order uses the persisted upload timestamp.
    candidates.sort(key=lambda item: item[0], reverse=True)
    scenes: list[tuple[datetime, dict[str, Any]]] = []
    for _, path in candidates:
        if len(scenes) >= limit:
            break
        try:
            scenes.append(_uploaded_scene(path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Skipping unusable scene manifest %s: %s", path.name, exc)
    scenes.sort(key=lambda item: (item[0], item[1]["scene_id"]), reverse=True)
    return [scene for _, scene in scenes]


def _cached_response(
    cached: dict[str, Any], sensor: str | None, plan: Plan
) -> dict[str, Any]:
    prediction = cached.get("prediction")
    answer = prediction.get("answer") if isinstance(prediction, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ArtifactError("The committed cached result is invalid.")
    model = get(MODEL_NAME)
    # The committed artifact records the absolute paths of the machine that produced it.
    image_paths = [public_path(path) for path in cached.get("image_paths", [])]
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
                    "image_paths": image_paths,
                    "question": cached["question"],
                    "n_images": len(image_paths),
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
        "notice": "Cached replay requested; showing the exact committed result for this scene and question. No live inference ran.",
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


def _manifest_for_scene(scene_id: str) -> tuple[SceneManifest | None, str | None]:
    if not INGESTED_SCENE_ID.fullmatch(scene_id):
        return None, "scene_manifest_missing"
    path = SCENE_MANIFEST_DIR / f"{scene_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return validate_scene_manifest(value), None
    except FileNotFoundError:
        return None, "scene_manifest_missing"
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None, "scene_manifest_invalid"


def scene_compatibility(scene_id: str, scene_id_2: str, workflow: str) -> dict[str, Any]:
    manifests: list[SceneManifest] = []
    failed: list[dict[str, str]] = []
    unknown: list[str] = []
    for index, scene in enumerate((scene_id, scene_id_2), 1):
        manifest, error = _manifest_for_scene(scene)
        if manifest is None:
            code = error or "scene_manifest_invalid"
            failed.append(
                {
                    "code": code,
                    "message": f"Scene {index} has no valid runtime manifest.",
                }
            )
            unknown.append(f"scene_{index}.manifest")
        else:
            manifests.append(manifest)
    if failed:
        return {
            "eligible": False,
            "requested_workflow": workflow,
            "verified_checks": [],
            "failed_checks": failed,
            "warnings": [],
            "unknown_metadata": unknown,
            "normalized_overlap_ratio": None,
            "resolution_ratio": None,
            "acquisition_interval_seconds": None,
            "operations_required": [],
            "reason_codes": list(dict.fromkeys(item["code"] for item in failed)),
        }
    return evaluate_compatibility(manifests[0], manifests[1], workflow)



def _optical_sar_paths(scene_ids: tuple[str, str]) -> list[str]:
    ordered: dict[str, str] = {}
    for scene_id in scene_ids:
        manifest, _ = _manifest_for_scene(scene_id)
        if manifest is None:
            raise AnalysisUnavailable("Both optical and SAR manifests are required.")
        modality = manifest["identity"]["modality"]
        family = "optical" if modality in {"optical", "multispectral"} else modality
        native_path = manifest["source"].get("native_path")
        path = INGESTED_RASTER_DIR / f"{scene_id}.tif" if isinstance(native_path, str) else None
        if family not in {"optical", "sar"} or path is None or not path.is_file():
            raise AnalysisUnavailable("Both optical and SAR native raster files are required.")
        ordered[family] = str(path)
    if set(ordered) != {"optical", "sar"}:
        raise AnalysisUnavailable("Exactly one optical and one SAR raster are required.")
    return [ordered["optical"], ordered["sar"]]


def _change_inputs(scene_ids: tuple[str, str]) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    timestamps: list[str] = []
    for scene_id in scene_ids:
        manifest, _ = _manifest_for_scene(scene_id)
        if manifest is None:
            raise AnalysisUnavailable("Both T1 and T2 manifests are required.")
        native_path = manifest["source"].get("native_path")
        path = INGESTED_RASTER_DIR / f"{scene_id}.tif" if isinstance(native_path, str) else None
        timestamp = manifest["identity"].get("acquisition_time")
        if path is None or not path.is_file() or not isinstance(timestamp, str):
            raise AnalysisUnavailable("Both T1 and T2 native rasters and timestamps are required.")
        paths.append(str(path))
        timestamps.append(timestamp)
    return paths, timestamps


def _record_unavailable(
    capability: str,
    provider: str | None,
    reason_code: str,
    detail: str,
    scene_ids: tuple[str, ...],
    question: str,
    sensor: str | None,
) -> None:
    """Record a failed live attempt without claiming that a model executed."""
    try:
        append_record(
            {
                "model_name": "not-executed",
                "model_version": "not-executed",
                "params": {
                    "execution_mode": "live",
                    "result_state": "unavailable",
                    "capability": capability,
                    "provider": provider,
                    "reason_code": reason_code,
                    "detail": detail,
                    "scene_ids": list(scene_ids),
                    "sensor": sensor,
                },
                "input_summary": {
                    "image_paths": [],
                    "question": question,
                    "n_images": 0,
                },
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
        )
    except (TraceIntegrityError, OSError) as exc:
        raise TracePersistenceError from exc

def analyze_scene(
    scene_id: str,
    question: str,
    sensor: str | None,
    capability: str | None = None,
    scene_id_2: str | None = None,
    execution_mode: str = "live",
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
    if plan.selected_capability in {OPTICAL_SAR, CHANGE_VQA}:
        assert scene_id_2 is not None
        compatibility = scene_compatibility(
            scene_id, scene_id_2, plan.selected_capability
        )
        if not compatibility["eligible"]:
            raise PairCompatibilityError(compatibility)
    if (
        execution_mode == "live"
        and plan.selected_capability not in {OPTICAL_SAR, CHANGE_VQA}
        and local_scene_image(scene_id) is None
    ):
        raise SceneNotFound("Local scene pixels are unavailable.")
    if not execution.executable and execution_mode == "live":
        readiness = capability_readiness(plan.selected_capability)
        if not readiness["available"]:
            reason_code = str(readiness["reason_code"])
            detail = str(readiness["detail"])
            _record_unavailable(
                plan.selected_capability,
                plan.provider,
                reason_code,
                detail,
                _scene_ids(scene_id, scene_id_2),
                question,
                sensor,
            )
            raise ProviderNotReady(
                plan.selected_capability,
                plan.provider,
                ModelReadiness(False, reason_code, detail),
            )
        raise CapabilityUnavailable(
            plan.unavailable_reason or "The selected capability cannot be executed."
        )
    if execution_mode == "cached_result":
        if plan.provider is None:
            raise CapabilityUnavailable(
                plan.unavailable_reason
                or "No provider is registered for the selected capability."
            )
        if not is_golden_eligible_plan(execution):
            raise AnalysisUnavailable(
                "No replay artifact is available for this capability. No answer was generated."
            )
        cached = find_cached_result(scene_id, question, plan.selected_capability)
        if cached is None:
            raise AnalysisUnavailable(
                "No exact measured result matches this scene and question. No answer was generated."
            )
        return _cached_response(cached, sensor, plan)

    if plan.selected_capability == OPTICAL_SAR:
        assert scene_id_2 is not None
        image_paths = _optical_sar_paths((scene_id, scene_id_2))
        pair_params = {
            "scene_id_2": normalize_scene_id(scene_id_2),
            "input_modalities": ["optical", "sar"],
        }
    elif plan.selected_capability == CHANGE_VQA:
        assert scene_id_2 is not None
        image_paths, timestamps = _change_inputs((scene_id, scene_id_2))
        pair_params = {
            "scene_id_2": normalize_scene_id(scene_id_2),
            "temporal_order": ["t1", "t2"],
            "acquisition_times": timestamps,
        }
    else:
        image_path = local_scene_image(scene_id)
        if image_path is None:
            raise SceneNotFound("Local scene pixels are unavailable.")
        image_paths = [str(image_path)]
        pair_params = {}

    try:
        result = execute_plan(
            execution,
            route_fn=route,
            image_paths=image_paths,
            question=question,
            base_params={
                "scene_id": normalize_scene_id(scene_id),
                **pair_params,
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
    except ProviderNotReady as exc:
        _record_unavailable(
            exc.capability,
            exc.provider,
            exc.reason_code,
            exc.detail,
            _scene_ids(scene_id, scene_id_2),
            question,
            sensor,
        )
        raise
    except (CapabilityUnavailable, UnknownCapability):
        raise
    except ModelExecutionTimeout as exc:
        logger.warning(
            "Model execution timed out after %ss (capability=%s, question=%r)",
            MODEL_EXECUTION_TIMEOUT_SECONDS,
            plan.selected_capability,
            log_excerpt(question),
            exc_info=True,
        )
        raise ModelUnavailable from exc
    except WorkerCrashed as exc:
        # The worker respawns on the next request, so a crash (e.g. GPU OOM)
        # is a retryable outage, not a model failure.
        logger.error(
            "Inference worker crashed (capability=%s, question=%r)",
            plan.selected_capability,
            log_excerpt(question),
            exc_info=True,
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
        if unavailable:
            logger.warning(
                "Live model inference unavailable (capability=%s, question=%r)",
                plan.selected_capability,
                log_excerpt(question),
                exc_info=True,
            )
            raise ModelUnavailable from exc
        logger.exception(
            "Model execution failed (capability=%s, question=%r)",
            plan.selected_capability,
            log_excerpt(question),
        )
        raise ModelExecutionError from exc


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
