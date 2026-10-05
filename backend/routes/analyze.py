import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CapabilitiesResponse,
    PlanResponse,
    SceneUploadResponse,
)
from backend.services import (
    AnalysisUnavailable,
    ArtifactError,
    InvalidImageUpload,
    ModelExecutionError,
    ModelUnavailable,
    PairCompatibilityError,
    SceneStorageError,
    SceneNotFound,
    analyze_scene,
    capabilities_overview,
    ingest_scene,
    local_scene_image,
    log_excerpt,
    plan_analysis,
)
from backend.scene_pack import ScenePackError, scene_catalog
from orchestrator.capabilities import CapabilityUnavailable, ProviderNotReady, UnknownCapability
from orchestrator.planner import InvalidPlanRequest
from orchestrator.router import InvalidModelOutput, TracePersistenceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _request_context(request: AnalyzeRequest) -> str:
    """Identify a failed request in logs without recording the full question."""
    fields = {
        "capability": request.capability,
        "scene_id": request.scene_id,
        "scene_id_2": request.scene_id_2,
        "question": request.question,
    }
    return " ".join(
        f"{name}={log_excerpt(value) if value is not None else None!r}"
        for name, value in fields.items()
    )


@router.post("/scenes", response_model=SceneUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_scene(
    file: UploadFile = File(...),
    modality: str | None = Form(default=None),
    sensor: str | None = Form(default=None),
    acquisition_timestamp: str | None = Form(default=None),
    polarization: str | None = Form(default=None),
    pair_group: str | None = Form(default=None),
    benchmark_source: str | None = Form(default=None),
) -> SceneUploadResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 20 MiB limit.")
    try:
        return SceneUploadResponse.model_validate(
            ingest_scene(
                data,
                file.filename or "upload",
                {
                    "modality": modality,
                    "sensor": sensor,
                    "acquisition_timestamp": acquisition_timestamp,
                    "polarization": polarization,
                    "pair_group": pair_group,
                    "benchmark_source": benchmark_source,
                },
            )
        )
    except InvalidImageUpload as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SceneStorageError as exc:
        logger.exception(
            "Scene upload could not be stored (filename=%r, size=%d bytes)",
            log_excerpt(file.filename),
            len(data),
        )
        raise HTTPException(status_code=500, detail="The uploaded image could not be stored.") from exc


@router.get("/scenes")
def scenes() -> dict:
    try:
        return scene_catalog()
    except ScenePackError as exc:
        logger.exception("Curated scene pack is unavailable")
        raise HTTPException(status_code=503, detail="Curated scene pack is unavailable.") from exc


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse.model_validate(capabilities_overview())


@router.post("/plan", response_model=PlanResponse)
def plan(request: AnalyzeRequest) -> PlanResponse:
    try:
        return PlanResponse.model_validate(
            plan_analysis(
                request.scene_id,
                request.question,
                request.sensor,
                request.capability,
                request.scene_id_2,
            )
        )
    except (InvalidPlanRequest, UnknownCapability) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/analyze", response_model=AnalyzeResponse, response_model_exclude_unset=True
)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return AnalyzeResponse.model_validate(
            analyze_scene(
                request.scene_id,
                request.question,
                request.sensor,
                request.capability,
                request.scene_id_2,
                request.execution_mode,
            )
        )
    except ArtifactError as exc:
        logger.exception("Analysis artifact unavailable (%s)", _request_context(request))
        raise HTTPException(
            status_code=503, detail="Required analysis artifacts are temporarily unavailable."
        ) from exc
    except SceneNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (AnalysisUnavailable, InvalidPlanRequest) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PairCompatibilityError as exc:
        raise HTTPException(status_code=422, detail=exc.result) from exc
    except UnknownCapability as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderNotReady as exc:
        logger.warning(
            "Provider not ready: %s/%s reason=%s (%s)",
            exc.capability,
            exc.provider,
            exc.reason_code,
            _request_context(request),
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "capability": exc.capability,
                "provider": exc.provider,
                "reason_code": exc.reason_code,
                "detail": exc.detail,
            },
        ) from exc
    except CapabilityUnavailable as exc:
        logger.warning(
            "Capability unavailable (%s)", _request_context(request), exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail="Required capability is not currently available.",
        ) from exc
    # ModelUnavailable / ModelExecutionError are logged with tracebacks where
    # backend.services raises them, so they are not logged again here.
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail="Live model inference is unavailable.") from exc
    except ModelExecutionError as exc:
        raise HTTPException(status_code=502, detail="Model execution failed.") from exc
    except InvalidModelOutput as exc:
        logger.exception("Model returned invalid output (%s)", _request_context(request))
        raise HTTPException(status_code=502, detail="Model returned an invalid response.") from exc
    except TracePersistenceError as exc:
        logger.exception("Execution trace persistence failed (%s)", _request_context(request))
        raise HTTPException(
            status_code=503,
            detail="Execution trace is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected analysis failure (%s)", _request_context(request))
        raise HTTPException(status_code=502, detail="Model execution failed.") from exc


@router.get("/scenes/{scene_id}/image")
def scene_image(scene_id: str) -> FileResponse:
    image_path = local_scene_image(scene_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Local scene pixels are unavailable.")
    media_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(image_path, media_type=media_type)
