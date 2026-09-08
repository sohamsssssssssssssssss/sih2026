from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.schemas import AnalyzeRequest, AnalyzeResponse, CapabilitiesResponse, SceneUploadResponse
from backend.services import (
    AnalysisUnavailable,
    ArtifactError,
    InvalidImageUpload,
    ModelExecutionError,
    ModelUnavailable,
    SceneStorageError,
    analyze_scene,
    capabilities_overview,
    ingest_scene,
    local_scene_image,
)
from orchestrator.capabilities import CapabilityUnavailable, UnknownCapability
from orchestrator.router import InvalidModelOutput, TracePersistenceError

router = APIRouter(prefix="/api", tags=["analysis"])
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/scenes", response_model=SceneUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_scene(file: UploadFile = File(...)) -> SceneUploadResponse:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the 20 MiB limit.")
    try:
        return SceneUploadResponse.model_validate(
            ingest_scene(data, file.filename or "upload")
        )
    except InvalidImageUpload as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SceneStorageError as exc:
        raise HTTPException(status_code=500, detail="The uploaded image could not be stored.") from exc


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse.model_validate(capabilities_overview())


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return AnalyzeResponse.model_validate(
            analyze_scene(request.scene_id, request.question, request.sensor, request.capability)
        )
    except ArtifactError as exc:
        raise HTTPException(
            status_code=503, detail="Required analysis artifacts are temporarily unavailable."
        ) from exc
    except AnalysisUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnknownCapability as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CapabilityUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail="Live model inference is unavailable.") from exc
    except ModelExecutionError as exc:
        raise HTTPException(status_code=502, detail="Model execution failed.") from exc
    except InvalidModelOutput as exc:
        raise HTTPException(status_code=502, detail="Model returned an invalid response.") from exc
    except TracePersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail="Execution trace is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Model execution failed.") from exc


@router.get("/scenes/{scene_id}/image")
def scene_image(scene_id: str) -> FileResponse:
    image_path = local_scene_image(scene_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Local scene pixels are unavailable.")
    return FileResponse(image_path, media_type="image/png")
