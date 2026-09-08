from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.schemas import AnalyzeRequest, AnalyzeResponse, SceneUploadResponse
from backend.services import (
    AnalysisUnavailable,
    ArtifactError,
    InvalidImageUpload,
    SceneStorageError,
    analyze_scene,
    ingest_scene,
    local_scene_image,
)

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


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return AnalyzeResponse.model_validate(
            analyze_scene(request.scene_id, request.question, request.sensor)
        )
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AnalysisUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/scenes/{scene_id}/image")
def scene_image(scene_id: str) -> FileResponse:
    image_path = local_scene_image(scene_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Local scene pixels are unavailable.")
    return FileResponse(image_path, media_type="image/png")
