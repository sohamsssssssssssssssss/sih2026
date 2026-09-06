from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.schemas import AnalyzeRequest, AnalyzeResponse
from backend.services import AnalysisUnavailable, ArtifactError, analyze_scene, local_scene_image

router = APIRouter(prefix="/api", tags=["analysis"])


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
