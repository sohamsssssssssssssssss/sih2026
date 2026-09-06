from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services import ArtifactError, sar_annotation, sar_render_path

router = APIRouter(prefix="/api", tags=["sar"])


@router.get("/sar/{scene}")
def sar(scene: str) -> dict:
    try:
        return sar_annotation(scene)
    except ArtifactError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sar/{scene}/image")
def sar_image(scene: str) -> FileResponse:
    image_path = sar_render_path(scene)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Local processed SAR render is unavailable.")
    return FileResponse(image_path, media_type="image/png")
