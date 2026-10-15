from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.sar_change_artifacts import verified_change_artifacts
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


@router.get("/sar/change/kosi-2024")
def kosi_change() -> dict:
    _, metadata = verified_change_artifacts()
    return metadata


@router.get("/sar/change/kosi-2024/raster")
def kosi_change_raster() -> FileResponse:
    raster, _ = verified_change_artifacts()
    return FileResponse(raster, media_type="image/tiff", filename="kosi-2024-sar-change.tif")
