from fastapi import APIRouter, HTTPException

from backend.services import ArtifactError, resolution_report

router = APIRouter(prefix="/api", tags=["resolution"])


@router.get("/resolution")
def resolution() -> dict:
    try:
        return resolution_report()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
