import logging
from threading import Lock

import httpx
from fastapi import APIRouter, HTTPException, status

from backend.schemas import DayWindow, Sentinel1PairRequest, Sentinel1PairResponse
from backend.sentinel1 import CREDENTIALS_HELP, CDSEError, cdse_credentials, fetch_pair, pair_request
from backend.services import InvalidImageUpload, SceneStorageError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["sentinel1"])
REDACTED = "[redacted]"
FETCH_IN_PROGRESS = "A Sentinel-1 fetch is already in progress; retry when it finishes."
# ponytail: one fetch per process guards the free CDSE quota; several uvicorn
# workers would each hold their own lock and would need a file lock instead.
_FETCH_LOCK = Lock()


def cdse_client() -> httpx.Client:
    """The HTTP client for CDSE; tests swap in a mock transport here."""
    return httpx.Client()


def _days(window: DayWindow | None) -> tuple[str, str] | None:
    return (window.start, window.end) if window else None


@router.post(
    "/sentinel1/pairs", response_model=Sentinel1PairResponse, status_code=status.HTTP_201_CREATED
)
def fetch_sentinel1_pair(request: Sentinel1PairRequest) -> Sentinel1PairResponse:
    """Search, pair, render and ingest; blocks for minutes, so FastAPI runs it in its threadpool."""
    try:
        bbox, before, after = pair_request(
            request.event_id, request.bbox, _days(request.before), _days(request.after)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    credentials = cdse_credentials()
    if credentials is None:
        raise HTTPException(status_code=503, detail=CREDENTIALS_HELP)
    if not _FETCH_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=FETCH_IN_PROGRESS)
    try:
        with cdse_client() as client:
            result = fetch_pair(
                client, *credentials, bbox, before, after, request.pair_group or request.event_id
            )
    # InvalidImageUpload is a ValueError, but here it means CDSE sent an unusable raster.
    except (CDSEError, InvalidImageUpload) as exc:
        # Upstream error excerpts are untrusted; never pass the secret back out.
        detail = str(exc).replace(credentials[1], REDACTED)
        logger.warning("Sentinel-1 fetch failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except ValueError as exc:  # bbox, AOI size or window order; raised before any CDSE call
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SceneStorageError as exc:
        logger.exception("A fetched Sentinel-1 scene could not be stored")
        raise HTTPException(status_code=500, detail="The fetched scene could not be stored.") from exc
    finally:
        _FETCH_LOCK.release()
    return Sentinel1PairResponse.model_validate(result)
