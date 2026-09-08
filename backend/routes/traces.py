from fastapi import APIRouter, HTTPException

from backend.schemas import TraceVerification
from orchestrator.trace import TraceIntegrityError, records, verify_chain

router = APIRouter(prefix="/api", tags=["traces"])
TRACE_UNAVAILABLE = (
    "Execution trace is temporarily unavailable because persisted trace integrity "
    "could not be verified."
)


@router.get("/traces")
def traces() -> dict:
    try:
        chain = records()
    except TraceIntegrityError as exc:
        raise HTTPException(status_code=503, detail=TRACE_UNAVAILABLE) from exc
    return {"records": list(reversed(chain)), "count": len(chain)}


@router.post("/traces/verify", response_model=TraceVerification)
def verify_traces() -> TraceVerification:
    try:
        verified, message = verify_chain()
    except TraceIntegrityError as exc:
        raise HTTPException(status_code=503, detail=TRACE_UNAVAILABLE) from exc
    return TraceVerification(verified=verified, message=message)
