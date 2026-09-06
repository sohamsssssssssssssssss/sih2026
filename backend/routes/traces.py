from fastapi import APIRouter

from backend.schemas import TraceVerification
from orchestrator.trace import records, verify_chain

router = APIRouter(prefix="/api", tags=["traces"])


@router.get("/traces")
def traces() -> dict:
    chain = records()
    return {"records": list(reversed(chain)), "count": len(chain)}


@router.post("/traces/verify", response_model=TraceVerification)
def verify_traces() -> TraceVerification:
    verified, message = verify_chain()
    return TraceVerification(verified=verified, message=message)
