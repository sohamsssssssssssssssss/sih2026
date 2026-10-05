"""FastAPI entrypoint for the local SatQuery AI migration."""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import analyze, resolution, sar, traces
from backend.schemas import HealthResponse
from backend.services import capabilities_overview
from orchestrator import trace as trace_store

app = FastAPI(
    title="SatQuery AI API",
    description="Offline-first read API over SatQuery's existing orchestration and committed artifacts.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(analyze.router)
app.include_router(resolution.router)
app.include_router(sar.router)
app.include_router(traces.router)


def _trace_check() -> dict[str, object]:
    """Load and verify the persisted trace chain; analysis cannot run without it."""
    try:
        trace_store.records()
    except (trace_store.TraceIntegrityError, OSError) as exc:
        return {"ok": False, "detail": str(exc) or type(exc).__name__}
    return {"ok": True, "detail": None}


def _capability_checks() -> dict[str, dict[str, object]]:
    """Per-capability provider readiness; never loads weights or runs inference."""
    return {
        str(entry["name"]): {
            "available": bool(entry["available"]),
            "reason_code": entry.get("reason_code"),
        }
        for entry in capabilities_overview()["capabilities"]
    }


@app.get(
    "/api/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Trace history is unusable."}},
)
def health(response: Response) -> HealthResponse:
    trace_check = _trace_check()
    capability_checks = _capability_checks()
    if not trace_check["ok"]:
        status = "unavailable"
        response.status_code = 503
    elif all(check["available"] for check in capability_checks.values()):
        status = "ready"
    else:
        status = "degraded"
    return HealthResponse.model_validate(
        {
            "status": status,
            "mode": "offline-first",
            "checks": {"trace": trace_check, "capabilities": capability_checks},
        }
    )
