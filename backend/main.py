"""FastAPI entrypoint for the local SatQuery AI migration."""

import os
import re

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import analyze, resolution, sar, traces
from backend.schemas import HealthResponse
from backend.services import capabilities_overview
from orchestrator import trace as trace_store

CORS_ORIGINS_ENV = "SATQUERY_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
LOCALHOST_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1):\d+"
# One exact browser origin: scheme, host, optional port. No path, query,
# userinfo, or wildcard, because CORS compares the Origin header verbatim.
_EXACT_ORIGIN = re.compile(r"https?://[A-Za-z0-9.-]+(:\d{1,5})?")


def cors_origins(raw: str | None) -> list[str]:
    """Localhost defaults plus exact origins from a comma-separated value.

    A deployment behind one reverse proxy serves the UI and ``/api`` from the
    same origin and needs nothing here; this is only for a UI on another
    origin. Malformed entries raise instead of being dropped, so a typo is a
    startup failure rather than a silently blocked browser.
    """
    origins = list(DEFAULT_CORS_ORIGINS)
    for entry in (raw or "").split(","):
        origin = entry.strip()
        if not origin:
            continue
        if origin.endswith("/") and origin.count("/") == 3:
            origin = origin[:-1]
        if "*" in origin or not _EXACT_ORIGIN.fullmatch(origin):
            raise ValueError(
                f"{CORS_ORIGINS_ENV} entries must be exact origins such as "
                f"https://satquery.example.org; got {entry.strip()!r}"
            )
        if origin not in origins:
            origins.append(origin)
    return origins


app = FastAPI(
    title="SatQuery AI API",
    description="Offline-first read API over SatQuery's existing orchestration and committed artifacts.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(os.environ.get(CORS_ORIGINS_ENV)),
    allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
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
