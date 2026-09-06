"""FastAPI entrypoint for the local SatQuery AI migration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import analyze, resolution, sar, traces

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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ready", "mode": "offline-first"}
