"""Typed request and response contracts for the SatQuery API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    scene_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    sensor: str = "LoveDA"


class ModelInfo(BaseModel):
    name: str
    version: str


class AnalyzeResponse(BaseModel):
    answer: str
    execution_mode: Literal["live", "cached_result"]
    results_artifact: str | None = None
    model: ModelInfo
    trace: dict[str, Any]
    notice: str


class TraceVerification(BaseModel):
    verified: bool
    message: str
