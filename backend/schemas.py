"""Typed request and response contracts for the SatQuery API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_QUESTION_LENGTH = 2000


class AnalyzeRequest(BaseModel):
    scene_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    sensor: str | None = None


class SceneUploadResponse(BaseModel):
    scene_id: str
    filename: str
    format: Literal["PNG", "JPEG"]
    width: int
    height: int
    sensor: None = None
    gsd: None = None
    location: None = None
    acquisition_date: None = None


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
