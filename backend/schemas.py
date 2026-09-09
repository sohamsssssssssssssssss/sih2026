"""Typed request and response contracts for the SatQuery API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_QUESTION_LENGTH = 2000


class AnalyzeRequest(BaseModel):
    scene_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    sensor: str | None = None
    capability: str | None = Field(
        default=None,
        description="Optional explicit capability; omitted requests are planned deterministically.",
    )


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


class CapabilityStatus(BaseModel):
    name: str
    available: bool
    provider: str | None


class CapabilitiesResponse(BaseModel):
    capabilities: list[CapabilityStatus]


class PlanStepSummary(BaseModel):
    step_id: str
    capability: str
    depends_on: list[str]
    required_inputs: list[str]
    provider_available: bool
    provider: str | None


class PlanResponse(BaseModel):
    planner_version: str
    rule_id: str
    requested_capability: str | None
    selected_capability: str
    executable: bool
    reason: str
    required_inputs: list[str]
    missing_inputs: list[str]
    provider_available: bool
    provider: str | None
    unavailable_reason: str | None
    execution_plan_version: str
    steps: list[PlanStepSummary]
    unavailable_capabilities: list[str]
