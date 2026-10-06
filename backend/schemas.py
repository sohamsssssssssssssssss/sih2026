"""Typed request and response contracts for the SatQuery API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_QUESTION_LENGTH = 2000
MAX_PAIR_GROUP_LENGTH = 256  # the declared-metadata limit in backend.services


class AnalyzeRequest(BaseModel):
    scene_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    execution_mode: Literal["live", "cached_result"] = "live"
    sensor: str | None = None
    capability: str | None = Field(
        default=None,
        description="Optional explicit capability; omitted requests are planned deterministically.",
    )
    scene_id_2: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional second scene for pairwise capabilities such as change_vqa. "
            "Omitting it keeps the single-scene planning behaviour unchanged."
        ),
    )


class SceneUploadResponse(BaseModel):
    scene_id: str
    filename: str
    format: Literal["PNG", "JPEG", "TIFF"]
    width: int
    height: int
    sensor: str | None = None
    gsd: str | None = None
    location: str | None = None
    acquisition_date: str | None = None


class UploadedScene(BaseModel):
    scene_id: str
    filename: str
    format: Literal["PNG", "JPEG", "TIFF"]
    width: int
    height: int
    modality: Literal["optical", "multispectral", "sar", "unknown"]
    sensor: str | None
    acquisition_time: str | None
    has_native_raster: bool
    georeferenced: bool
    uploaded_at: str = Field(description="ISO 8601 UTC upload time.")


class SceneCatalogResponse(BaseModel):
    version: str
    scenes: list[dict[str, Any]] = Field(description="Curated scene-pack entries.")
    uploads: list[UploadedScene] = Field(
        description="Scenes uploaded through POST /api/scenes, newest first (capped)."
    )


class ModelInfo(BaseModel):
    name: str
    version: str


class AnalyzeResponse(BaseModel):
    answer: str
    evidence: list[dict[str, Any]] | None = None
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
    registered: bool
    available: bool
    state: Literal["AVAILABLE", "UNAVAILABLE", "NOT_IMPLEMENTED"]
    provider: str | None
    reason_code: str | None
    detail: str | None


class CapabilitiesResponse(BaseModel):
    capabilities: list[CapabilityStatus]


class TraceHealth(BaseModel):
    ok: bool
    detail: str | None


class CapabilityHealth(BaseModel):
    available: bool
    reason_code: str | None


class HealthChecks(BaseModel):
    trace: TraceHealth
    capabilities: dict[str, CapabilityHealth]


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    mode: Literal["offline-first"] = "offline-first"
    checks: HealthChecks


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


class DayWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str = Field(description="First UTC day, YYYY-MM-DD.")
    end: str = Field(description="Last UTC day, YYYY-MM-DD; included.")


class Sentinel1PairRequest(BaseModel):
    """Either a catalogued event_id, or all of bbox, before and after."""

    model_config = ConfigDict(extra="forbid")

    event_id: str | None = Field(default=None, min_length=1, description="A catalogued flood event.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="WEST, SOUTH, EAST, NORTH in degrees."
    )
    before: DayWindow | None = None
    after: DayWindow | None = None
    pair_group: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PAIR_GROUP_LENGTH,
        description="Defaults to event_id, else to the two CDSE product ids.",
    )

    @model_validator(mode="after")
    def exactly_one_form(self) -> "Sentinel1PairRequest":
        explicit = (self.bbox, self.before, self.after)
        if self.event_id is not None and any(value is not None for value in explicit):
            raise ValueError("event_id replaces bbox, before and after")
        if self.event_id is None and any(value is None for value in explicit):
            raise ValueError("give event_id, or all of bbox, before and after")
        return self


class Sentinel1Scene(BaseModel):
    scene_id: str
    acquisition_id: str
    acquisition_time: str = Field(description="ISO 8601 UTC acquisition time.")
    orbit_state: str
    relative_orbit: int | None
    valid_fraction: float = Field(description="Fraction of pixels covered and not in radar shadow.")


class Sentinel1PairResponse(BaseModel):
    pair_group: str
    crs: str
    width: int
    height: int
    before: Sentinel1Scene
    after: Sentinel1Scene
