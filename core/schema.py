"""SatQuery data contracts (MASTER_PLAN Part 4.2). Contracts only: no tools, API or models.

Every model is frozen, rejects unknown fields and NaN/inf, and round-trips through JSON.
Regenerate docs/schemas/ with `python -m core.schema`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from core.schema_version import SCHEMA_VERSION
from orchestrator.trace import _canonical  # the trace's canonical JSON; one hashing scheme repo-wide

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
Unit = Annotated[float, Field(ge=0, le=1)]
Pct = Annotated[float, Field(ge=0, le=100)]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def scene_id_for(pixels: bytes, metadata: dict[str, Any]) -> str:
    """sha256 hex of pixel bytes + canonical metadata JSON (same canonical form as orchestrator.trace)."""
    return hashlib.sha256(pixels + _canonical(metadata).encode("utf-8")).hexdigest()


class Scene(_Contract):
    scene_id: Sha256
    sensor: Literal["S2", "S1", "RGB_HR", "S1S2", "UNKNOWN"]
    bands: list[str]
    units: Literal["reflectance_x1e4", "dB", "uint8", "unknown"]
    gsd_m: Annotated[float, Field(gt=0)]
    crs: NonEmpty
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    acquired: date | None = None
    cloud_pct: Pct | None = None
    nodata_pct: Pct
    source: NonEmpty
    region: str | None = None

    @field_validator("bounds")
    @classmethod
    def _ordered(cls, b):
        if not (b[0] < b[2] and b[1] < b[3]):
            raise ValueError("bounds must be (minx, miny, maxx, maxy) with minx<maxx and miny<maxy")
        return b


class Patch(_Contract):
    patch_id: NonEmpty
    scene_id: Sha256
    window: tuple[int, int, int, int]  # row_off, col_off, height, width
    split: Literal["train", "val", "test", "holdout_india"]
    region: NonEmpty
    labels: dict[str, Any]

    @field_validator("window")
    @classmethod
    def _window(cls, w):
        if min(w) < 0 or w[2] == 0 or w[3] == 0:
            raise ValueError("window offsets must be >=0 and height,width >0")
        return w


class PackManifest(_Contract):
    pack_id: NonEmpty
    version: NonEmpty
    created: datetime
    builder_commit: GitSha
    files: dict[str, Sha256]
    n_patches: Annotated[int, Field(ge=0)]
    splits: dict[str, Annotated[int, Field(ge=0)]]
    regions: dict[str, Annotated[int, Field(ge=0)]]
    band_order: list[str]
    units: NonEmpty
    norm_stats: dict[str, Any]
    checks: dict[str, bool]
    sources: list[str]
    licence: NonEmpty  # "unknown" allowed
    redistribution: Literal["allowed", "not_allowed", "unknown"]

    @field_validator("created")
    @classmethod
    def _utc(cls, t):
        if t.utcoffset() != timedelta(0):
            raise ValueError("created must be timezone-aware UTC")
        return t

    @model_validator(mode="after")
    def _splits_sum(self):
        if sum(self.splits.values()) != self.n_patches:
            raise ValueError(f"splits sum to {sum(self.splits.values())}, n_patches is {self.n_patches}")
        return self


class Query(_Contract):
    text: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    qtype: Literal["landcover", "presence", "flood", "change", "count", "locate", "open", "context"]
    needs_gsd_m: Annotated[float, Field(gt=0)] | None = None
    needs_sensor: list[str] = []


class Evidence(_Contract):
    kind: Literal["mask", "boxes", "fractions", "diff", "text"]
    uri: str | None = None
    summary: dict[str, Any] = {}


class ToolResult(_Contract):
    tool: NonEmpty
    tool_version: NonEmpty
    checkpoint_sha: Sha256 | None = None
    value: Any
    raw_confidence: Unit | None = None
    evidence: list[Evidence] = []
    latency_ms: Annotated[int, Field(ge=0)]


class Answer(_Contract):
    status: Literal["answered", "refused", "low_confidence"]
    text: str
    confidence: Unit | None = None
    refusal_reason: Literal["gsd_too_coarse", "cloud", "sensor_missing", "tool_unvalidated",
                            "checks_disagree", "out_of_distribution"] | None = None
    sensors_used: list[str] = []
    primary: ToolResult | None = None
    verification: dict[str, Literal["agree", "disagree", "skipped"]] = {}
    trace_id: NonEmpty

    @model_validator(mode="after")
    def _status_invariants(self):
        if self.status == "refused":
            if self.refusal_reason is None:
                raise ValueError("refused answers need a refusal_reason")
            if self.confidence is not None:
                raise ValueError("refused answers carry no confidence")
        elif self.status == "answered":
            if self.refusal_reason is not None:
                raise ValueError("answered answers have no refusal_reason")
            if self.primary is None:
                raise ValueError("answered answers need a primary ToolResult")
        elif self.primary is None:  # low_confidence
            raise ValueError("low_confidence answers need a primary ToolResult")
        return self


class EvalRecord(_Contract):
    eval_id: NonEmpty
    commit: GitSha
    pack_id: NonEmpty
    checkpoint_sha: Sha256 | None = None
    metric: NonEmpty
    value: float
    n: Annotated[int, Field(gt=0)]
    ci95: tuple[float, float] | None = None
    control: Literal["blank", "shuffled", "swapped", "identical", "degraded"] | None = None
    hypothesis_ref: NonEmpty  # path to the pre-committed hypothesis

    @model_validator(mode="after")
    def _ci_brackets_value(self):
        if self.ci95 is not None and not (self.ci95[0] <= self.value <= self.ci95[1]):
            raise ValueError(f"ci95 {self.ci95} must satisfy lo <= value ({self.value}) <= hi")
        return self


class ToolStatus(_Contract):
    name: NonEmpty
    version: NonEmpty
    status: Literal["validated", "experimental_unvalidated", "disabled"] = "experimental_unvalidated"  # fail closed
    measured: dict[str, float] = {}
    evidence_refs: list[str] = []


MODELS: tuple[type[_Contract], ...] = (Scene, Patch, PackManifest, Query, Evidence, ToolResult, Answer, EvalRecord, ToolStatus)


def export_json_schemas(out_dir: str | Path = Path(__file__).resolve().parents[1] / "docs" / "schemas") -> list[Path]:
    """Write one JSON Schema per model to out_dir (default docs/schemas/); returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in MODELS:
        schema = {"x-schema-version": SCHEMA_VERSION, **model.model_json_schema()}
        path = out / f"{model.__name__}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in export_json_schemas():
        print(p)
