"""T06 contract tests. No network, no real data."""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from core import schema as S
from core.schema_version import SCHEMA_VERSION

H64, H40 = "a" * 64, "b" * 40


def tool_result(**kw):
    return S.ToolResult(**{"tool": "landcover", "tool_version": "0.1", "checkpoint_sha": H64, "value": {"cropland": 0.41},
                           "raw_confidence": 0.8, "evidence": [S.Evidence(kind="fractions", summary={"cropland": 0.41})],
                           "latency_ms": 120, **kw})


VALID = {
    S.Scene: dict(scene_id=H64, sensor="S2", bands=["B02", "B03", "B04", "B08"], units="reflectance_x1e4", gsd_m=10.0,
                  crs="EPSG:32643", bounds=(0.0, 0.0, 1200.0, 1200.0), acquired=date(2021, 6, 1), cloud_pct=3.5,
                  nodata_pct=0.0, source="wc-pack", region="punjab"),
    S.Patch: dict(patch_id="p1", scene_id=H64, window=(0, 0, 120, 120), split="holdout_india", region="assam",
                  labels={"fractions": [0.5, 0.5]}),
    S.PackManifest: dict(pack_id="wc", version="1", created=datetime(2026, 11, 16, tzinfo=timezone.utc), builder_commit=H40,
                         files={"s1.npy": H64}, n_patches=3, splits={"test": 2, "val": 1}, regions={"punjab": 3},
                         band_order=["VV", "VH"], units="dB", norm_stats={"VV": [-12.7, 5.1]}, checks={"nodata": True},
                         sources=["https://example.invalid/x"], licence="unknown", redistribution="unknown"),
    S.Query: dict(text="How much cropland?", qtype="landcover", needs_gsd_m=10.0, needs_sensor=["S2"]),
    S.Evidence: dict(kind="mask", uri="masks/m.tif", summary={"water": 0.07}),
    S.ToolResult: dict(tool="det", tool_version="1", value=3, latency_ms=0),
    S.Answer: dict(status="answered", text="41% cropland", confidence=0.7, sensors_used=["S2"], primary=None,
                   verification={"cross_sensor": "agree"}, trace_id="t1"),
    S.EvalRecord: dict(eval_id="e1", commit=H40, pack_id="wc", metric="macro_map", value=0.6, n=300, ci95=(0.5, 0.7),
                       control="shuffled", hypothesis_ref="docs/hyp/t11.md"),
    S.ToolStatus: dict(name="landcover", version="0.1", measured={"macro_map": 0.6254}, evidence_refs=["RESULTS.md#R2"]),
}
VALID[S.Answer]["primary"] = tool_result()


def make(model, **kw):
    return model(**{**VALID[model], **kw})


@pytest.mark.parametrize("model", S.MODELS, ids=lambda m: m.__name__)
def test_valid_construction_and_json_round_trip(model):
    obj = make(model)
    assert model.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize("model", S.MODELS, ids=lambda m: m.__name__)
def test_frozen_and_extra_forbidden(model):
    obj = make(model)
    field = next(iter(model.model_fields))
    with pytest.raises(ValidationError):
        setattr(obj, field, getattr(obj, field))
    with pytest.raises(ValidationError, match="Extra inputs"):
        make(model, surprise=1)


@pytest.mark.parametrize("model, bad", [
    (S.Scene, dict(bounds=(5.0, 0.0, 1.0, 1.0))),
    (S.Scene, dict(bounds=(0.0, 5.0, 1.0, 1.0))),
    (S.Scene, dict(gsd_m=0)),
    (S.Scene, dict(cloud_pct=101)),
    (S.Scene, dict(nodata_pct=-1)),
    (S.Scene, dict(scene_id="not-a-hash")),
    (S.Scene, dict(sensor="Landsat")),
    (S.Patch, dict(window=(-1, 0, 10, 10))),
    (S.Patch, dict(window=(0, 0, 0, 10))),
    (S.Patch, dict(split="holdout")),
    (S.PackManifest, dict(splits={"test": 1})),
    (S.PackManifest, dict(created=datetime(2026, 11, 16))),
    (S.PackManifest, dict(created=datetime(2026, 11, 16, tzinfo=timezone(timedelta(hours=5, minutes=30))))),
    (S.PackManifest, dict(builder_commit="abc")),
    (S.PackManifest, dict(files={"s1.npy": "xyz"})),
    (S.PackManifest, dict(redistribution="maybe")),
    (S.Query, dict(text="   ")),
    (S.Query, dict(qtype="weather")),
    (S.ToolResult, dict(checkpoint_sha="short")),
    (S.ToolResult, dict(raw_confidence=1.5)),
    (S.ToolResult, dict(latency_ms=-1)),
    (S.Answer, dict(confidence=1.2)),
    (S.Answer, dict(verification={"x": "maybe"})),
    (S.EvalRecord, dict(n=0)),
    (S.EvalRecord, dict(ci95=(0.7, 0.8))),
    (S.EvalRecord, dict(value=float("nan"), ci95=None)),
    (S.EvalRecord, dict(control="random")),
    (S.EvalRecord, dict(commit=H64)),
    (S.ToolStatus, dict(status="beta")),
])
def test_field_constraints_reject(model, bad):
    with pytest.raises(ValidationError):
        make(model, **bad)


@pytest.mark.parametrize("bad", [
    dict(status="refused", refusal_reason=None, confidence=None),          # refused without reason
    dict(status="refused", refusal_reason="cloud", confidence=0.5),        # refused with confidence
    dict(status="answered", refusal_reason="cloud"),                       # answered with reason
    dict(status="answered", primary=None),                                 # answered without primary
    dict(status="low_confidence", primary=None),                           # low_confidence without primary
])
def test_answer_invariants_reject(bad):
    with pytest.raises(ValidationError):
        make(S.Answer, **bad)


def test_answer_invariants_accept():
    refused = make(S.Answer, status="refused", refusal_reason="gsd_too_coarse", confidence=None, primary=None)
    assert refused.primary is None
    assert make(S.Answer, status="low_confidence", confidence=0.3).status == "low_confidence"


def test_scene_id_deterministic_and_pixel_sensitive():
    px = np.zeros((2, 4, 4), np.uint16)
    meta = {"sensor": "S2", "bands": ["B02", "B03"], "gsd_m": 10.0}
    a = S.scene_id_for(px.tobytes(), meta)
    assert a == S.scene_id_for(px.tobytes(), dict(reversed(list(meta.items()))))  # key order irrelevant
    px2 = px.copy(); px2[1, 3, 3] = 1
    assert S.scene_id_for(px2.tobytes(), meta) != a
    assert S.scene_id_for(px.tobytes(), {**meta, "gsd_m": 20.0}) != a
    assert make(S.Scene, scene_id=a).scene_id == a


def test_tool_status_fails_closed():
    assert S.ToolStatus(name="new", version="0").status == "experimental_unvalidated"


def test_export_json_schemas(tmp_path):
    paths = S.export_json_schemas(tmp_path)
    assert sorted(p.name for p in paths) == sorted(f"{m.__name__}.json" for m in S.MODELS)
    for p in paths:
        schema = json.loads(p.read_text())
        assert schema["x-schema-version"] == SCHEMA_VERSION and schema["title"] == p.stem


def test_committed_schemas_are_current(tmp_path):
    committed = S.Path(S.__file__).resolve().parents[1] / "docs" / "schemas"
    for p in S.export_json_schemas(tmp_path):
        assert (committed / p.name).read_text() == p.read_text(), f"docs/schemas/{p.name} stale: run python -m core.schema"
