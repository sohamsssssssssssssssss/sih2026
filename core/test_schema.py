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


# --- R01 I1/I2: payloads must be JSON values, rejected at construction -----------------

DEEP_PAYLOAD = {
    "fractions": [0.1, None, True, 2, "x"],
    "nested": {"list": [{"ok": False}], "empty": []},
    "flag": None,
}


@pytest.mark.parametrize("bad", [
    float("nan"),
    float("inf"),
    float("-inf"),
    {"a": [float("nan")]},
    ["ok", {"deep": float("inf")}],
    {"a": {"b": [1, float("-inf")]}},
    b"bytes",
    b"\xff",
    object(),
    (1, 2),
    {1, 2},
    {1: "a"},
    {"a": (1, 2)},
    ["ok", {2: "b"}],
], ids=["nan", "inf", "-inf", "nested-nan", "list-inf", "deep-inf", "bytes", "invalid-utf8", "object",
        "tuple", "set", "int-key", "nested-tuple", "nested-int-key"])
def test_non_json_payloads_rejected_everywhere(bad):
    with pytest.raises(ValidationError):
        tool_result(value=bad)
    with pytest.raises(ValidationError):
        make(S.Patch, labels={"x": bad})
    with pytest.raises(ValidationError):
        make(S.Evidence, summary={"x": bad})
    with pytest.raises(ValidationError):
        make(S.PackManifest, norm_stats={"VV": bad})


def test_numpy_payload_rejected_with_guidance():
    with pytest.raises(ValidationError, match=r"ndarray is not valid JSON at value; call \.tolist\(\)"):
        tool_result(value=np.zeros((2, 2)))
    with pytest.raises(ValidationError, match=r"\.tolist\(\)"):
        make(S.Patch, labels={"mask": np.zeros((2, 2), dtype=bool)})


def test_payload_error_names_field_and_path():
    with pytest.raises(ValidationError, match=r"non-finite float nan is not valid JSON at value\[0\]"):
        tool_result(value=[float("nan")])
    with pytest.raises(ValidationError, match=r"is not valid JSON at labels\['a'\]\[1\]"):
        make(S.Patch, labels={"a": [1.0, float("inf")]})


def test_valid_payloads_are_accepted():
    assert tool_result(value=DEEP_PAYLOAD).value == DEEP_PAYLOAD
    assert make(S.Patch, labels=DEEP_PAYLOAD).labels == DEEP_PAYLOAD
    assert make(S.Evidence, summary=DEEP_PAYLOAD).summary == DEEP_PAYLOAD
    assert make(S.PackManifest, norm_stats={"VV": DEEP_PAYLOAD}).norm_stats["VV"] == DEEP_PAYLOAD


# --- R01 I3: sequences are immutable tuples; mappings stay dicts -----------------------

def test_sequence_fields_are_tuples():
    sc = make(S.Scene)
    assert isinstance(sc.bands, tuple)
    with pytest.raises(AttributeError):
        sc.bands.append("B12")
    assert isinstance(tool_result().evidence, tuple)
    with pytest.raises(TypeError):
        tool_result().evidence[0] = None
    assert isinstance(make(S.Answer).sensors_used, tuple)
    assert isinstance(make(S.PackManifest).band_order, tuple)
    assert isinstance(make(S.Query).needs_sensor, tuple)
    assert isinstance(make(S.ToolStatus).evidence_refs, tuple)


def test_mapping_mutation_is_detected_by_revalidation():
    pm = make(S.PackManifest)
    dump = pm.model_dump()
    dump["splits"]["test"] = 99
    with pytest.raises(ValidationError, match="splits sum to 100"):
        S.PackManifest.model_validate(dump)


PAYLOAD_OVERRIDES = {
    S.Scene: {"bands": ("B02", "B03")},
    S.Patch: {"labels": DEEP_PAYLOAD},
    S.PackManifest: {"norm_stats": {"VV": DEEP_PAYLOAD}},
    S.Query: {"needs_sensor": ("S2", "S1")},
    S.Evidence: {"summary": DEEP_PAYLOAD},
    S.ToolResult: {"value": DEEP_PAYLOAD},
    S.Answer: {"primary": tool_result(value=DEEP_PAYLOAD)},
    S.EvalRecord: {},
    S.ToolStatus: {"evidence_refs": ("RESULTS.md#R2",)},
}


@pytest.mark.parametrize("model", S.MODELS, ids=lambda m: m.__name__)
def test_payload_round_trip_through_dump_and_json(model):
    obj = make(model, **PAYLOAD_OVERRIDES[model])
    assert model.model_validate(obj.model_dump()) == obj
    assert model.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize("model", [S.PackManifest, S.Answer, S.EvalRecord], ids=lambda m: m.__name__)
def test_after_validators_hold_on_revalidation(model):
    obj = make(model)
    assert model.model_validate(obj.model_dump()) == obj
    assert model.model_validate_json(obj.model_dump_json()) == obj


# --- R01 I4: counts and offsets are strict ints ----------------------------------------

@pytest.mark.parametrize("window", [(True, 0, 10, 10), (0, True, 10, 10), (0, 0, True, 10), (0, 0, 10, True)])
def test_window_rejects_bool_offsets(window):
    with pytest.raises(ValidationError):
        make(S.Patch, window=window)


def test_counts_reject_bool_and_float():
    with pytest.raises(ValidationError):
        make(S.EvalRecord, n=True)
    with pytest.raises(ValidationError):
        make(S.EvalRecord, n=3.0)
    with pytest.raises(ValidationError):
        tool_result(latency_ms=True)
    with pytest.raises(ValidationError):
        tool_result(latency_ms=1.0)
    with pytest.raises(ValidationError):
        make(S.PackManifest, n_patches=True)
    with pytest.raises(ValidationError):
        make(S.PackManifest, splits={"test": True, "val": 1})


# --- R01 "Untested rules" --------------------------------------------------------------

@pytest.mark.parametrize("window", [(0, 0, 10, 0), (0, 0, -5, 10), (0, 0, 10, -5), (0, 0, 0, 0), (-1, 0, 10, 10)])
def test_patch_window_boundaries(window):
    with pytest.raises(ValidationError):
        make(S.Patch, window=window)


def test_uppercase_hex_rejected():
    with pytest.raises(ValidationError):
        make(S.Scene, scene_id="A" * 64)
    with pytest.raises(ValidationError):
        tool_result(checkpoint_sha="A" * 64)
    with pytest.raises(ValidationError):
        make(S.EvalRecord, commit="B" * 40)


@pytest.mark.parametrize("bounds", [(0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0, 0.0), (5.0, 0.0, 5.0, 10.0)])
def test_scene_bounds_equality_rejected(bounds):
    with pytest.raises(ValidationError):
        make(S.Scene, bounds=bounds)


def test_scene_pct_boundaries():
    assert make(S.Scene, cloud_pct=100.0, nodata_pct=100.0).nodata_pct == 100.0
    with pytest.raises(ValidationError):
        make(S.Scene, cloud_pct=100.1)
    with pytest.raises(ValidationError):
        make(S.Scene, nodata_pct=100.1)


def test_scene_id_metadata_and_band_sensitivity():
    px = np.zeros((2, 4, 4), np.uint16)
    base = {"sensor": "S2", "bands": ["B02", "B03"], "nested": {"a": 1, "b": 2}}
    a = S.scene_id_for(px.tobytes(), base)
    # nested key order is irrelevant, outer key order too
    assert a == S.scene_id_for(px.tobytes(), {"nested": {"b": 2, "a": 1}, "bands": ["B02", "B03"], "sensor": "S2"})
    # a non-gsd metadata field and the band order both change the id
    assert a != S.scene_id_for(px.tobytes(), {**base, "sensor": "S1"})
    assert a != S.scene_id_for(px.tobytes(), {**base, "bands": ["B03", "B02"]})
    assert a != S.scene_id_for(px.tobytes(), {**base, "nested": {"a": 2, "b": 2}})
    # same bytes in a different shape give the same id: shape is not part of the contract
    assert a == S.scene_id_for(px.reshape(-1).tobytes(), base)


def test_pack_manifest_created_forms():
    with pytest.raises(ValidationError):
        make(S.PackManifest, created=date(2026, 11, 16))
    assert make(S.PackManifest, created="2026-11-16T00:00:00Z").created.utcoffset() == timedelta(0)
    weird = timezone(timedelta(0), "IST")  # zero offset: same instant, accepted
    assert make(S.PackManifest, created=datetime(2026, 11, 16, tzinfo=weird)).created.utcoffset() == timedelta(0)


def test_tool_result_confidence_endpoints():
    assert tool_result(raw_confidence=0.0).raw_confidence == 0.0
    assert tool_result(raw_confidence=1.0).raw_confidence == 1.0


def test_answer_allowed_combinations():
    assert make(S.Answer, status="refused", refusal_reason="cloud", confidence=None).primary is not None
    assert make(S.Answer, status="low_confidence", refusal_reason="cloud").refusal_reason == "cloud"
    assert make(S.Answer, confidence=None).confidence is None


def test_eval_record_ci95_reversed_and_equal():
    with pytest.raises(ValidationError):
        make(S.EvalRecord, value=0.6, ci95=(0.9, 0.5))
    assert make(S.EvalRecord, value=0.6, ci95=(0.6, 0.6)).ci95 == (0.6, 0.6)


@pytest.mark.parametrize("status", ["validated", "experimental_unvalidated", "disabled"])
def test_tool_status_values(status):
    assert make(S.ToolStatus, status=status).status == status


@pytest.mark.parametrize("text", ["\n\t ", "\u00a0", "\u2003"])
def test_query_whitespace_variants_rejected(text):
    with pytest.raises(ValidationError):
        make(S.Query, text=text)
