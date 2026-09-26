"""Frozen routing + validation benchmark for the deterministic SatQuery orchestrator.

Every request goes through the real FastAPI app (TestClient), the real planner,
execution-plan builder, pair-compatibility gate and readiness gate. Two regimes:

- ``stub_ready``: every provider reports ready and ``services.route`` is replaced
  by a recorder, so the benchmark observes *which capability would be dispatched*
  without running any model. No model output is produced or scored.
- ``actual``: this machine's real readiness. Unready providers must fail closed;
  ready deterministic providers really execute on the generated fixture rasters.

Labels come from eval/routing/queries.v1.jsonl, committed before this file ran.

    python -m eval.routing.evaluate --experiment-id SQ-YYYYMMDD-5XX
"""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import backend.services as services
import orchestrator.trace as trace_store
from backend.main import app
from models.base import ModelReadiness
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel
from orchestrator.capabilities import capabilities_status

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "queries.v1.jsonl"
RESULTS_DIR = HERE / "results"
MULTI_STEP_RULE = "temporal_change_then_grounding"
SAR_UNSUPPORTED_PREFIX = "Single-image SAR"
GENERIC_UNAVAILABLE = "Required capability is not currently available."
PROVIDER_MODELS = (QwenVLModel, GroundingDINOModel, OpticalSARModel, ChangeModel)


def load_queries(path: Path = QUERIES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def geotiff(
    bands: int,
    *,
    crs: str = "EPSG:32643",
    origin: tuple[float, float] = (500000.0, 2000000.0),
    res: float = 10.0,
    size: tuple[int, int] = (8, 8),
    offset: float = 0.0,
    georeferenced: bool = True,
) -> bytes:
    width, height = size
    profile = {"driver": "GTiff", "width": width, "height": height, "count": bands,
               "dtype": "float32", "nodata": 0}
    if georeferenced:
        profile.update(crs=crs, transform=from_origin(*origin, res, res))
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            for band in range(1, bands + 1):
                values = np.arange(width * height).reshape(height, width) + band + 1 + offset
                dataset.write(values.astype("float32"), band)
        return memory.read()


def png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), (20, 80, 140)).save(output, format="PNG")
    return output.getvalue()


OPTICAL = {"modality": "multispectral", "sensor": "test-optical",
           "acquisition_timestamp": "2026-01-01T00:00:00Z", "pair_group": "p1"}
SAR = {"modality": "sar", "sensor": "test-sar", "acquisition_timestamp": "2026-01-02T00:00:00Z",
       "polarization": "VV,VH", "pair_group": "p1"}
T2 = {**OPTICAL, "acquisition_timestamp": "2026-02-01T00:00:00Z"}


class Harness:
    """Isolated app state (trace, scene storage) under one readiness regime."""

    def __init__(self, regime: str):
        self.regime = regime
        self.dispatched: list[str] = []
        self.client = TestClient(app)

    @contextmanager
    def active(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            patches = {
                (trace_store, "TRACE_PATH"): root / "trace.jsonl",
                (trace_store, "_TRACE"): [],
                (trace_store, "_LOADED_PATH"): None,
                (services, "INGESTED_SCENE_DIR"): root / "scenes",
                (services, "INGESTED_RASTER_DIR"): root / "rasters",
                (services, "SCENE_MANIFEST_DIR"): root / "manifests",
            }
            if self.regime == "stub_ready":
                patches[(services, "route")] = self._record_route
                for model in PROVIDER_MODELS:
                    patches[(model, "readiness")] = lambda _: ModelReadiness(True)
            for (target, name), value in patches.items():
                stack.enter_context(mock.patch.object(target, name, value))
            yield self

    def _record_route(self, **kwargs):
        self.dispatched.append(kwargs["capability"])
        return {"answer": "stub", "trace": {
            "model_name": "stub", "model_version": "stub",
            "params": {"execution_mode": "live", "capability": kwargs["capability"]}}}

    def upload(self, name: str, data: bytes, meta: dict | None = None) -> str:
        response = self.client.post("/api/scenes", files={"file": (name, data, "application/octet-stream")},
                                    data=meta or {})
        if response.status_code != 201:
            raise RuntimeError(f"fixture upload failed: {response.status_code} {response.text}")
        return response.json()["scene_id"]

    def standard_scenes(self) -> dict[str, tuple[str, str | None] | None]:
        t1 = self.upload("t1.tif", geotiff(5), OPTICAL)
        optical = self.upload("optical.tif", geotiff(5), OPTICAL)
        return {
            "single": (self.upload("scene.png", png()), None),
            "temporal_pair": (t1, self.upload("t2.tif", geotiff(5, offset=3.0), T2)),
            "optical_sar_pair": (optical, self.upload("sar.tif", geotiff(3), SAR)),
            "missing_scene": None,
            "nonexistent_scene": ("scene_" + "0" * 32, None),
            "traversal_scene": ("../../etc/passwd", None),
            "same_scene_twice": (t1, t1),
        }


def request_body(item: dict, scenes: dict) -> dict:
    body = {"question": item["question"] * item.get("repeat", 1)}
    pair = scenes[item["scenes"]]
    if pair is not None:
        body["scene_id"] = pair[0]
        if pair[1] is not None:
            body["scene_id_2"] = pair[1]
    for key in ("capability", "sensor", "execution_mode"):
        if key in item:
            body[key] = item[key]
    return body


def reason_from_error(detail) -> str:
    """Map an API error detail to the label vocabulary in queries.v1.jsonl."""
    if isinstance(detail, list):  # pydantic request validation
        for error in detail:
            field = error.get("loc", [None])[-1]
            kind = error.get("type")
            if field == "question" and kind == "string_too_short":
                return "empty_question"
            if field == "question" and kind == "string_too_long":
                return "question_too_long"
            if field == "scene_id" and kind == "missing":
                return "missing_scene"
            if field == "execution_mode":
                return "invalid_execution_mode"
        return "schema_invalid"
    if isinstance(detail, dict):
        return "pair_incompatible" if "eligible" in detail else "provider_not_ready"
    text = str(detail)
    for needle, reason in (("Unknown capability", "unknown_capability"),
                           ("non-empty question", "empty_question"),
                           ("requires two scenes", "missing_second_scene"),
                           ("requires a scene", "missing_scene"),
                           ("scene pixels are unavailable", "scene_not_found")):
        if needle in text:
            return reason
    return "rejected_other"


def plan_label(status: int, body: dict) -> str:
    """The planner's routing decision, independent of provider readiness."""
    if status != 200:
        return f"reject:{reason_from_error(body.get('detail'))}"
    if body["missing_inputs"]:
        missing = "missing_second_scene" if "second_scene" in body["missing_inputs"] else "missing_scene"
        return f"reject:{missing}"
    if body["rule_id"] == MULTI_STEP_RULE:
        return "reject:multi_step_not_executable"
    if (body["unavailable_reason"] or "").startswith(SAR_UNSUPPORTED_PREFIX):
        return "reject:single_image_sar_unsupported"
    return body["selected_capability"]


def analyze_label(status: int, body: dict, planned: str) -> str:
    """End-to-end outcome of POST /api/analyze."""
    if status == 200:
        return f"dispatched:{body['trace']['params']['capability']}"
    detail = body.get("detail")
    if status == 503 and detail == GENERIC_UNAVAILABLE:
        # The HTTP body is generic; the specific reason is only visible in /api/plan.
        return planned if planned.startswith("reject:") else "reject:capability_unavailable"
    if status == 503 and isinstance(detail, dict) and "reason_code" in detail:
        return f"unavailable:{detail['reason_code']}"
    if status in (404, 422):
        return f"reject:{reason_from_error(detail)}"
    return f"error:{status}"


def expected_label(item: dict) -> str:
    expected = item["expected"]
    if expected["outcome"] == "route":
        return expected["capability"]
    if expected["outcome"] == "reject":
        return f"reject:{expected['reason']}"
    return "clarify"


def is_wrong_dispatch(item: dict, dispatched: str | None) -> bool:
    """A dispatch the label does not license; a non-dispatch is never 'wrong' here."""
    if dispatched is None:
        return False
    expected = item["expected"]
    if expected["outcome"] == "route":
        return dispatched != expected["capability"]
    if expected["outcome"] == "clarify":
        return dispatched not in expected["acceptable"]
    return True


def score(item: dict, planned: str, outcome: str) -> dict:
    expected = item["expected"]
    dispatched = outcome.split(":", 1)[1] if outcome.startswith("dispatched:") else None
    # Gate-less baseline: whenever planning succeeds, the selected capability runs
    # (no missing-input, multi-step, SAR-single-image, pair, or scene checks).
    baseline = item.get("_selected")
    row = {
        "id": item["id"], "category": item["category"], "variant": item["variant"],
        "expected": expected_label(item), "planned": planned, "outcome": outcome,
        "dispatched": dispatched, "baseline_dispatch": baseline,
        "wrong_dispatch": is_wrong_dispatch(item, dispatched),
        "baseline_wrong_dispatch": is_wrong_dispatch(item, baseline),
    }
    if expected["outcome"] == "route":
        row["plan_correct"] = planned == expected["capability"]
        row["correct"] = dispatched == expected["capability"]
    elif expected["outcome"] == "reject":
        row["correct"] = dispatched is None
        row["reason_correct"] = outcome == f"reject:{expected['reason']}"
    else:
        row["correct"] = dispatched is None or dispatched in expected["acceptable"]
        row["explicit_clarification"] = outcome == "reject:needs_clarification"
    return row


def rate(rows: list[dict], key: str) -> dict:
    hits = sum(bool(row[key]) for row in rows)
    return {"n": len(rows), "hits": hits, "rate": hits / len(rows) if rows else None}


def summarize(rows: list[dict]) -> dict:
    routed = [row for row in rows if not row["expected"].startswith(("reject:", "clarify"))]
    rejects = [row for row in rows if row["expected"].startswith("reject:")]
    ambiguous = [row for row in rows if row["expected"] == "clarify"]
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        confusion[row["expected"]][row["planned"]] += 1
    by = lambda rows_, key, metric: {value: rate([r for r in rows_ if r[key] == value], metric)
                                     for value in sorted({r[key] for r in rows_})}
    return {
        "routing_plan_accuracy": rate(routed, "plan_correct"),
        "routing_plan_accuracy_by_category": by(routed, "category", "plan_correct"),
        "routing_plan_accuracy_by_variant": by(routed, "variant", "plan_correct"),
        "routing_dispatch_accuracy": rate(routed, "correct"),
        "rejection_rate": rate(rejects, "correct"),
        "rejection_rate_by_category": by(rejects, "category", "correct"),
        "rejection_reason_accuracy": rate(rejects, "reason_correct"),
        "unsupported_rejection_rate": rate([r for r in rejects if r["category"] == "unsupported"], "correct"),
        "ambiguous_acceptable_handling": rate(ambiguous, "correct"),
        "ambiguous_explicit_clarification": rate(ambiguous, "explicit_clarification"),
        "abstention": {
            "items": len(rows),
            "system_dispatches": sum(r["dispatched"] is not None for r in rows),
            "system_wrong_dispatches": sum(r["wrong_dispatch"] for r in rows),
            "baseline_dispatches": sum(r["baseline_dispatch"] is not None for r in rows),
            "baseline_wrong_dispatches": sum(r["baseline_wrong_dispatch"] for r in rows),
            "system_false_refusals": sum(r["dispatched"] is None for r in routed),
        },
        "confusion_matrix": {expected: dict(counts) for expected, counts in sorted(confusion.items())},
    }


def run_queries(harness: Harness, queries: list[dict]) -> list[dict]:
    scenes = harness.standard_scenes()
    rows = []
    for item in queries:
        body = request_body(item, scenes)
        plan = harness.client.post("/api/plan", json=body)
        plan_body = plan.json()
        planned = plan_label(plan.status_code, plan_body)
        item = {**item, "_selected": plan_body.get("selected_capability") if plan.status_code == 200 else None}
        records_before = len(trace_store.records())
        response = harness.client.post("/api/analyze", json=body)
        trace_after = trace_store.records()
        row = score(item, planned, analyze_label(response.status_code, response.json(), planned))
        row["http_status"] = response.status_code
        new = trace_after[records_before:]
        row["trace_records_added"] = [record["model_name"] for record in new]
        rows.append(row)
    return rows


PAIR_CASES = [
    # (name, workflow, expected reason code or None for a positive control, scene 1, scene 2)
    ("change_valid", "change_vqa", None, {}, {"meta": T2, "offset": 3.0}),
    ("change_modality", "change_vqa", "modalities_incompatible", {}, {"bands": 3, "meta": {**SAR, "acquisition_timestamp": "2026-02-01T00:00:00Z"}}),
    ("change_time_missing", "change_vqa", "acquisition_time_missing", {}, {"meta": {**T2, "acquisition_timestamp": ""}}),
    ("change_time_reversed", "change_vqa", "acquisition_order_invalid", {"meta": T2}, {}),
    ("change_time_equal", "change_vqa", "acquisition_order_invalid", {}, {}),
    ("change_pair_group", "change_vqa", "pair_group_mismatch", {}, {"meta": {**T2, "pair_group": "p2"}}),
    ("change_crs_missing", "change_vqa", "crs_missing", {}, {"meta": T2, "georeferenced": False}),
    ("change_no_overlap", "change_vqa", "overlap_below_threshold", {}, {"meta": T2, "origin": (600000.0, 2000000.0)}),
    ("change_resolution", "change_vqa", "resolution_ratio_exceeded", {}, {"meta": T2, "res": 20.0}),
    ("change_dimensions", "change_vqa", "dimensions_incompatible", {}, {"meta": T2, "size": (10, 8)}),
    ("change_crs_differs", "change_vqa", "reprojection_required", {}, {"meta": T2, "crs": "EPSG:32644"}),
    ("change_grid_shift", "change_vqa", "grid_alignment_incompatible", {}, {"meta": T2, "origin": (500005.0, 2000000.0)}),
    ("osar_valid", "optical_sar", None, {}, {"bands": 3, "meta": SAR}),
    ("osar_modality", "optical_sar", "modalities_incompatible", {}, {"meta": T2}),
    ("osar_pol_missing", "optical_sar", "polarization_missing", {}, {"bands": 3, "meta": {**SAR, "polarization": ""}}),
    ("osar_pol_unsupported", "optical_sar", "polarization_unsupported", {}, {"bands": 3, "meta": {**SAR, "polarization": "XX"}}),
    ("osar_sensor_missing", "optical_sar", "acquisition_metadata_missing", {}, {"bands": 3, "meta": {**SAR, "sensor": ""}}),
    ("osar_time_missing", "optical_sar", "acquisition_metadata_missing", {}, {"bands": 3, "meta": {**SAR, "acquisition_timestamp": ""}}),
    ("osar_pair_group", "optical_sar", "pair_group_mismatch", {}, {"bands": 3, "meta": {**SAR, "pair_group": "p2"}}),
    ("osar_crs_missing", "optical_sar", "crs_missing", {}, {"bands": 3, "meta": SAR, "georeferenced": False}),
    ("osar_no_overlap", "optical_sar", "overlap_below_threshold", {}, {"bands": 3, "meta": SAR, "origin": (600000.0, 2000000.0)}),
    ("osar_resolution", "optical_sar", "resolution_ratio_exceeded", {}, {"bands": 3, "meta": SAR, "res": 50.0}),
    ("osar_dimensions", "optical_sar", "dimensions_incompatible", {}, {"bands": 3, "meta": SAR, "size": (10, 8)}),
    ("osar_crs_differs", "optical_sar", "reprojection_required", {}, {"bands": 3, "meta": SAR, "crs": "EPSG:32644"}),
    ("osar_grid_shift", "optical_sar", "grid_alignment_differs", {}, {"bands": 3, "meta": SAR, "origin": (500005.0, 2000000.0)}),
]


def _pair_scene(harness: Harness, name: str, spec: dict) -> str:
    geometry = {key: spec[key] for key in ("crs", "origin", "res", "size", "offset", "georeferenced") if key in spec}
    return harness.upload(f"{name}.tif", geotiff(spec.get("bands", 5), **geometry), spec.get("meta", OPTICAL))


def run_pair_suite(harness: Harness) -> list[dict]:
    rows = []
    for name, workflow, expected, first, second in PAIR_CASES:
        body = {"scene_id": _pair_scene(harness, f"{name}_1", first),
                "scene_id_2": _pair_scene(harness, f"{name}_2", second),
                "question": "Compare these scenes.", "capability": workflow}
        response = harness.client.post("/api/analyze", json=body)
        detail = response.json().get("detail")
        codes = detail.get("reason_codes", []) if isinstance(detail, dict) else []
        dispatched = response.status_code == 200
        rows.append({
            "case": name, "workflow": workflow, "expected_code": expected,
            "http_status": response.status_code, "reason_codes": codes, "dispatched": dispatched,
            "passed": dispatched if expected is None else (not dispatched and expected in codes),
        })
    return rows


def summarize_pairs(rows: list[dict]) -> dict:
    negatives = [row for row in rows if row["expected_code"] is not None]
    positives = [row for row in rows if row["expected_code"] is None]
    return {
        "invalid_pair_rejection": rate([{**r, "rejected": not r["dispatched"]} for r in negatives], "rejected"),
        "invalid_pair_expected_code_reported": rate(negatives, "passed"),
        "valid_pair_acceptance": rate(positives, "passed"),
    }


def summarize_unavailable(rows: list[dict], status: list[dict]) -> dict:
    unready = {entry["name"] for entry in status if not entry["available"]}
    targeted = [row for row in rows if row["planned"] in unready and row["http_status"] != 404]
    return {
        "unready_capabilities": sorted(unready),
        "requests_routed_to_unready_provider": len(targeted),
        "structured_503": sum(row["outcome"].startswith("unavailable:") for row in targeted),
        "answered_200": sum(row["http_status"] == 200 for row in targeted),
        "not_executed_trace_records": sum(row["trace_records_added"] == ["not-executed"] for row in targeted),
        "sar_single_image_reported_as_provider_unavailable": sum(
            row["expected"] == "reject:single_image_sar_unsupported" and row["outcome"].startswith("unavailable:")
            for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    args = parser.parse_args()
    output = RESULTS_DIR / f"{args.experiment_id}.json"
    started = datetime.now(timezone.utc).isoformat()
    queries = load_queries(args.queries)
    with Harness("stub_ready").active() as harness:
        stub_rows = run_queries(harness, queries)
        pair_rows = run_pair_suite(harness)
    with Harness("actual").active() as harness:
        status = capabilities_status()
        actual_rows = run_queries(harness, queries)
    result = {
        "experiment_id": args.experiment_id,
        "queries": str(args.queries.relative_to(HERE.parents[1])),
        "n_queries": len(queries),
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "stub_ready": {"summary": summarize(stub_rows), "rows": stub_rows},
        "pair_suite": {"summary": summarize_pairs(pair_rows), "rows": pair_rows},
        "actual": {"capabilities_status": status, "summary": summarize(actual_rows),
                   "provider_unavailable": summarize_unavailable(actual_rows, status), "rows": actual_rows},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:  # results are never overwritten
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: result[key]["summary"] for key in ("stub_ready", "pair_suite")}
                     | {"provider_unavailable": result["actual"]["provider_unavailable"]}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
