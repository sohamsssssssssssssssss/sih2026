"""Evaluate the deterministic change baseline on the real Sentinel-2 pair pack.

Research evaluation, not a capability claim. For each acquired pair it:
1. uploads T1/T2 through the canonical API (ingestion -> pair gate -> provider
   -> trace), exactly as a user would;
2. recomputes the provider's per-pixel rule for diagnostics and asserts the
   pixel counts match the provider's own evidence;
3. conditions detections on the Sentinel-2 Scene Classification (SCL) layer,
   an optional analyst reference box, and an estimated residual T1/T2 shift;
4. re-runs the provider on an SCL-cloud/shadow-masked variant of the inputs.

There are no pixel labels. "Reference" means dated public event records plus
visual inspection of RGB renders, declared in data/manifests/change/pairs.v1.json.
"""

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.change.model import CHANGE_THRESHOLD, ChangeModel  # noqa: E402

SPEC_PATH = ROOT / "data" / "manifests" / "change" / "pairs.v1.json"
PROVENANCE_DIR = ROOT / "data" / "manifests" / "change"
DATA_ROOT = Path.home() / "satquery-data" / "change"
RESULTS_PATH = ROOT / "eval" / "results" / "change-real-pairs-v1.json"
SCL_CLEAR = (4, 5, 6, 11)  # vegetation, not-vegetated, water, snow/ice
SCL_CLOUD = (8, 9, 10)  # cloud medium/high probability, thin cirrus
SCL_SHADOW = (2, 3)  # dark area / topographic shadow, cloud shadow
QUESTION = "Measure visual change between these observations."


def load_pair(pair_dir: Path) -> dict:
    arrays = {}
    for label in ("t1", "t2"):
        with rasterio.open(pair_dir / f"{label}.tif") as dataset:
            arrays[label] = dataset.read()
        with rasterio.open(pair_dir / f"{label}_scl.tif") as dataset:
            arrays[f"{label}_scl"] = dataset.read(1)
    return arrays


def provider_rule(first: np.ndarray, second: np.ndarray) -> dict:
    """Per-pixel replica of change-deterministic's multispectral branch (diagnostics only)."""
    valid = (first[4] > 0) & (second[4] > 0)
    valid &= np.all(np.isfinite(first[1:4]), axis=0) & np.all(np.isfinite(second[1:4]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        indices = {}
        for label, image in (("t1", first), ("t2", second)):
            green, red, nir = image[1].astype(np.float64), image[2].astype(np.float64), image[3].astype(np.float64)
            valid &= (np.abs(nir + red) > 1e-12) & (np.abs(green + nir) > 1e-12)
            indices[label] = ((nir - red) / (nir + red), (green - nir) / (green + nir))
        delta_ndvi = np.where(valid, indices["t2"][0] - indices["t1"][0], np.nan)
        delta_ndwi = np.where(valid, indices["t2"][1] - indices["t1"][1], np.nan)
    magnitude = np.clip((np.abs(delta_ndvi) + np.abs(delta_ndwi)) / 4.0, 0.0, 1.0)
    changed = valid & (np.nan_to_num(magnitude) > CHANGE_THRESHOLD)
    return {"valid": valid, "changed": changed, "delta_ndvi": delta_ndvi,
            "delta_ndwi": delta_ndwi, "magnitude": magnitude}


def fraction(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.count_nonzero(denominator))
    return int(np.count_nonzero(numerator & denominator)) / count if count else None


def scl_conditioning(rule: dict, scl1: np.ndarray, scl2: np.ndarray) -> dict:
    clear = np.isin(scl1, SCL_CLEAR) & np.isin(scl2, SCL_CLEAR) & rule["valid"]
    cloud = (np.isin(scl1, SCL_CLOUD) | np.isin(scl2, SCL_CLOUD)) & rule["valid"]
    shadow = (np.isin(scl1, SCL_SHADOW) | np.isin(scl2, SCL_SHADOW)) & rule["valid"] & ~cloud
    changed, valid = rule["changed"], rule["valid"]
    total_changed = int(np.count_nonzero(changed))
    # ESA's SCL water class flipping between dates: an independent-algorithm (but still
    # spectral, so not fully independent) reference for water-extent change.
    water_transition = ((scl1 == 6) ^ (scl2 == 6)) & valid
    return {
        "scl_water_transition_fraction_of_valid": fraction(water_transition, valid),
        "changed_fraction_on_scl_water_transition": fraction(changed, water_transition),
        "clear_both_fraction_of_valid": fraction(clear, valid),
        "cloud_any_fraction_of_valid": fraction(cloud, valid),
        "shadow_any_fraction_of_valid": fraction(shadow, valid),
        "changed_fraction_on_clear_both": fraction(changed, clear),
        "changed_fraction_on_cloud_any": fraction(changed, cloud),
        "changed_fraction_on_shadow_any": fraction(changed, shadow),
        "share_of_changed_in_cloud_or_shadow": (
            int(np.count_nonzero(changed & (cloud | shadow))) / total_changed if total_changed else None
        ),
    }


def reference_box_stats(rule: dict, boxes: list[list[float]] | None) -> dict | None:
    """Detections inside vs outside the union of analyst-drawn normalized xyxy boxes."""
    if not boxes:
        return None
    height, width = rule["valid"].shape
    inside = np.zeros_like(rule["valid"])
    for box in boxes:
        inside[int(box[1] * height):int(np.ceil(box[3] * height)), int(box[0] * width):int(np.ceil(box[2] * width))] = True
    changed, valid = rule["changed"], rule["valid"]
    inside_rate = fraction(changed, inside & valid)
    outside_rate = fraction(changed, ~inside & valid)
    total_changed = int(np.count_nonzero(changed))
    return {
        "boxes_normalized_xyxy": boxes,
        "box_fraction_of_valid": fraction(inside, valid),
        "changed_fraction_inside": inside_rate,
        "changed_fraction_outside": outside_rate,
        "inside_outside_rate_ratio": inside_rate / outside_rate if inside_rate is not None and outside_rate else None,
        "share_of_changed_inside": int(np.count_nonzero(changed & inside)) / total_changed if total_changed else None,
    }


def _parabolic_peak(profile: np.ndarray, index: int) -> float:
    left, centre, right = profile[index - 1], profile[index], profile[(index + 1) % profile.size]
    denominator = left - 2 * centre + right
    value = index + (0.5 * (left - right) / denominator if denominator else 0.0)
    return float(value - profile.size if value > profile.size / 2 else value)


def phase_correlation_shift(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> dict | None:
    """Sub-pixel dx/dy such that T2(x) ~= T1(x - d), from NIR phase correlation.

    `peak` is the normalized correlation-surface maximum (1 = identical up to a
    shift); a low peak means the scenes differ too much for the shift to mean
    anything. ponytail: single global translation; a registration diagnostic
    for controls, not a correction.
    """
    if np.count_nonzero(valid) < 0.5 * valid.size:
        return None
    window = np.outer(np.hanning(valid.shape[0]), np.hanning(valid.shape[1]))
    a = np.where(valid, first, first[valid].mean()) * window
    b = np.where(valid, second, second[valid].mean()) * window
    spectrum = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    surface = np.abs(np.fft.ifft2(spectrum / (np.abs(spectrum) + 1e-12)))
    row, column = np.unravel_index(np.argmax(surface), surface.shape)
    return {
        "dx": _parabolic_peak(surface[row, :], column),
        "dy": _parabolic_peak(surface[:, column], row),
        "peak": float(surface[row, column]),
    }


def masked_variant(pair_dir: Path, arrays: dict) -> dict:
    """Re-run the provider with dataMask additionally excluding SCL cloud/shadow/other in either date."""
    keep = np.isin(arrays["t1_scl"], SCL_CLEAR) & np.isin(arrays["t2_scl"], SCL_CLEAR)
    paths = []
    for label in ("t1", "t2"):
        source = pair_dir / f"{label}.tif"
        target = pair_dir / f"{label}_sclmasked.tif"
        with rasterio.open(source) as dataset:
            profile, data = dataset.profile, dataset.read()
            descriptions = dataset.descriptions
        data[4] = np.where(keep, data[4], 0)
        with rasterio.open(target, "w", **profile) as dataset:
            dataset.write(data)
            for index, name in enumerate(descriptions, 1):
                dataset.set_band_description(index, name)
        paths.append(str(target))
    try:
        result = ChangeModel().infer(paths, QUESTION)
    except ValueError as exc:  # the provider refuses pairs with no co-valid pixels
        return {"status": "provider_refused", "detail": str(exc)}
    statistics = result["evidence"][1]
    return {
        "status": "ran",
        "rule": "dataMask &= SCL in {4,5,6,11} at both dates",
        "changed_fraction": statistics["changed_fraction"],
        "changed_pixels": statistics["changed_pixels"],
        "valid_pixels": result["evidence"][2]["valid_pixels"],
    }


@contextmanager
def isolated_runtime(runtime: Path):
    """Point the canonical API's scene store and trace at a scratch runtime outside git."""
    from backend import services
    from orchestrator import trace as trace_store

    saved = {name: getattr(services, name) for name in ("INGESTED_SCENE_DIR", "INGESTED_RASTER_DIR", "SCENE_MANIFEST_DIR")}
    saved_trace = trace_store.TRACE_PATH
    services.INGESTED_SCENE_DIR = runtime / "scenes"
    services.INGESTED_RASTER_DIR = runtime / "rasters"
    services.SCENE_MANIFEST_DIR = runtime / "manifests"
    trace_store.TRACE_PATH = runtime / "trace.jsonl"
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    try:
        yield trace_store
    finally:
        for name, value in saved.items():
            setattr(services, name, value)
        trace_store.TRACE_PATH = saved_trace
        trace_store._TRACE.clear()
        trace_store._LOADED_PATH = None


def api_run(client, pair_id: str, observations: list[dict], paths: list[Path], trace_store) -> dict:
    """Upload in the given order (first = claimed T1) and request change_vqa live."""
    scene_ids = []
    for observation, path in zip(observations, paths):
        response = client.post(
            "/api/scenes",
            files={"file": (path.name, path.read_bytes(), "image/tiff")},
            data={
                "modality": "multispectral",
                "sensor": "Sentinel-2-L2A",
                "acquisition_timestamp": observation["acquisition_time"],
                "pair_group": pair_id,
                "benchmark_source": f"earth-search:{observation['collection']}:{observation['stac_item_id']}",
            },
        )
        if response.status_code != 201:
            return {"stage": "upload", "status_code": response.status_code, "detail": response.json()}
        scene_ids.append(response.json()["scene_id"])
    response = client.post(
        "/api/analyze",
        json={"scene_id": scene_ids[0], "scene_id_2": scene_ids[1], "question": QUESTION,
              "capability": "change_vqa", "execution_mode": "live"},
    )
    body = response.json()
    outcome = {"stage": "analyze", "status_code": response.status_code}
    if response.status_code != 200:
        outcome["gate"] = body.get("detail")
        return outcome
    outcome.update(
        model=body["model"],
        execution_mode=body.get("execution_mode"),
        confidence_present="confidence" in body,
        answer=body["answer"],
        evidence=body["evidence"],
        trace_chain_valid=trace_store.verify_chain()[0],
    )
    return outcome


def _day_of_year(value: str) -> int:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timetuple().tm_yday


def acquisition_context(observations: list[dict]) -> dict:
    times = [datetime.fromisoformat(o["acquisition_time"].replace("Z", "+00:00")) for o in observations]
    return {
        "interval_days": round((times[1] - times[0]).total_seconds() / 86400, 2),
        "day_of_year": [_day_of_year(o["acquisition_time"]) for o in observations],
        "platforms": [o["platform"] for o in observations],
        "processing_baselines": [o["processing_baseline"] for o in observations],
        "sun_elevation_deg": [o["sun_elevation_deg"] for o in observations],
        "tile_cloud_cover_percent": [o["tile_cloud_cover_percent"] for o in observations],
        "aoi_nodata_fraction": [o["aoi_nodata_fraction"] for o in observations],
    }


def evaluate_pair(client, spec: dict, provenance: dict, pair_dir: Path, trace_store, render_dir: Path | None) -> dict:
    observations = provenance["observations"]
    paths = [pair_dir / "t1.tif", pair_dir / "t2.tif"]
    result = {
        "pair_id": spec["pair_id"],
        "category": spec["category"],
        "expected_outcome": spec["expected_outcome"],
        "expected_reason_code": spec.get("expected_reason_code"),
        "input_sha256": [o["raster"]["sha256"] for o in observations],
        "acquisition": acquisition_context(observations),
        "api": api_run(client, spec["pair_id"], observations, paths, trace_store),
    }
    if result["api"].get("status_code") != 200:
        return result
    arrays = load_pair(pair_dir)
    rule = provider_rule(arrays["t1"], arrays["t2"])
    statistics = result["api"]["evidence"][1]
    coverage = result["api"]["evidence"][2]
    replica = {"changed_pixels": int(rule["changed"].sum()), "valid_pixels": int(rule["valid"].sum())}
    if replica != {"changed_pixels": statistics["changed_pixels"], "valid_pixels": coverage["valid_pixels"]}:
        raise AssertionError(f"{spec['pair_id']}: diagnostic replica {replica} disagrees with provider evidence")
    result["diagnostics"] = {
        "replica_matches_provider": True,
        "scl": scl_conditioning(rule, arrays["t1_scl"], arrays["t2_scl"]),
        "reference_box": reference_box_stats(rule, spec.get("reference_boxes_norm")),
        "nir_shift_px_t2_vs_t1": phase_correlation_shift(arrays["t1"][3], arrays["t2"][3], rule["valid"]),
        "negative_reflectance_fraction": float(np.mean(np.any(arrays["t1"][:4] < 0, axis=0) | np.any(arrays["t2"][:4] < 0, axis=0))),
    }
    result["scl_masked_variant"] = masked_variant(pair_dir, arrays)
    if render_dir is not None:
        result["render"] = str(render_panel(spec, arrays, rule, render_dir))
    return result


def rgb(image: np.ndarray, low: float, high: float) -> np.ndarray:
    stack = np.stack([image[2], image[1], image[0]], axis=-1)
    return np.clip((stack - low) / (high - low), 0, 1) ** (1 / 1.8)


def _shared_stretch(arrays: dict) -> tuple[float, float]:
    values = np.concatenate([arrays[label][:3][:, arrays[label][4] > 0].ravel() for label in ("t1", "t2")])
    low, high = np.percentile(values, (2, 98))
    return float(low), float(high)


def render_rgb_preview(spec: dict, arrays: dict, render_dir: Path) -> Path:
    """T1/T2 true colour only; used to draw reference boxes before any change output is seen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = _shared_stretch(arrays)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for axis, label in zip(axes, ("t1", "t2")):
        axis.imshow(rgb(arrays[label], low, high), extent=(0, 1, 1, 0))
        axis.set_title(f"{label.upper()} {spec[f'{label}_item']}", fontsize=8)
        axis.set_xticks(np.linspace(0, 1, 11))
        axis.set_yticks(np.linspace(0, 1, 11))
        axis.tick_params(labelsize=6)
        axis.grid(color="white", alpha=0.35, linewidth=0.5)
    figure.suptitle(f"{spec['pair_id']} (normalized grid for reference boxes)", fontsize=10)
    render_dir.mkdir(parents=True, exist_ok=True)
    path = render_dir / f"{spec['pair_id']}__rgb.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def render_panel(spec: dict, arrays: dict, rule: dict, render_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = _shared_stretch(arrays)
    cloud_shadow = ~(np.isin(arrays["t1_scl"], SCL_CLEAR) & np.isin(arrays["t2_scl"], SCL_CLEAR))
    panels = [
        ("T1 RGB", rgb(arrays["t1"], low, high), {}),
        ("T2 RGB", rgb(arrays["t2"], low, high), {}),
        ("dNDVI (T2-T1)", rule["delta_ndvi"], {"cmap": "RdYlGn", "vmin": -0.6, "vmax": 0.6}),
        ("dNDWI (T2-T1)", rule["delta_ndwi"], {"cmap": "BrBG", "vmin": -0.6, "vmax": 0.6}),
        (f"changed (magnitude > {CHANGE_THRESHOLD})", rule["changed"], {"cmap": "Reds", "vmin": 0, "vmax": 1}),
        ("SCL not clear in T1 or T2", cloud_shadow, {"cmap": "Greys", "vmin": 0, "vmax": 1}),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, (title, image, style) in zip(axes.ravel(), panels):
        shown = axis.imshow(image, extent=(0, 1, 1, 0), interpolation="nearest", **style)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        if style.get("cmap") in {"RdYlGn", "BrBG"}:
            figure.colorbar(shown, ax=axis, fraction=0.046)
        for box in spec.get("reference_boxes_norm") or []:
            axis.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         fill=False, edgecolor="cyan", linewidth=1.2))
    figure.suptitle(f"{spec['pair_id']}: {spec['t1_item']} -> {spec['t2_item']}", fontsize=10)
    render_dir.mkdir(parents=True, exist_ok=True)
    path = render_dir / f"{spec['pair_id']}.png"
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    return path


def reversed_order_check(client, spec: dict, provenance: dict, pair_dir: Path, trace_store) -> dict:
    """Upload the same real pair with T2 declared first: the gate must refuse it."""
    observations = list(reversed(provenance["observations"]))
    paths = [pair_dir / "t2.tif", pair_dir / "t1.tif"]
    return {
        "pair_id": f"{spec['pair_id']}::reversed-order",
        "category": "rejection_temporal_order",
        "expected_outcome": "gate_rejection",
        "expected_reason_code": "acquisition_order_invalid",
        "api": api_run(client, f"{spec['pair_id']}-reversed", observations, paths, trace_store),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--rgb-only", action="store_true",
                        help="only write T1/T2 true-colour previews (run before drawing reference boxes)")
    parser.add_argument("--register", type=int, metavar="FIRST_ID",
                        help="also write immutable registry records SQ-<today>-<FIRST_ID...>")
    args = parser.parse_args()

    specs = json.loads(SPEC_PATH.read_text())["pairs"]
    if args.rgb_only:
        for spec in specs:
            pair_dir = args.data_root / spec["pair_id"]
            if (pair_dir / "t2.tif").is_file() and not spec.get("invalidated"):
                print(render_rgb_preview(spec, load_pair(pair_dir), args.data_root / "renders"))
        return 0

    from fastapi.testclient import TestClient

    from backend.main import app

    started = datetime.now(timezone.utc).isoformat()
    results = []
    with isolated_runtime(args.data_root / "runtime") as trace_store:
        client = TestClient(app)
        for spec in specs:
            provenance_path = PROVENANCE_DIR / f"{spec['pair_id']}.provenance.json"
            if spec.get("invalidated"):
                results.append({"pair_id": spec["pair_id"], "category": spec["category"],
                                "expected_outcome": spec["expected_outcome"], "status": "invalidated",
                                "invalidated": spec["invalidated"]})
                continue
            if not provenance_path.is_file():
                results.append({"pair_id": spec["pair_id"], "category": spec["category"],
                                "expected_outcome": spec["expected_outcome"], "status": "not_acquired"})
                continue
            provenance = json.loads(provenance_path.read_text())
            pair_dir = args.data_root / spec["pair_id"]
            render_dir = None if args.no_render else args.data_root / "renders"
            results.append(evaluate_pair(client, spec, provenance, pair_dir, trace_store, render_dir))
            if spec["pair_id"] == "urban-jewar-airport-2021-2025":
                results.append(reversed_order_check(client, spec, provenance, pair_dir, trace_store))
    report = {
        "schema": "satquery.change_real_pairs_eval.v1",
        "provider": {"name": ChangeModel.name, "version": ChangeModel.version, "threshold": CHANGE_THRESHOLD},
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "spec": str(SPEC_PATH.relative_to(ROOT)),
        "scl_groups": {"clear": SCL_CLEAR, "cloud": SCL_CLOUD, "shadow": SCL_SHADOW},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for result in results:
        result["registry_status"], result["registry_reason"] = classify(result)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(f"wrote {args.output}")
    if args.register is not None:
        report = json.loads(args.output.read_text())
        command = " ".join([Path(sys.executable).name, *sys.argv])
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        print("registered", register(report, args.register, date, command))
    return 0


MIN_VALID_FRACTION = 0.5
MIN_CLEAR_FRACTION = 0.9
MIN_CLOUD_FRACTION_FOR_CLOUD_CONTROL = 0.05


def classify(result: dict) -> tuple[str, str]:
    """Registry status = whether the case produced valid evidence, NOT whether the baseline was right."""
    api = result.get("api") or {}
    if result.get("status") == "not_acquired":
        return "FAILED", "pair was not acquired"
    if result.get("status") == "invalidated":
        return "FAILED", f"invalidated: {result['invalidated']['reason']}"
    if result["expected_outcome"] == "gate_rejection":
        codes = (api.get("gate") or {}).get("reason_codes", [])
        if api.get("status_code") == 422 and result["expected_reason_code"] in codes:
            return "PASSED", f"gate rejected as expected with {codes}"
        return "FAILED", f"expected rejection {result['expected_reason_code']}, got {api.get('status_code')} {codes}"
    if api.get("status_code") != 200:
        return "FAILED", f"unexpected API outcome {api.get('status_code')}: {api.get('gate') or api.get('detail')}"
    valid_fraction = api["evidence"][2]["valid_fraction"]
    scl = result["diagnostics"]["scl"]
    if result["category"] == "control_cloud_contamination":
        if (scl["cloud_any_fraction_of_valid"] or 0) < MIN_CLOUD_FRACTION_FOR_CLOUD_CONTROL:
            return "INCONCLUSIVE", "cloud control AOI is essentially cloud-free"
        return "PASSED", "cloud present in AOI as intended"
    if valid_fraction < MIN_VALID_FRACTION:
        return "INCONCLUSIVE", f"co-valid fraction {valid_fraction:.3f} < {MIN_VALID_FRACTION}"
    if (scl["clear_both_fraction_of_valid"] or 0) < MIN_CLEAR_FRACTION:
        return "INCONCLUSIVE", f"SCL clear-both fraction {scl['clear_both_fraction_of_valid']:.3f} < {MIN_CLEAR_FRACTION}"
    return "PASSED", "valid, predominantly clear co-registered pair measured"


def _metrics(result: dict) -> dict:
    api = result.get("api") or {}
    metrics = {"api_status_code": api.get("status_code"),
               "gate_reason_codes": (api.get("gate") or {}).get("reason_codes")}
    if api.get("status_code") == 200:
        statistics, coverage = api["evidence"][1], api["evidence"][2]
        metrics.update(
            changed_fraction=statistics["changed_fraction"],
            valid_fraction=coverage["valid_fraction"],
            magnitude_p50=statistics["change_magnitude"]["p50"],
            magnitude_p95=statistics["change_magnitude"]["p95"],
            delta_ndvi_mean=statistics["delta_ndvi"]["mean"],
            delta_ndwi_mean=statistics["delta_ndwi"]["mean"],
            scl=result["diagnostics"]["scl"],
            reference_box=result["diagnostics"]["reference_box"],
            nir_shift_px_t2_vs_t1=result["diagnostics"]["nir_shift_px_t2_vs_t1"],
            scl_masked_changed_fraction=result["scl_masked_variant"].get("changed_fraction"),
        )
    return metrics


def register(report: dict, first_id: int, date: str, command: str) -> list[str]:
    """Write one immutable registry record per case, IDs assigned in spec order."""
    from eval.registry import write_record

    written = []
    for offset, result in enumerate(report["results"]):
        status, reason = classify(result)
        experiment_id = f"SQ-{date}-{first_id + offset:03d}"
        write_record({
            "experiment_id": experiment_id,
            "title": f"change-deterministic on real S2 pair: {result['pair_id']}",
            "dataset": "SatQuery real Sentinel-2 change pair pack (Earth Search sentinel-2-c1-l2a windows)",
            "dataset_version": report["spec"],
            "dataset_checksum": {"t1_sha256": (result.get("input_sha256") or [None, None])[0],
                                 "t2_sha256": (result.get("input_sha256") or [None, None])[1]},
            "split": "evaluation-only; no training or tuning",
            "model_id": report["provider"]["name"],
            "model_revision": report["provider"]["version"],
            "adapter_revision": None,
            "seed": None,
            "hyperparameters": {"threshold": report["provider"]["threshold"]},
            "command": command,
            "started_at": report["started_at"],
            "ended_at": report["ended_at"],
            "metrics": {"category": result.get("category"), "expected_outcome": result.get("expected_outcome"),
                        **_metrics(result)},
            "artifact_paths": {"results": str(RESULTS_PATH.relative_to(ROOT)),
                               "render": result.get("render")},
            "status": status,
            "notes": reason,
        })
        written.append(experiment_id)
    return written


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
