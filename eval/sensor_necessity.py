"""Optical-only vs SAR-only vs fusion water/land evaluation on one co-gridded case.

Research evaluation, not a production capability. The question is narrow:
for pixels with an independent, temporally stable reference (JRC Global
Surface Water, Landsat-derived, 1984-2020), does SAR recover open-water /
land separation where Sentinel-2 is optically obstructed?

Predictions reuse the frozen CDSE rule thresholds from
data/manifests/cdse/sensor-necessity-results.v1.json without retuning.
Reference and metric definitions are in docs/research/optical-sar-sensor-necessity.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RULE_PATH = ROOT / "data" / "manifests" / "cdse" / "sensor-necessity-results.v1.json"
# Sen2Cor SCL classes: 3 cloud shadow, 8 cloud medium prob., 9 cloud high prob., 10 thin cirrus.
OBSTRUCTED_SCL = (3, 8, 9, 10)
WATER_MIN_OCCURRENCE = 90  # GSW percent; >= this is "permanent water" reference
EROSION_PIXELS = 3  # 30 m at 10 m spacing: one GSW pixel away from any class boundary
MIN_STRATUM_PIXELS = 500  # per reference class; smaller strata are reported as not evaluable
METHODS = ("optical_only", "sar_only", "fusion_and_frozen", "fusion_cloud_gated")


def frozen_rule(path: Path = FROZEN_RULE_PATH) -> dict:
    return json.loads(path.read_text())["locked_rule"]


def reference_masks(occurrence: np.ndarray, erosion: int = EROSION_PIXELS) -> tuple[np.ndarray, np.ndarray]:
    """Eroded permanent-water and never-water masks; transitional pixels belong to neither."""
    valid = occurrence <= 100
    water = valid & (occurrence >= WATER_MIN_OCCURRENCE)
    land = valid & (occurrence == 0)
    if erosion:
        structure = np.ones((3, 3), dtype=bool)
        water = binary_erosion(water, structure, iterations=erosion)
        land = binary_erosion(land, structure, iterations=erosion)
    return water, land


def predictions(optical: np.ndarray, sar: np.ndarray, scl: np.ndarray, rule: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Water predictions per method, the joint-valid mask, and the obstruction mask.

    optical: (5, H, W) B02, B03, B04, B08 reflectance + dataMask.
    sar: (3, H, W) linear VV, VH + dataMask.
    """
    _, green, _, nir, optical_mask = optical
    vv, vh, sar_mask = sar
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (green - nir) / (green + nir)
    valid = (optical_mask > 0) & (sar_mask > 0) & np.isfinite(ndwi) & np.isfinite(vv) & np.isfinite(vh)
    obstructed = np.isin(scl, OBSTRUCTED_SCL)
    optical_water = ndwi > rule["ndwi_strictly_greater_than"]
    sar_water = (vv <= rule["vv_linear_gamma0_terrain_max"]) & (vh <= rule["vh_linear_gamma0_terrain_max"])
    methods = {
        "optical_only": optical_water,
        "sar_only": sar_water,
        "fusion_and_frozen": optical_water & sar_water,
        "fusion_cloud_gated": np.where(obstructed, sar_water, optical_water),
    }
    return methods, valid, obstructed


def score(predicted: np.ndarray, water: np.ndarray, land: np.ndarray, stratum: np.ndarray) -> dict:
    tp = int(np.count_nonzero(predicted & water & stratum))
    fn = int(np.count_nonzero(~predicted & water & stratum))
    fp = int(np.count_nonzero(predicted & land & stratum))
    tn = int(np.count_nonzero(~predicted & land & stratum))
    evaluable = tp + fn >= MIN_STRATUM_PIXELS and tn + fp >= MIN_STRATUM_PIXELS
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "reference_water_pixels": tp + fn,
        "reference_land_pixels": tn + fp,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "water_recall": recall,
        "land_specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if evaluable else None,
        "water_iou": tp / (tp + fn + fp) if evaluable and tp + fn + fp else None,
        "evaluable": evaluable,
    }


def evaluate_arrays(optical, sar, scl, occurrence, rule: dict, exclude_columns: tuple[int, int] | None = None) -> dict:
    """`exclude_columns` drops a [start, end) column band from every stratum (post-hoc sensitivity only)."""
    water, land = reference_masks(occurrence)
    methods, valid, obstructed = predictions(optical, sar, scl, rule)
    if exclude_columns:
        valid = valid.copy()
        valid[:, exclude_columns[0]:exclude_columns[1]] = False
    strata = {"all": valid, "optically_clear": valid & ~obstructed, "optically_obstructed": valid & obstructed}
    return {
        "obstructed_fraction_of_valid": float(np.count_nonzero(obstructed & valid) / max(1, np.count_nonzero(valid))),
        "strata": {
            stratum: {method: score(methods[method], water, land, mask) for method in METHODS}
            for stratum, mask in strata.items()
        },
    }


def _read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read()


def _pair_validation_and_provider(case: Path, provenance: dict) -> dict:
    """Run the canonical ingestion, pair-compatibility check and optical-sar provider."""
    from backend.services import ingest_scene, scene_compatibility, INGESTED_RASTER_DIR
    from models.optical_sar.model import OpticalSARModel

    group = f"sensor-necessity-{case.name}"
    optical = ingest_scene((case / "s2.tif").read_bytes(), "s2.tif", {
        "modality": "multispectral", "sensor": "Sentinel-2 L2A",
        "acquisition_timestamp": provenance["s2"]["datetime"], "pair_group": group,
        "benchmark_source": "planetary-computer"})
    sar = ingest_scene((case / "s1.tif").read_bytes(), "s1.tif", {
        "modality": "sar", "sensor": "Sentinel-1 RTC", "polarization": "VV,VH",
        "acquisition_timestamp": provenance["s1"]["datetime"], "pair_group": group,
        "benchmark_source": "planetary-computer"})
    compatibility = scene_compatibility(optical["scene_id"], sar["scene_id"], "optical_sar")
    provider = None
    if compatibility["eligible"]:
        result = OpticalSARModel().infer([
            str(INGESTED_RASTER_DIR / f"{optical['scene_id']}.tif"),
            str(INGESTED_RASTER_DIR / f"{sar['scene_id']}.tif"),
        ], "Is open water present?")
        provider = {"confidence": result["confidence"], "evidence": [
            {key: value for key, value in item.items() if key != "source"} for item in result["evidence"]
        ]}
    return {"compatibility": compatibility, "provider": provider}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="directory written by scripts/acquire_pc_s1s2_case.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-cols", type=int, nargs=2, metavar=("START", "END"),
                        help="post-hoc sensitivity: drop a column band whose reference is shown to be invalid")
    args = parser.parse_args()
    provenance = json.loads((args.case / "provenance.json").read_text())
    rule = frozen_rule()
    result = {
        "case": args.case.name,
        "rule": rule,
        "rule_source": str(FROZEN_RULE_PATH.relative_to(ROOT)),
        "definitions": {
            "reference_water": f"GSW occurrence >= {WATER_MIN_OCCURRENCE}, eroded {EROSION_PIXELS} px",
            "reference_land": f"GSW occurrence == 0, eroded {EROSION_PIXELS} px",
            "obstructed": f"S2 SCL in {list(OBSTRUCTED_SCL)}",
            "min_stratum_pixels_per_class": MIN_STRATUM_PIXELS,
        },
        "input_sha256": {name: meta["sha256"] for name, meta in provenance["files"].items()},
        "post_hoc_excluded_columns": args.exclude_cols,
        **evaluate_arrays(
            _read(args.case / "s2.tif"), _read(args.case / "s1.tif"),
            _read(args.case / "scl.tif")[0], _read(args.case / "gsw_occurrence.tif")[0], rule,
            tuple(args.exclude_cols) if args.exclude_cols else None,
        ),
    }
    if not args.exclude_cols:
        result.update(_pair_validation_and_provider(args.case, provenance))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({s: {m: v["balanced_accuracy"] for m, v in r.items()} for s, r in result["strata"].items()}, indent=2))


if __name__ == "__main__":
    main()
