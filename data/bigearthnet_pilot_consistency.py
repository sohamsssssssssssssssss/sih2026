"""Cross-check the tracked BigEarthNet pilot manifests against each other.

Uses only tracked files (pilot manifest, reference-map audit, member lists);
no imagery is read. What cannot be verified without the S1/S2 archives is
reported as unverified rather than assumed.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from rasterio.warp import transform

from data.bigearthnet_pilot import CLC_TO_19, verify_pilot_manifest

MANIFEST_DIR = Path(__file__).parent / "manifests" / "bigearthnet"


def _s2_location(patch_id: str) -> tuple[str, int, int]:
    tile, row, col = patch_id.rsplit("_", 3)[1:]
    return tile.removeprefix("T"), int(row), int(col)


def _s1_location(s1_name: str) -> tuple[str, int, int]:
    tile, row, col = s1_name.rsplit("_", 3)[1:]
    return tile, int(row), int(col)


def _time(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def _lines(name: str) -> list[str]:
    return (MANIFEST_DIR / name).read_text().split()


def consistency_report(manifest: dict, reference_audit: dict) -> dict:
    verify_pilot_manifest(manifest)
    references = {record["patch_id"]: record for record in reference_audit["records"]}
    rows, issues = [], []
    for pair in manifest["pairs"]:
        patch_id, s1_name = pair["patch_id"], pair["s1_name"]
        tile, row, col = _s2_location(patch_id)
        reference = references.get(patch_id)
        s2_time, s1_time = _time(patch_id.split("_")[2]), _time(s1_name.split("_")[4])
        entry = {
            "patch_id": patch_id,
            "split": pair["split"],
            "same_tile_row_col": _s1_location(s1_name) == (tile, row, col),
            "s1_minus_s2_hours": (s1_time - s2_time).total_seconds() / 3600,
            "reference_map_present": reference is not None,
        }
        if reference:
            epsg = int(reference["crs"].split(":")[1])
            xs, ys = transform("EPSG:4326", reference["crs"], [pair["longitude"]], [pair["latitude"]])
            left, bottom, right, top = reference["bounds"]
            reference_labels = {CLC_TO_19[c] for c in reference["class_ids"]} - {"Unlabeled"}
            entry.update(
                utm_zone_matches_tile=epsg == 32600 + int(tile[:2]),
                grid_is_120x120_at_10m=(reference["width"], reference["height"], reference["resolution"]) == (120, 120, [10.0, 10.0]),
                reference_time_matches_patch=reference["acquisition_timestamp"] == patch_id.split("_")[2],
                latlon_inside_reference_footprint=left <= xs[0] <= right and bottom <= ys[0] <= top,
                labels_equal_reference_map_classes=set(pair["labels"]) == reference_labels,
                labels_missing_from_reference_map=sorted(set(pair["labels"]) - reference_labels),
                reference_classes_missing_from_labels=sorted(reference_labels - set(pair["labels"])),
            )
        rows.append(entry)
        failed = [key for key, value in entry.items() if value is False]
        if failed:
            issues.append({"patch_id": patch_id, "failed": failed})
    expected = {
        "s1-members.v1.txt": [path for pair in manifest["pairs"] for path in pair["expected_assets"]["s1_bands"]],
        "s2-members.v1.txt": [path for pair in manifest["pairs"] for path in pair["expected_assets"]["s2_bands"]],
        "reference-map-members.v1.txt": [pair["expected_assets"]["reference_map"] for pair in manifest["pairs"]],
    }
    hours = sorted(abs(row["s1_minus_s2_hours"]) for row in rows)
    return {
        "version": "1.0",
        "inputs": ["pilot-candidates.v1.json", "reference-map-audit.v1.json", *expected],
        "manifest_sha256_verified": True,
        "pair_count": len(rows),
        "member_lists_match_manifest": {name: _lines(name) == paths for name, paths in expected.items()},
        "checks_passed": {
            key: sum(row.get(key) is True for row in rows)
            for key in ("same_tile_row_col", "reference_map_present", "utm_zone_matches_tile",
                        "grid_is_120x120_at_10m", "reference_time_matches_patch",
                        "latlon_inside_reference_footprint", "labels_equal_reference_map_classes")
        },
        "abs_s1_s2_separation_hours": {"min": hours[0], "median": hours[len(hours) // 2], "max": hours[-1]},
        "flagged_cloud_or_shadow": sum(pair["contains_cloud_or_shadow"] for pair in manifest["pairs"]),
        "flagged_seasonal_snow": sum(pair["contains_seasonal_snow"] for pair in manifest["pairs"]),
        "unverified_without_imagery": [
            "S1/S2 archive member existence", "S1/S2 native CRS, transforms, nodata and band units",
            "S1 polarization content and calibration", "pixel-level S1-S2 alignment",
        ],
        "issues": issues,
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=MANIFEST_DIR / "pilot-consistency.v1.json")
    args = parser.parse_args()
    report = consistency_report(
        json.loads((MANIFEST_DIR / "pilot-candidates.v1.json").read_text()),
        json.loads((MANIFEST_DIR / "reference-map-audit.v1.json").read_text()),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("member_lists_match_manifest", "checks_passed",
                                              "abs_s1_s2_separation_hours", "issues")}, indent=2)[:4000])


if __name__ == "__main__":
    main()
