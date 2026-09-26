import copy
import json

import pytest

from data.bigearthnet_pilot_consistency import MANIFEST_DIR, consistency_report

MANIFEST = json.loads((MANIFEST_DIR / "pilot-candidates.v1.json").read_text())
REFERENCE = json.loads((MANIFEST_DIR / "reference-map-audit.v1.json").read_text())


def test_tracked_pilot_manifests_are_mutually_consistent():
    report = consistency_report(MANIFEST, REFERENCE)

    assert report["issues"] == []
    assert all(report["member_lists_match_manifest"].values())
    assert set(report["checks_passed"].values()) == {100}
    assert report["abs_s1_s2_separation_hours"]["max"] < 72


def test_mismatched_reference_labels_are_reported():
    reference = copy.deepcopy(REFERENCE)
    reference["records"][0]["class_ids"] = [511]  # Inland waters only

    report = consistency_report(MANIFEST, reference)

    flagged = {issue["patch_id"] for issue in report["issues"]}
    assert reference["records"][0]["patch_id"] in flagged


def test_tampered_manifest_fails_checksum():
    manifest = copy.deepcopy(MANIFEST)
    manifest["pairs"][0]["split"] = "train" if manifest["pairs"][0]["split"] != "train" else "test"

    with pytest.raises(ValueError, match="checksum"):
        consistency_report(manifest, REFERENCE)
