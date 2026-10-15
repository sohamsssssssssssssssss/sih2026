"""Offline artifact API tests: synthetic local bundle, no acquisition or inference."""

import copy
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend import sar_change_artifacts as artifacts

METADATA = "/api/sar/change/kosi-2024"
RASTER = METADATA + "/raster"


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    count = 749 * 1649
    report = {
        "grid": copy.deepcopy(artifacts.GRID), "formula": copy.deepcopy(artifacts.FORMULA),
        "input_archives": copy.deepcopy(artifacts.INPUTS),
        "validity_criteria": {"valid_pixels": count, "total_pixels": count, "valid_fraction": 1.0},
        "statistics": {band: {"count": count, "min": -5., "max": 4., "mean": -1., "std": 2.}
                       for band in ("VV_change_dB", "VH_change_dB")},
        "code_fingerprints": {key: "a" * 64 for key in artifacts.CODE_IDS},
        "provenance_caveat": "PRIVATE_SECRET /private/capture",
    }
    for side in ("before", "after"):
        report["input_archives"][side]["path"] = "/private/capture/PRIVATE_SECRET.tar"
    raster = tmp_path / "sar_change.tif"
    raster.write_bytes(b"synthetic local raster bytes")
    raster_hash = hashlib.sha256(raster.read_bytes()).hexdigest()
    report["output_artifacts"] = {"sar_change_tif_sha256": raster_hash,
                                  "bands": ["VV_change_dB", "VH_change_dB", "valid_mask"],
                                  "nodata_value": -9999.}
    monkeypatch.setattr(artifacts, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(artifacts, "TIFF_SHA256", raster_hash)

    def publish(value=report):
        data = json.dumps(value).encode()
        (tmp_path / "sar_change.json").write_bytes(data)
        monkeypatch.setattr(artifacts, "JSON_SHA256", hashlib.sha256(data).hexdigest())

    publish()
    return tmp_path, report, publish


def test_curated_metadata_and_scientific_label(bundle):
    response = TestClient(app).get(METADATA)
    assert response.status_code == 200
    data = response.json()
    assert data["artifact_id"] == "kosi-2024-sar-change"
    assert data["scientific_result_type"] == "sar_backscatter_change"
    assert data["scientific_label"] == "Measured SAR backscatter change; flood interpretation unvalidated."
    assert data["flood_interpretation_validated"] is False
    assert data["raster_download_url"] == RASTER
    assert data["statistics"] == bundle[1]["statistics"]
    assert data["input_archives"] == artifacts.INPUTS
    assert data["formula"] == artifacts.FORMULA
    assert data["grid"] == artifacts.GRID
    assert set(data["code_fingerprints"]) == set(artifacts.CODE_IDS.values())
    assert data["output_artifact_hashes"] == {"sar_change.tif": artifacts.TIFF_SHA256,
                                             "sar_change.json": artifacts.JSON_SHA256}
    assert all(text not in response.text for text in (
        "PRIVATE_SECRET", "/private/", str(bundle[0]), "water_change_statistics",
        "water_change_polygons", "village_flooding", "userdata.json"))


def test_raster_retrieval(bundle):
    response = TestClient(app).get(RASTER)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert response.content == (bundle[0] / "sar_change.tif").read_bytes()
    assert 'filename="kosi-2024-sar-change.tif"' in response.headers["content-disposition"]
    assert str(bundle[0]) not in str(response.headers)


@pytest.mark.parametrize("endpoint", [METADATA, RASTER])
@pytest.mark.parametrize("filename", ["sar_change.tif", "sar_change.json"])
def test_missing_artifact_rejected(bundle, endpoint, filename):
    (bundle[0] / filename).unlink()
    response = TestClient(app).get(endpoint)
    assert response.status_code == 404
    assert str(bundle[0]) not in response.text


@pytest.mark.parametrize("endpoint", [METADATA, RASTER])
@pytest.mark.parametrize("filename", ["sar_change.tif", "sar_change.json"])
def test_wrong_checksum_rejected(bundle, endpoint, filename):
    with (bundle[0] / filename).open("ab") as output:
        output.write(b"corrupt")
    response = TestClient(app).get(endpoint)
    assert response.status_code == 409
    assert str(bundle[0]) not in response.text


@pytest.mark.parametrize("endpoint", [METADATA, RASTER])
@pytest.mark.parametrize("bad", ["recorded_hash", "product", "date", "input_hash", "grid", "formula",
                                 "bands", "statistics", "nonfinite", "coverage", "fingerprint", "missing"])
def test_invalid_metadata_rejected(bundle, endpoint, bad):
    _, report, publish = bundle
    if bad == "recorded_hash":
        report["output_artifacts"]["sar_change_tif_sha256"] = "b" * 64
    elif bad in ("product", "date", "input_hash"):
        key = {"product": "product_id", "date": "acquisition_date", "input_hash": "sha256"}[bad]
        report["input_archives"]["before"][key] = "PRIVATE_SECRET"
    elif bad == "grid":
        report["grid"]["bounds"][0] += 10
    elif bad == "formula":
        report["formula"]["units"] = "PRIVATE_SECRET"
    elif bad == "bands":
        report["output_artifacts"]["bands"].reverse()
    elif bad == "statistics":
        report["statistics"]["VV_change_dB"]["count"] = 1
    elif bad == "nonfinite":
        report["statistics"]["VV_change_dB"]["mean"] = float("nan")
    elif bad == "coverage":
        report["validity_criteria"]["valid_fraction"] = .5
    elif bad == "fingerprint":
        report["code_fingerprints"]["backend/sentinel1.py"] = "/private/PRIVATE_SECRET"
    else:
        report.pop("statistics")
    publish()
    response = TestClient(app).get(endpoint)
    assert response.status_code == 422
    assert "PRIVATE_SECRET" not in response.text


@pytest.mark.parametrize("endpoint", [METADATA, RASTER])
def test_malformed_json_rejected(bundle, monkeypatch, endpoint):
    data = b'{"PRIVATE_SECRET":'
    (bundle[0] / "sar_change.json").write_bytes(data)
    monkeypatch.setattr(artifacts, "JSON_SHA256", hashlib.sha256(data).hexdigest())
    response = TestClient(app).get(endpoint)
    assert response.status_code == 422
    assert "PRIVATE_SECRET" not in response.text


def test_no_arbitrary_artifact_ids_or_paths(bundle):
    assert TestClient(app).get("/api/sar/change/other/raster").status_code == 404
    response = TestClient(app).get(METADATA, params={"path": "/private/PRIVATE_SECRET", "artifact_id": "other"})
    assert response.status_code == 200
    assert response.json()["artifact_id"] == "kosi-2024-sar-change"
    assert "PRIVATE_SECRET" not in response.text
