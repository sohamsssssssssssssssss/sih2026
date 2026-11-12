from io import BytesIO
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.crs import CRS

from backend import services
from backend.main import app
from models.base import ModelReadiness
from models.grounding_dino import GroundingDINOModel
from models.qwen_vl import QwenVLModel
from orchestrator import router, trace


@pytest.mark.parametrize("gsd,status", [(10, "not_resolvable"), (0.3, "detected"), (None, "gsd_unknown")])
def test_gate_api_paths(gsd, status, monkeypatch, ready_providers):
    monkeypatch.setenv("SATQUERY_VQA_ADAPTER", "/synthetic")
    monkeypatch.setattr(services, "scene_gsd", lambda _: (gsd, "test_metadata"))
    if status == "not_resolvable":
        for cls in (GroundingDINOModel, QwenVLModel):
            monkeypatch.setattr(cls, "readiness", lambda _: ModelReadiness(False, "UNAVAILABLE", "offline"))
    calls = []
    def infer(model, paths, prompt, timeout):
        calls.append(model)
        return {"evidence": [{"type": "bounding_box", "coordinates": [0, 0, 0.5, 0.5],
                "coordinate_space": "normalized_xyxy", "confidence": 0.9}]}
    monkeypatch.setattr(router.inference_worker, "run", infer)
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={"scene_id": services.GOLDEN_SCENE_ID,
            "question": "How many buildings are there?", "execution_mode": "live"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["count_status"] == status and data["gsd_m"] == gsd
    assert data["required_gsd_m"] == 1 and data["object_class"] == "building"
    assert data["trace"]["params"]["count_status"] == status
    if status == "not_resolvable":
        assert not calls and "count" not in data
        assert data["model"]["name"] == "count-resolution-gate"
    else:
        assert calls == ["grounding-dino-swint"] and data["count"] == 1
        if gsd is None:
            assert data["unverified_gsd"] is True and "unverified_gsd" in data["notice"]
    assert trace.verify_chain()[0]


@pytest.mark.parametrize("epsg,expected", [(32643, 10), (4326, None), (2263, None)])
def test_tiff_upload_gsd_and_units(epsg, expected, tmp_path, monkeypatch):
    for attr, name in (("INGESTED_SCENE_DIR", "scenes"), ("INGESTED_RASTER_DIR", "rasters"), ("SCENE_MANIFEST_DIR", "manifests")):
        monkeypatch.setattr(services, attr, tmp_path / name)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=4, height=4, count=1, dtype="uint16", crs=CRS.from_epsg(epsg), transform=from_origin(100, 200, 10, 10)) as raster:
            raster.write(np.ones((4, 4), dtype="uint16"), 1)
        data = memory.read()
    uploaded = services.ingest_scene(data, "scene.tif")
    assert uploaded["gsd"] == ("10 m/px" if expected else None)
    assert services.scene_gsd(uploaded["scene_id"])[0] == expected


def test_metadata_only_and_preview_gsd(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "SCENE_MANIFEST_DIR", tmp_path)
    assert services.scene_gsd("example_gsd0.3") == (0.3, "scene_id")
    scene_id = "scene_" + "a" * 32
    path = tmp_path / f"{scene_id}.json"
    value = {"source": {"filename": "tile_gsd10.tif"}}
    path.write_text(json.dumps(value))
    assert services.scene_gsd(scene_id) == (10, "scene_metadata")
    value.update(raster={"crs_wkt": CRS.from_epsg(32643).to_wkt(), "transform": [0.3, 0, 100, 0, -0.3, 200],
                 "georeferencing_status": "affine", "width": 8, "height": 8}, preview={"width": 4, "height": 4})
    path.write_text(json.dumps(value))
    assert services.scene_gsd(scene_id) == (0.6, "projected_crs_preview")
