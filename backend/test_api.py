import hashlib
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("ready_providers")
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import backend.services as services
import models.paths as model_paths
import orchestrator.trace as trace_store
from backend.main import app
from backend.routes import analyze as analyze_routes
from backend.schemas import MAX_QUESTION_LENGTH
from models.base import ModelReadiness
from models.qwen_vl import QwenVLModel
from orchestrator import capabilities
from orchestrator import router as model_router


@pytest.fixture(autouse=True)
def isolated_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    services.load_results.cache_clear()
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    monkeypatch.setattr(trace_store, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(services, "INGESTED_SCENE_DIR", tmp_path / "scenes")
    monkeypatch.setattr(services, "INGESTED_RASTER_DIR", tmp_path / "rasters")
    monkeypatch.setattr(services, "SCENE_MANIFEST_DIR", tmp_path / "manifests")
    yield
    services.load_results.cache_clear()
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def image_bytes(format_name: str, size: tuple[int, int] = (4, 3)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(output, format=format_name)
    return output.getvalue()


def tiff_bytes(
    *,
    georeferenced: bool = True,
    bands: int = 2,
    size: tuple[int, int] = (6, 4),
    dtype: str = "uint16",
) -> bytes:
    width, height = size
    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": bands,
        "dtype": dtype,
        "nodata": 0,
    }
    if georeferenced:
        profile.update(crs="EPSG:32643", transform=from_origin(500000, 2000000, 10, 10))
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            for band in range(1, bands + 1):
                values = np.arange(width * height).reshape(height, width) + band
                dataset.write(values.astype(dtype), band)
        return memory.read()


def upload(
    client: TestClient,
    filename: str,
    data: bytes,
    metadata: dict[str, str] | None = None,
    content_type: str = "application/octet-stream",
) -> object:
    return client.post(
        "/api/scenes",
        files={"file": (filename, data, content_type)},
        data=metadata or {},
    )


def test_health_ready_when_trace_and_all_capabilities_are_ready(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["mode"] == "offline-first"
    assert body["checks"]["trace"] == {"ok": True, "detail": None}
    assert set(body["checks"]["capabilities"]) == set(capabilities.KNOWN_CAPABILITIES)
    assert all(
        check == {"available": True, "reason_code": None}
        for check in body["checks"]["capabilities"].values()
    )


def test_health_degraded_when_a_capability_is_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        QwenVLModel,
        "readiness",
        lambda _: ModelReadiness(False, "CUDA_UNAVAILABLE", "A CUDA GPU is required."),
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["mode"] == "offline-first"
    assert body["checks"]["trace"]["ok"] is True
    assert body["checks"]["capabilities"][capabilities.SINGLE_IMAGE_VQA] == {
        "available": False,
        "reason_code": "CUDA_UNAVAILABLE",
    }
    assert body["checks"]["capabilities"][capabilities.OPTICAL_SAR]["available"] is True


def test_health_unavailable_with_503_when_trace_is_corrupted(client: TestClient) -> None:
    trace_store.TRACE_PATH.write_text("not json\n", encoding="utf-8")
    trace_store._LOADED_PATH = None

    response = client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["mode"] == "offline-first"
    assert body["checks"]["trace"]["ok"] is False
    assert "malformed" in body["checks"]["trace"]["detail"]
    assert set(body["checks"]["capabilities"]) == set(capabilities.KNOWN_CAPABILITIES)


def test_known_golden_scene_serves_real_png(client: TestClient) -> None:
    response = client.get(f"/api/scenes/{services.GOLDEN_SCENE_ID}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_scene_image_returns_clean_404_for_unknown_scene(client: TestClient) -> None:
    response = client.get("/api/scenes/unknown-scene/image")
    assert response.status_code == 404
    assert response.json() == {"detail": "Local scene pixels are unavailable."}


def test_scene_image_rejects_path_traversal(client: TestClient) -> None:
    response = client.get("/api/scenes/..%2F..%2Fetc%2Fpasswd_gsd0.3/image")
    assert response.status_code == 404


def test_resolution_returns_committed_rungs(client: TestClient) -> None:
    response = client.get("/api/resolution")
    assert response.status_code == 200
    payload = response.json()
    assert list(payload["per_rung"]) == ["0.3", "1.0", "2.0", "5.0", "10.0"]
    assert payload["degenerate_rungs"] == ["5.0", "10.0"]
    assert payload["per_rung"]["0.3"]["open_accuracy"] == 0.335


def test_sar_returns_human_labeled_annotation(client: TestClient) -> None:
    response = client.get("/api/sar/mumbai")
    assert response.status_code == 200
    payload = response.json()
    assert payload["human_validation"] is True
    assert payload["title"] == "Mumbai coastal"
    assert set(payload["summaries"]) == {"water", "built_up", "vegetation", "terrain"}


def test_golden_analysis_replays_only_when_explicitly_requested(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gpu(**_: object) -> dict:
        raise RuntimeError("Qwen2.5-VL inference requires a CUDA GPU")

    monkeypatch.setattr(services, "route", no_gpu)
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "sensor": "LoveDA",
            "execution_mode": "cached_result",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Yes"
    assert payload["execution_mode"] == "cached_result"
    assert payload["trace"]["params"]["results_artifact"] == services.RESULTS_RELATIVE_PATH
    assert payload["trace"]["params"]["planner_version"] == "phase0-rules-v1"
    assert payload["trace"]["params"]["planner_rule"] == "default_single_image_vqa"
    # The committed artifact stores Kaggle absolute paths; only the file name is replayed.
    assert payload["trace"]["input_summary"]["image_paths"] == [
        "loveda_LoveDA_images_png_0_gsd0.3.png"
    ]
    traces = client.get("/api/traces")
    assert traces.status_code == 200
    for text in (response.text, traces.text):
        assert "/kaggle/" not in text
        assert str(services.ROOT) not in text



def test_live_failure_never_falls_back_to_cached_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(RuntimeError("CUDA GPU unavailable")),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
        },
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert trace_store.records() == []


def test_unready_provider_is_consistent_across_status_plan_and_execution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    monkeypatch.setattr(
        QwenVLModel,
        "readiness",
        lambda _: ModelReadiness(False, "CUDA_UNAVAILABLE", "A CUDA GPU is required."),
    )
    monkeypatch.setattr(services, "route", lambda **_: pytest.fail("unready provider ran"))
    request = {"scene_id": created["scene_id"], "question": "What is visible?"}

    status = client.get("/api/capabilities").json()["capabilities"][0]
    plan = client.post("/api/plan", json=request).json()
    response = client.post("/api/analyze", json=request)

    assert status["registered"] is True
    assert status["available"] is False
    assert status["reason_code"] == "CUDA_UNAVAILABLE"
    assert plan["provider_available"] is False
    assert plan["executable"] is False
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "capability": "single_image_vqa",
        "provider": "qwen2.5vl-3b",
        "reason_code": "CUDA_UNAVAILABLE",
        "detail": "A CUDA GPU is required.",
    }
    records = trace_store.records()
    assert len(records) == 1
    assert records[0]["params"]["result_state"] == "unavailable"

def test_unmatched_query_never_fabricates(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("CUDA GPU unavailable at /private/model secret-token")
        ),
    )
    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": "Invent an answer", "sensor": "LoveDA"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert "/private/model" not in response.text
    assert "secret-token" not in response.text
    assert "showing the exact committed result" not in response.text


def test_trace_history_and_verification(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "route", lambda **_: (_ for _ in ()).throw(RuntimeError("no GPU")))
    client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": services.GOLDEN_QUESTION, "sensor": "LoveDA", "execution_mode": "cached_result"},
    )
    history = client.get("/api/traces").json()
    assert history["count"] == 1
    verification = client.post("/api/traces/verify").json()
    assert verification == {"verified": True, "message": "Chain verified (1 records)"}


def test_modified_cached_artifact_fails_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modified = tmp_path / "ladder.json"
    modified.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(services, "RESULTS_PATH", modified)
    services.load_results.cache_clear()
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "execution_mode": "cached_result",
        },
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Required analysis artifacts are temporarily unavailable."
    }
    assert trace_store.records() == []


@pytest.mark.parametrize("endpoint", [("get", "/api/traces"), ("post", "/api/traces/verify")])

def test_corrupt_trace_returns_sanitized_503(
    client: TestClient, endpoint: tuple[str, str]
) -> None:
    corrupt_payload = "private-corrupt-payload"
    trace_store.TRACE_PATH.write_text(corrupt_payload + "\n", encoding="utf-8")

    method, url = endpoint
    response = getattr(client, method)(url)
    body = response.text

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Execution trace is temporarily unavailable because persisted trace "
            "integrity could not be verified."
        )
    }
    assert str(trace_store.TRACE_PATH) not in body
    assert corrupt_payload not in body
    assert "TraceIntegrityError" not in body
    assert "Traceback" not in body


@pytest.mark.parametrize("endpoint", [("get", "/api/traces"), ("post", "/api/traces/verify")])
def test_non_ascii_record_hash_returns_sanitized_503(
    client: TestClient, endpoint: tuple[str, str]
) -> None:
    corrupt_hash = "é"
    trace_store.TRACE_PATH.write_text(
        json.dumps(
            {"prev_hash": "", "record_hash": corrupt_hash}, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    method, url = endpoint
    response = getattr(client, method)(url)
    body = response.text

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Execution trace is temporarily unavailable because persisted trace "
            "integrity could not be verified."
        )
    }
    assert str(trace_store.TRACE_PATH) not in body
    assert corrupt_hash not in body
    assert "TraceIntegrityError" not in body
    assert "Traceback" not in body


def test_png_upload_returns_factual_metadata(client: TestClient) -> None:
    response = upload(client, "satellite.png", image_bytes("PNG", (7, 5)))

    assert response.status_code == 201
    payload = response.json()
    assert re.fullmatch(r"scene_[0-9a-f]{32}", payload["scene_id"])
    assert payload == {
        "scene_id": payload["scene_id"],
        "filename": "satellite.png",
        "format": "PNG",
        "width": 7,
        "height": 5,
        "sensor": None,
        "gsd": None,
        "location": None,
        "acquisition_date": None,
    }
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{payload['scene_id']}.json").read_text()
    )
    assert manifest["source"]["native_path"] is None
    assert manifest["source"]["format"] == "PNG"
    assert manifest["raster"] is None



def test_uploaded_pixels_do_not_inherit_curated_scene_identity(
    client: TestClient,
) -> None:
    asset = services.local_scene_image(services.GOLDEN_SCENE_ID)
    assert asset is not None
    response = upload(client, "copied-golden.png", asset.read_bytes(), content_type="image/png")
    assert response.status_code == 201
    payload = response.json()
    assert payload["scene_id"].startswith("scene_")
    assert payload["sensor"] is None
    assert payload["gsd"] is None
    assert payload["location"] is None
    assert payload["acquisition_date"] is None
    scene_id = payload["scene_id"]
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{scene_id}.json").read_text()
    )
    assert manifest["identity"]["provenance"] == {}
    assert manifest["grouping"]["original_split"] is None

def test_jpeg_upload_is_served_as_canonical_png(client: TestClient) -> None:
    response = upload(client, "satellite.jpg", image_bytes("JPEG"))

    assert response.status_code == 201
    payload = response.json()
    assert payload["format"] == "JPEG"
    retrieved = client.get(f"/api/scenes/{payload['scene_id']}/image")
    assert retrieved.status_code == 200
    assert retrieved.headers["content-type"] == "image/png"
    with Image.open(BytesIO(retrieved.content)) as image:
        assert image.format == "PNG"
        assert image.size == (4, 3)


def test_geotiff_content_is_preserved_and_manifested_regardless_of_name(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(services, "MAX_PREVIEW_DIMENSION", 3)
    original = tiff_bytes()
    response = upload(client, "renamed.bin", original)

    assert response.status_code == 201
    payload = response.json()
    assert payload["format"] == "TIFF"
    assert payload["filename"] == "renamed.bin"
    scene_id = payload["scene_id"]
    native = services.INGESTED_RASTER_DIR / f"{scene_id}.tif"
    preview = services.INGESTED_SCENE_DIR / f"{scene_id}.png"
    manifest_path = services.SCENE_MANIFEST_DIR / f"{scene_id}.json"
    assert native.read_bytes() == original
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "1.0"
    assert manifest["source"] == {
        "filename": "renamed.bin",
        "format": "TIFF",
        "native_path": f"data/runtime/rasters/{scene_id}.tif",
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    raster = manifest["raster"]
    assert raster["crs_epsg"] == 32643
    assert raster["transform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0]
    assert raster["bounds"] == [500000.0, 1999960.0, 500060.0, 2000000.0]
    assert raster["resolution"] == [10.0, 10.0]
    assert raster["band_count"] == 2
    assert raster["dtypes"] == ["uint16", "uint16"]
    assert raster["nodata"] == [0.0, 0.0]
    assert raster["georeferencing_status"] == "affine"
    assert raster["pairing_ready"] is True
    assert manifest["preview"]["width"] == 3
    assert manifest["preview"]["height"] == 2
    assert preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    served = client.get(f"/api/scenes/{scene_id}/image")
    assert served.status_code == 200
    assert served.content == preview.read_bytes()
    assert served.content != original


def test_plain_tiff_records_missing_georeferencing(client: TestClient) -> None:
    response = upload(client, "plain.tiff", tiff_bytes(georeferenced=False, bands=1))

    assert response.status_code == 201
    scene_id = response.json()["scene_id"]
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{scene_id}.json").read_text(encoding="utf-8")
    )
    raster = manifest["raster"]
    assert raster["crs_wkt"] is None
    assert raster["crs_epsg"] is None
    assert raster["georeferencing_status"] == "missing_crs"
    assert raster["pairing_ready"] is False


def test_tiff_safety_limit_rejects_without_residue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "MAX_RASTER_PIXELS", 10)

    response = upload(client, "large.tif", tiff_bytes(size=(6, 4)))

    assert response.status_code == 422
    assert not services.INGESTED_SCENE_DIR.exists()
    assert not services.INGESTED_RASTER_DIR.exists()
    assert not services.SCENE_MANIFEST_DIR.exists()


def test_tiff_band_limit_rejects_without_residue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "MAX_RASTER_BANDS", 1)

    response = upload(client, "too-many-bands.tif", tiff_bytes(bands=2))

    assert response.status_code == 422
    assert not services.INGESTED_SCENE_DIR.exists()


def test_tiff_unsupported_complex_content_is_rejected(client: TestClient) -> None:
    response = upload(client, "complex.tif", tiff_bytes(bands=1, dtype="complex64"))

    assert response.status_code == 422
    assert not services.INGESTED_SCENE_DIR.exists()


def test_malformed_tiff_signature_is_rejected_without_residue(client: TestClient) -> None:
    response = upload(client, "not-really.png", b"II*\x00broken")

    assert response.status_code == 422
    assert not services.INGESTED_SCENE_DIR.exists()
    assert not services.INGESTED_RASTER_DIR.exists()
    assert not services.SCENE_MANIFEST_DIR.exists()


def test_spoofed_filename_and_mime_do_not_override_tiff_content(
    client: TestClient,
) -> None:
    response = upload(
        client,
        "looks-optical.jpg",
        tiff_bytes(),
        {"modality": "sar", "polarization": "VV"},
        "image/jpeg",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["format"] == "TIFF"
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{payload['scene_id']}.json").read_text()
    )
    assert manifest["identity"]["modality"] == "sar"
    assert manifest["identity"]["polarizations"] == ["VV"]


@pytest.mark.parametrize(("format_name", "filename"), [("PNG", "benchmark.tif"), ("JPEG", "benchmark.bin")])
def test_png_jpeg_accept_declared_benchmark_provenance(
    client: TestClient, format_name: str, filename: str
) -> None:
    response = upload(
        client,
        filename,
        image_bytes(format_name),
        {
            "modality": "multispectral",
            "sensor": "Sentinel-2 MSI",
            "acquisition_timestamp": "2026-01-02T03:04:05Z",
            "pair_group": "pilot-pair-7",
            "benchmark_source": "audited-pilot",
        },
        "image/tiff",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["format"] == format_name
    assert payload["sensor"] == "Sentinel-2 MSI"
    assert payload["acquisition_date"] == "2026-01-02T03:04:05+00:00"
    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{payload['scene_id']}.json").read_text()
    )
    assert manifest["identity"]["benchmark_source"] == "audited-pilot"
    assert manifest["identity"]["provenance"]["benchmark_source"] == "user_declared_upload"
    assert manifest["grouping"]["pair_group"] == "pilot-pair-7"
    assert manifest["raster"] is None


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"modality": "thermal"}, "Unsupported declared modality."),
        (
            {"acquisition_timestamp": "2026-01-02T03:04:05"},
            "Acquisition timestamp must include a timezone.",
        ),
    ],
)
def test_invalid_declared_metadata_is_rejected(
    client: TestClient, metadata: dict[str, str], message: str
) -> None:
    response = upload(client, "scene.png", image_bytes("PNG"), metadata)
    assert response.status_code == 422
    assert response.json() == {"detail": message}
    assert not services.INGESTED_SCENE_DIR.exists()


def test_incompatible_pair_returns_structured_result_without_model_dispatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "modality": "optical",
        "sensor": "test-optical",
        "acquisition_timestamp": "2026-01-01T00:00:00Z",
        "pair_group": "pair-1",
    }
    first = upload(client, "one.tif", tiff_bytes(), common).json()["scene_id"]
    second = upload(client, "two.tif", tiff_bytes(), common).json()["scene_id"]
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": first,
            "scene_id_2": second,
            "question": "Compare optical and SAR.",
            "capability": "optical_sar",
        },
    )

    assert response.status_code == 422
    compatibility = response.json()["detail"]
    assert compatibility["eligible"] is False
    assert compatibility["requested_workflow"] == "optical_sar"
    assert "modalities_incompatible" in compatibility["reason_codes"]
    assert trace_store.records() == []


def test_compatible_pair_executes_joint_optical_sar_provider(
    client: TestClient,
) -> None:
    optical = upload(
        client,
        "optical.tif",
        tiff_bytes(bands=5, dtype="float32"),
        {
            "modality": "multispectral",
            "sensor": "test-optical",
            "acquisition_timestamp": "2026-01-01T00:00:00Z",
            "pair_group": "pair-1",
        },
    ).json()["scene_id"]
    sar = upload(
        client,
        "sar.tif",
        tiff_bytes(bands=3, dtype="float32"),
        {
            "modality": "sar",
            "sensor": "test-sar",
            "acquisition_timestamp": "2026-01-02T00:00:00Z",
            "polarization": "VV,VH",
            "pair_group": "pair-1",
        },
    ).json()["scene_id"]

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": optical,
            "scene_id_2": sar,
            "question": "Compare optical and SAR.",
            "capability": "optical_sar",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["execution_mode"] == "live"
    assert payload["results_artifact"] is None
    assert payload["model"] == {
        "name": "optical-sar-deterministic",
        "version": "sentinel2-indices__sentinel1-backscatter-v1",
    }
    assert [item["type"] for item in payload["evidence"]] == [
        "optical_statistics", "sar_statistics", "joint_valid_coverage"
    ]
    assert payload["trace"]["input_summary"]["n_images"] == 2
    assert payload["trace"]["params"]["input_modalities"] == ["optical", "sar"]
    records = trace_store.records()
    assert len(records) == 1
    assert records[0]["model_name"] == "optical-sar-deterministic"
    assert records[0]["input_summary"]["question"] == "Compare optical and SAR."
    assert trace_store.verify_chain()[0] is True


def test_compatible_bitemporal_pair_executes_change_provider(
    client: TestClient,
) -> None:
    before = upload(
        client,
        "before.tif",
        tiff_bytes(bands=3, dtype="float32"),
        {
            "modality": "optical",
            "sensor": "test-rgb",
            "acquisition_timestamp": "2026-01-01T00:00:00Z",
            "pair_group": "change-pair-1",
        },
    ).json()["scene_id"]
    after = upload(
        client,
        "after.tif",
        tiff_bytes(bands=3, dtype="float32"),
        {
            "modality": "optical",
            "sensor": "test-rgb",
            "acquisition_timestamp": "2026-01-02T00:00:00Z",
            "pair_group": "change-pair-1",
        },
    ).json()["scene_id"]

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": before,
            "scene_id_2": after,
            "question": "Did built-up area increase?",
            "capability": "change_vqa",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == {
        "name": "change-deterministic",
        "version": "bitemporal-difference-v2",
    }
    assert "confidence" not in payload
    assert "does not infer semantic change classes" in payload["answer"]
    assert [item["type"] for item in payload["evidence"]] == [
        "temporal_inputs", "change_statistics", "temporal_valid_coverage"
    ]
    assert payload["trace"]["input_summary"]["n_images"] == 2
    assert payload["trace"]["params"]["temporal_order"] == ["t1", "t2"]
    assert payload["trace"]["params"]["acquisition_times"] == [
        "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"
    ]
    assert trace_store.records()[0]["model_name"] == "change-deterministic"
    assert trace_store.verify_chain()[0] is True


def sar_tiff_bytes(water_rows: slice) -> bytes:
    """VV/VH/dataMask GeoTIFF: land at about -10 dB, ``water_rows`` at about -23 dB."""
    vv = np.full((40, 40), 0.1, dtype="float32")
    vv[water_rows] = 0.005
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=40, height=40, count=3, dtype="float32",
            crs="EPSG:32645", transform=from_origin(500000, 2830000, 10, 10),
        ) as dataset:
            dataset.write(np.stack([vv, vv / 5, np.ones_like(vv)]))
            dataset.descriptions = ("VV", "VH", "dataMask")
        return memory.read()


def test_sar_pair_flood_question_returns_water_change_geojson(client: TestClient) -> None:
    metadata = {"modality": "sar", "sensor": "Sentinel-1", "polarization": "VV,VH", "pair_group": "flood-1"}
    before = upload(
        client, "before.tif", sar_tiff_bytes(slice(0, 5)),
        {**metadata, "acquisition_timestamp": "2026-08-01T00:00:00Z"},
    ).json()["scene_id"]
    after = upload(
        client, "after.tif", sar_tiff_bytes(slice(0, 15)),
        {**metadata, "acquisition_timestamp": "2026-08-20T00:00:00Z"},
    ).json()["scene_id"]

    response = client.post(
        "/api/analyze",
        json={"scene_id": before, "scene_id_2": after, "question": "Did flooding expand between these scenes?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"]["name"] == "change-deterministic"
    assert payload["trace"]["params"]["capability"] == "change_vqa"
    evidence = {item["type"]: item for item in payload["evidence"]}
    assert evidence["water_change_statistics"]["new_water_ha"] == pytest.approx(4.0, rel=1e-2)
    features = evidence["water_change_polygons"]["geojson"]["features"]
    assert [feature["properties"] for feature in features] == [{"change": "new_water", "pixels": 400}]
    assert "4.0 ha became open water" in payload["answer"]
    assert trace_store.verify_chain()[0] is True


def test_scene_id_is_not_derived_from_malicious_filename(client: TestClient) -> None:
    response = upload(client, "../../owned.png", image_bytes("PNG"))

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "owned.png"
    assert payload["scene_id"] != "owned"
    assert all(value not in payload["scene_id"] for value in ("/", "\\", ".."))


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("fake.png", b"plain text"),
        ("corrupt.jpg", b"\xff\xd8corrupt"),
        ("empty.png", b""),
        ("unsupported.bmp", image_bytes("BMP")),
    ],
)
def test_invalid_uploads_are_rejected(
    client: TestClient, filename: str, data: bytes
) -> None:
    response = upload(client, filename, data)

    assert response.status_code == 422
    assert not services.INGESTED_SCENE_DIR.exists()
    assert str(services.INGESTED_SCENE_DIR) not in response.text


def test_oversized_upload_returns_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert analyze_routes.MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    monkeypatch.setattr(analyze_routes, "MAX_UPLOAD_BYTES", 8)

    response = upload(client, "large.png", b"123456789")

    assert response.status_code == 413
    assert "20 MiB" in response.json()["detail"]
    assert not services.INGESTED_SCENE_DIR.exists()


def test_decompression_bomb_warning_is_sanitized(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    response = upload(client, "large-dimensions.png", image_bytes("PNG", (4, 4)))

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The uploaded file is not a safe, valid image."
    }
    assert "DecompressionBomb" not in response.text
    assert not services.INGESTED_SCENE_DIR.exists()


def test_uploaded_png_can_be_retrieved(client: TestClient) -> None:
    created = upload(client, "scene.png", image_bytes("PNG", (6, 2))).json()

    response = client.get(f"/api/scenes/{created['scene_id']}/image")

    assert response.status_code == 200
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (6, 2)


def test_unknown_scene_id_returns_clean_404(client: TestClient) -> None:
    response = client.get(f"/api/scenes/scene_{'0' * 32}/image")

    assert response.status_code == 404
    assert response.json() == {"detail": "Local scene pixels are unavailable."}


@pytest.mark.parametrize("scene_id", ["..%2F..%2Fetc%2Fpasswd", "..\\..\\etc\\passwd"])
def test_unsafe_scene_ids_return_404(client: TestClient, scene_id: str) -> None:
    response = client.get(f"/api/scenes/{scene_id}/image")

    assert response.status_code == 404
    assert "/etc/passwd" not in response.text


def test_missing_scene_returns_404_before_provider_readiness(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        QwenVLModel,
        "readiness",
        lambda _: ModelReadiness(False, "CUDA_UNAVAILABLE", "private runtime detail"),
    )

    response = client.post(
        "/api/analyze",
        json={"scene_id": "scene_missing", "question": "What is visible?"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Local scene pixels are unavailable."}
    assert "private runtime detail" not in response.text
    assert trace_store.records() == []


def test_uploaded_scene_is_resolved_for_live_analysis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    routed: dict[str, object] = {}

    def fake_route(**kwargs: object) -> dict:
        routed.update(kwargs)
        return {
            "answer": "uploaded scene analyzed",
            "trace": {
                "model_name": services.MODEL_NAME,
                "model_version": "test",
                "params": {"execution_mode": "live"},
            },
        }

    monkeypatch.setattr(services, "route", fake_route)
    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "answer",
        "execution_mode",
        "results_artifact",
        "model",
        "trace",
        "notice",
    }
    assert payload["answer"] == "uploaded scene analyzed"
    assert payload["notice"] == "Live Qwen2.5-VL-3B inference completed."
    assert routed["image_paths"] == [
        str(services.INGESTED_SCENE_DIR / f"{created['scene_id']}.png")
    ]
    assert routed["params"] == {
        "scene_id": created["scene_id"],
        "sensor": None,
        "execution_mode": "live",
    }
    assert routed["planner_version"] == "phase0-rules-v1"
    assert routed["planner_rule"] == "default_single_image_vqa"
    assert routed["requested_capability"] is None


def test_uploaded_scene_never_uses_golden_cached_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": created["scene_id"],
            "question": services.GOLDEN_QUESTION,
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model execution failed."}
    assert "showing the exact committed result" not in response.text
    assert "model unavailable" not in response.text


def test_failed_storage_leaves_no_scene(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = image_bytes("PNG")

    def fail_save(*_: object, **__: object) -> None:
        raise OSError("private storage failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    response = upload(client, "scene.png", data)

    assert response.status_code == 500
    assert response.json() == {"detail": "The uploaded image could not be stored."}
    assert "private storage failure" not in response.text
    assert list(services.INGESTED_SCENE_DIR.iterdir()) == []
    assert list(services.SCENE_MANIFEST_DIR.iterdir()) == []


def test_failed_tiff_manifest_leaves_no_native_or_preview(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_manifest(_: object) -> None:
        raise ValueError("invalid manifest")

    monkeypatch.setattr(services, "validate_scene_manifest", fail_manifest)
    response = upload(client, "scene.tif", tiff_bytes())

    assert response.status_code == 500
    assert list(services.INGESTED_SCENE_DIR.iterdir()) == []
    assert list(services.INGESTED_RASTER_DIR.iterdir()) == []
    assert list(services.SCENE_MANIFEST_DIR.iterdir()) == []


def test_valid_model_output_returns_answer_and_one_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test",
        infer=lambda **_: {"answer": "  Yes  ", "evidence": []},
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Yes"
    assert response.json()["execution_mode"] == "live"
    assert len(trace_store.records()) == 1
    assert trace_store.verify_chain()[0] is True


@pytest.mark.parametrize(
    "model_output",
    [
        None,
        [],
        "answer",
        {},
        {"answer": ""},
        {"answer": "   "},
        {"answer": "Yes", "evidence": None},
        {"answer": "Yes", "evidence": {}},
    ],
)
def test_invalid_model_outputs_return_502_without_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model_output: object
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(version="test", infer=lambda **_: model_output)
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model returned an invalid response."}
    assert repr(model_output) not in response.text
    assert trace_store.records() == []


def test_generic_model_exception_returns_sanitized_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()

    def fail(**_: object) -> dict:
        raise RuntimeError("CUDA exploded at /Users/secret/model.pt token=abc123")

    monkeypatch.setattr(
        model_router, "get", lambda _: SimpleNamespace(version="test", infer=fail)
    )
    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model execution failed."}
    assert "/Users/secret/model.pt" not in response.text
    assert "token=abc123" not in response.text
    assert "RuntimeError" not in response.text
    assert trace_store.records() == []


def test_no_gpu_returns_503_for_uploaded_scene(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()

    def no_gpu(**_: object) -> dict:
        raise RuntimeError("CUDA GPU unavailable with secret-token")

    monkeypatch.setattr(
        model_router, "get", lambda _: SimpleNamespace(version="test", infer=no_gpu)
    )
    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert "secret-token" not in response.text
    assert trace_store.records() == []


@pytest.mark.parametrize(
    "failure",
    [
        trace_store.TraceIntegrityError("private trace detail"),
        OSError("/tmp/private/path/secret"),
    ],
)
def test_trace_failure_after_valid_output_returns_503_without_answer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)
    monkeypatch.setattr(
        model_router,
        "append_record",
        lambda _: (_ for _ in ()).throw(failure),
    )

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Execution trace is temporarily unavailable."}
    assert "Yes" not in response.text
    assert "private" not in response.text
    assert "OSError" not in response.text
    assert "Traceback" not in response.text
    assert trace_store.records() == []


def test_invalid_execution_mode_returns_502_without_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test",
        infer=lambda **_: {
            "answer": "Yes",
            "evidence": [],
            "execution_mode": "cached_result",
        },
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model returned an invalid response."}
    assert trace_store.records() == []


def test_invalid_model_metadata_returns_502_without_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version=None,
        infer=lambda **_: {"answer": "Yes", "evidence": []},
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model returned an invalid response."}
    assert trace_store.records() == []


def test_unexpected_service_exception_returns_sanitized_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> dict:
        raise RuntimeError("failure at /Users/secret/service.py token=abc123")

    monkeypatch.setattr(analyze_routes, "analyze_scene", fail)

    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": "Question"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model execution failed."}
    assert "/Users/secret/service.py" not in response.text
    assert "token=abc123" not in response.text
    assert "RuntimeError" not in response.text


def test_invalid_cached_answer_fails_before_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "find_cached_result",
        lambda *_: {
            "tile_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "prediction": {},
        },
    )
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(RuntimeError("CUDA GPU unavailable")),
    )

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "sensor": "LoveDA",
            "execution_mode": "cached_result",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Required analysis artifacts are temporarily unavailable."
    }
    assert trace_store.records() == []


def test_whitespace_question_is_rejected_without_execution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    response = client.post(
        "/api/analyze", json={"scene_id": created["scene_id"], "question": "   "}
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "A non-empty question is required. No answer was generated."
    }


def test_oversized_question_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": "x" * (MAX_QUESTION_LENGTH + 1)},
    )

    assert response.status_code == 422


def test_model_timeout_bounds_request_without_success_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    release = Event()
    finished = Event()

    def wait_forever(**_: object) -> dict:
        try:
            release.wait(2)
            return {"answer": "late answer", "evidence": []}
        finally:
            finished.set()

    monkeypatch.setattr(
        model_router,
        "get",
        lambda _: SimpleNamespace(version="test", infer=wait_forever),
    )
    monkeypatch.setattr(services, "MODEL_EXECUTION_TIMEOUT_SECONDS", 0.01)

    started = time.monotonic()
    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )
    elapsed = time.monotonic() - started
    release.set()

    assert finished.wait(1)
    assert elapsed < 0.5
    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert "late answer" not in response.text
    assert trace_store.records() == []


def test_capability_resolution_maps_vqa_to_qwen() -> None:
    resolved = capabilities.resolve_provider(capabilities.SINGLE_IMAGE_VQA)
    assert resolved.provider_name == "qwen2.5vl-3b"
    assert resolved.model_name == services.MODEL_NAME


def test_provider_metadata_is_truthful() -> None:
    resolved = capabilities.resolve_provider(capabilities.SINGLE_IMAGE_VQA)
    model = model_router.get(resolved.model_name)
    assert resolved.provider_name == model.name
    assert resolved.model_version == model.version
    status = {entry["name"]: entry for entry in capabilities.capabilities_status()}
    assert status[capabilities.SINGLE_IMAGE_VQA] == {
        "name": "single_image_vqa",
        "registered": True,
        "available": True,
        "state": "AVAILABLE",
        "provider": "qwen2.5vl-3b",
        "reason_code": None,
        "detail": None,
    }


def test_capabilities_endpoint_reports_truthful_availability(
    client: TestClient,
) -> None:
    payload = client.get("/api/capabilities").json()
    assert payload == {
        "capabilities": [
            {"name": "single_image_vqa", "registered": True, "available": True, "state": "AVAILABLE", "provider": "qwen2.5vl-3b", "reason_code": None, "detail": None},
            {"name": "grounding", "registered": True, "available": True, "state": "AVAILABLE", "provider": "grounding-dino-swint", "reason_code": None, "detail": None},
            {"name": "change_vqa", "registered": True, "available": True, "state": "AVAILABLE", "provider": "change-deterministic", "reason_code": None, "detail": None},
            {"name": "optical_sar", "registered": True, "available": True, "state": "AVAILABLE", "provider": "optical-sar-deterministic", "reason_code": None, "detail": None},
        ]
    }


def test_trace_records_requested_capability(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "What is visible?"},
    )

    assert response.status_code == 200
    records = trace_store.records()
    assert len(records) == 1
    assert records[0]["params"]["capability"] == "single_image_vqa"
    assert records[0]["params"]["planner_version"] == "phase0-rules-v1"
    assert records[0]["params"]["planner_rule"] == "default_single_image_vqa"
    assert records[0]["params"]["requested_capability"] is None
    assert records[0]["model_name"] == "qwen2.5vl-3b"
    assert records[0]["model_version"] == "test"
    assert trace_store.verify_chain()[0] is True


def test_unknown_capability_is_rejected_without_execution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_: object) -> dict:
        raise AssertionError("model must not run")

    monkeypatch.setattr(services, "route", fail)
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "capability": "time_travel",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown capability: time_travel"}
    assert trace_store.records() == []


def test_grounding_analysis_returns_normalized_bounding_box_evidence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    evidence = [
        {
            "type": "bounding_box",
            "label": "building",
            "coordinates": [0.1, 0.2, 0.7, 0.8],
            "coordinate_space": "normalized_xyxy",
            "confidence": 0.91,
            "source_scene_id": None,
        }
    ]
    model = SimpleNamespace(
        version="test-grounding",
        infer=lambda **_: {"answer": "Found 1 match.", "evidence": evidence},
    )

    monkeypatch.setattr(model_router, "get", lambda _: model)
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": created["scene_id"],
            "question": "Locate the building.",
            "capability": "grounding",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Found 1 match."
    assert payload["evidence"] == evidence
    assert payload["model"] == {
        "name": "grounding-dino-swint",
        "version": "test-grounding",
    }
    assert payload["trace"]["params"]["capability"] == "grounding"
    assert payload["trace"]["input_summary"]["question"] == "Locate the building."
    assert len(trace_store.records()) == 1
    assert trace_store.verify_chain()[0] is True


@pytest.mark.parametrize("capability", ["change_vqa", "optical_sar"])
def test_explicit_pair_capability_reports_missing_second_scene_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "capability": capability,
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "This request requires two scenes."}
    assert trace_store.records() == []


def test_golden_fallback_rejects_unsupported_capability(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_: object) -> dict:
        raise RuntimeError("Grounding DINO inference requires a CUDA GPU")

    monkeypatch.setattr(services, "route", fail)
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "capability": "grounding",
        },
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert "showing the exact committed result" not in response.text
    assert trace_store.records() == []


def test_inferred_grounding_reports_model_unavailable_truthfully(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("Grounding DINO inference requires a CUDA GPU")
        ),
    )
    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": "Where is the building?"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    assert trace_store.records() == []


@pytest.mark.parametrize(
    "question",
    [
        "What changed between these images?",
        "Compare the optical and SAR images.",
    ],
)
def test_inferred_pair_capability_requires_second_scene(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, question: str
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": question},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "This request requires two scenes."}
    assert trace_store.records() == []


def test_explicit_vqa_overrides_grounding_wording(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    monkeypatch.setattr(
        model_router,
        "get",
        lambda _: SimpleNamespace(
            version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
        ),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": created["scene_id"],
            "question": "Where is the building?",
            "capability": "single_image_vqa",
        },
    )
    assert response.status_code == 200
    params = response.json()["trace"]["params"]
    assert params["capability"] == "single_image_vqa"
    assert params["planner_rule"] == "explicit_capability"
    assert params["requested_capability"] == "single_image_vqa"


def test_caller_cannot_forge_planner_trace_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    monkeypatch.setattr(
        model_router,
        "get",
        lambda _: SimpleNamespace(
            version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
        ),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": created["scene_id"],
            "question": "Is water visible?",
            "planner_version": "forged",
            "planner_rule": "forged",
        },
    )
    assert response.status_code == 200
    params = response.json()["trace"]["params"]
    assert params["planner_version"] == "phase0-rules-v1"
    assert params["planner_rule"] == "default_single_image_vqa"


def test_sar_labeled_single_image_analysis_is_not_claimed_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": "Is water visible?",
            "sensor": "SAR",
        },
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Required capability is not currently available."
    }
    assert trace_store.records() == []


def test_plan_endpoint_reports_vqa_without_model_or_trace(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    request = {
        "scene_id": services.GOLDEN_SCENE_ID,
        "question": "Is there a building?",
    }
    first = client.post("/api/plan", json=request)
    second = client.post("/api/plan", json=request)
    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json() == {
        "planner_version": "phase0-rules-v1",
        "rule_id": "default_single_image_vqa",
        "requested_capability": None,
        "selected_capability": "single_image_vqa",
        "executable": True,
        "reason": "The request asks about the contents of a single scene.",
        "required_inputs": ["single_scene"],
        "missing_inputs": [],
        "provider_available": True,
        "provider": "qwen2.5vl-3b",
        "unavailable_reason": None,
        "execution_plan_version": "phase0-plan-v1",
        "steps": [
            {
                "step_id": "step_1",
                "capability": "single_image_vqa",
                "depends_on": [],
                "required_inputs": ["single_scene"],
                "provider_available": True,
                "provider": "qwen2.5vl-3b",
            }
        ],
        "unavailable_capabilities": [],
    }
    assert trace_store.records() == []


def test_plan_endpoint_reports_unavailable_and_missing_inputs(
    client: TestClient,
) -> None:
    grounding = client.post(
        "/api/plan",
        json={"scene_id": "scene_a", "question": "Locate the road."},
    )
    change = client.post(
        "/api/plan",
        json={"scene_id": "scene_a", "question": "What changed?"},
    )
    assert grounding.status_code == 200
    assert grounding.json()["selected_capability"] == "grounding"
    assert grounding.json()["provider_available"] is True
    assert grounding.json()["executable"] is True
    assert change.status_code == 200
    assert change.json()["selected_capability"] == "change_vqa"
    assert change.json()["missing_inputs"] == ["second_scene"]
    assert change.json()["executable"] is False


def test_plan_endpoint_shows_explicit_override(client: TestClient) -> None:
    response = client.post(
        "/api/plan",
        json={
            "scene_id": "scene_a",
            "question": "Where is the building?",
            "capability": "single_image_vqa",
        },
    )
    assert response.status_code == 200
    assert response.json()["selected_capability"] == "single_image_vqa"
    assert response.json()["requested_capability"] == "single_image_vqa"
    assert response.json()["rule_id"] == "explicit_capability"


def test_plan_endpoint_rejects_unknown_capability(client: TestClient) -> None:
    response = client.post(
        "/api/plan",
        json={
            "scene_id": "scene_a",
            "question": "Question",
            "capability": "time_travel",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown capability: time_travel"}
    assert trace_store.records() == []


def test_plan_endpoint_reports_available_grounding_step(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/plan",
        json={"scene_id": "scene_a", "question": "Where is the building?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_plan_version"] == "phase0-plan-v1"
    assert payload["steps"] == [
        {
            "step_id": "step_1",
            "capability": "grounding",
            "depends_on": [],
            "required_inputs": ["single_scene"],
            "provider_available": True,
            "provider": "grounding-dino-swint",
        }
    ]
    assert payload["unavailable_capabilities"] == []
    assert payload["executable"] is True


def test_plan_endpoint_reports_two_step_chain_for_temporal_localization(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/plan",
        json={
            "scene_id": "scene_a",
            "question": "Where did flooding increase between these two scenes?",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rule_id"] == "temporal_change_then_grounding"
    assert payload["selected_capability"] == "change_vqa"
    assert payload["steps"] == [
        {
            "step_id": "step_1",
            "capability": "change_vqa",
            "depends_on": [],
            "required_inputs": ["scene_pair"],
            "provider_available": True,
            "provider": "change-deterministic",
        },
        {
            "step_id": "step_2",
            "capability": "grounding",
            "depends_on": ["step_1"],
            "required_inputs": ["step_1.output"],
            "provider_available": True,
            "provider": "grounding-dino-swint",
        },
    ]
    assert payload["unavailable_capabilities"] == []
    assert payload["executable"] is False
    assert trace_store.records() == []


@pytest.mark.parametrize("endpoint", ["/api/plan", "/api/analyze"])
def test_planner_rejects_punctuation_only_question_as_client_error(
    client: TestClient, endpoint: str
) -> None:
    response = client.post(
        endpoint,
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": " ... !!! "},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "A non-empty question is required."}
    assert trace_store.records() == []


def test_duplicate_provider_registration_is_deterministic() -> None:
    from orchestrator.capabilities import Provider

    provider = Provider(
        name="duplicate-probe",
        version="0.0.1",
        capabilities=frozenset({"single_image_vqa"}),
        model_name="qwen2.5vl-3b",
    )
    with pytest.raises(ValueError, match="already registered to another provider"):
        capabilities.register_provider(provider)
    resolved = capabilities.resolve_provider(capabilities.SINGLE_IMAGE_VQA)
    assert resolved.provider_name == "qwen2.5vl-3b"


def test_registry_inspection_does_not_expose_mutable_state() -> None:
    status = capabilities.capabilities_status()
    status.append({"name": "forged", "available": True, "provider": "x"})
    again = capabilities.capabilities_status()
    assert [entry["name"] for entry in again] == list(capabilities.KNOWN_CAPABILITIES)
    assert all(entry["name"] != "forged" for entry in again)


def test_plan_endpoint_writes_no_trace_and_invokes_no_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(AssertionError("model must not run")),
    )
    for question in (
        "Is there a building in this image?",
        "Where is the building?",
        "Where did flooding increase between these two scenes?",
    ):
        response = client.post(
            "/api/plan", json={"scene_id": "scene_a", "question": question}
        )
        assert response.status_code == 200
    assert trace_store.records() == []


def test_multi_step_plan_analyze_invokes_no_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("Grounding DINO inference requires a CUDA GPU")
        ),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": "Where did flooding increase?",
        },
    )
    # Structural inputs are checked first: the combined change→grounding plan
    # requires a scene pair the current API cannot supply, so it fails as a
    # client error before any availability check or provider invocation.
    assert response.status_code == 422
    assert response.json() == {"detail": "This request requires two scenes."}
    assert trace_store.records() == []


def test_multi_step_request_on_golden_scene_cannot_use_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        services,
        "route",
        lambda **_: (_ for _ in ()).throw(
            RuntimeError("Grounding DINO inference requires a CUDA GPU")
        ),
    )
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": "Where did flooding increase?",
        },
    )
    assert response.status_code == 422
    assert "showing the exact committed result" not in response.text
    assert trace_store.records() == []

    # A structurally complete request is impossible for the combined plan on
    # the current API; a single-scene grounding plan must also never see cache.
    grounding = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "capability": "grounding",
        },
    )
    assert grounding.status_code == 503
    assert "showing the exact committed result" not in grounding.text
    assert trace_store.records() == []


def test_trace_records_execution_step_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={"scene_id": created["scene_id"], "question": "Is water visible?"},
    )

    assert response.status_code == 200
    records = trace_store.records()
    assert len(records) == 1
    params = records[0]["params"]
    assert params["execution_plan_version"] == "phase0-plan-v1"
    assert params["execution_step_id"] == "step_1"
    assert params["execution_step_index"] == 1
    assert params["execution_step_count"] == 1
    assert params["capability"] == "single_image_vqa"
    assert trace_store.verify_chain()[0] is True


def test_caller_cannot_forge_execution_step_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    model = SimpleNamespace(
        version="test", infer=lambda **_: {"answer": "Yes", "evidence": []}
    )
    monkeypatch.setattr(model_router, "get", lambda _: model)

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": created["scene_id"],
            "question": "Is water visible?",
            "execution_step_id": "forged_step",
            "execution_plan_version": "forged-plan",
        },
    )

    assert response.status_code == 200
    params = response.json()["trace"]["params"]
    assert params["execution_plan_version"] == "phase0-plan-v1"
    assert params["execution_step_id"] == "step_1"
    assert params["execution_step_index"] == 1
    assert params["execution_step_count"] == 1


def test_change_analysis_and_traces_never_expose_absolute_server_paths(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Treat the test sandbox as the repository root so runtime rasters live "in" the repo.
    monkeypatch.setattr(model_paths, "REPO_ROOT", tmp_path)
    scene_ids = [
        upload(
            client,
            f"{name}.tif",
            tiff_bytes(bands=3, dtype="float32"),
            {
                "modality": "optical",
                "sensor": "test-rgb",
                "acquisition_timestamp": timestamp,
                "pair_group": "change-pair-leak",
            },
        ).json()["scene_id"]
        for name, timestamp in (
            ("before", "2026-01-01T00:00:00Z"),
            ("after", "2026-01-02T00:00:00Z"),
        )
    ]

    response = client.post(
        "/api/analyze",
        json={
            "scene_id": scene_ids[0],
            "scene_id_2": scene_ids[1],
            "question": "Did built-up area increase?",
            "capability": "change_vqa",
        },
    )
    traces = client.get("/api/traces")

    assert response.status_code == 200, response.text
    assert traces.status_code == 200
    expected = [f"rasters/{scene_id}.tif" for scene_id in scene_ids]
    inputs = response.json()["evidence"][0]
    assert inputs["type"] == "temporal_inputs"
    assert [inputs["t1"]["path"], inputs["t2"]["path"]] == expected
    assert all(len(inputs[key]["sha256"]) == 64 for key in ("t1", "t2"))
    assert response.json()["trace"]["input_summary"]["image_paths"] == expected
    assert traces.json()["records"][0]["input_summary"]["image_paths"] == expected
    for text in (response.text, traces.text):
        assert str(tmp_path) not in text
        assert str(services.ROOT) not in text
    assert trace_store.verify_chain()[0] is True


def test_unexpected_analyze_exception_logs_error_with_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*_: object) -> dict:
        raise RuntimeError("unexpected-service-failure")

    monkeypatch.setattr(analyze_routes, "analyze_scene", fail)
    question = "Is there a building here? " + "private-detail " * 20

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/api/analyze",
            json={"scene_id": services.GOLDEN_SCENE_ID, "question": question},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Model execution failed."}
    records = [record for record in caplog.records if record.name == analyze_routes.logger.name]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    assert "Traceback" in caplog.text
    assert "unexpected-service-failure" in caplog.text
    message = record.getMessage()
    assert services.GOLDEN_SCENE_ID in message
    assert "Is there a building here?" in message
    assert question.strip() not in caplog.text


@pytest.mark.parametrize(
    ("failure", "status_code", "level"),
    [
        ("tensor shape mismatch", 502, logging.ERROR),
        ("Qwen2.5-VL inference requires a CUDA GPU", 503, logging.WARNING),
    ],
)
def test_model_failures_log_traceback_without_changing_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
    status_code: int,
    level: int,
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()

    def fail(**_: object) -> dict:
        raise RuntimeError(failure)

    monkeypatch.setattr(
        model_router, "get", lambda _: SimpleNamespace(version="test", infer=fail)
    )
    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/api/analyze",
            json={"scene_id": created["scene_id"], "question": "What is visible?"},
        )

    assert response.status_code == status_code
    assert failure not in response.text
    records = [record for record in caplog.records if record.name == services.logger.name]
    assert len(records) == 1
    assert records[0].levelno == level
    assert records[0].exc_info is not None
    assert "single_image_vqa" in records[0].getMessage()
    assert failure in caplog.text
    assert "Traceback" in caplog.text


def test_model_timeout_logs_warning_with_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    created = upload(client, "scene.png", image_bytes("PNG")).json()
    release = Event()
    monkeypatch.setattr(
        model_router,
        "get",
        lambda _: SimpleNamespace(
            version="test", infer=lambda **_: release.wait(2) and {"answer": "late"}
        ),
    )
    monkeypatch.setattr(services, "MODEL_EXECUTION_TIMEOUT_SECONDS", 0.01)

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/api/analyze",
            json={"scene_id": created["scene_id"], "question": "What is visible?"},
        )
    release.set()

    assert response.status_code == 503
    records = [record for record in caplog.records if record.name == services.logger.name]
    assert [record.levelno for record in records] == [logging.WARNING]
    assert records[0].exc_info is not None
    assert "timed out" in records[0].getMessage()


def test_inference_worker_crash_returns_503_and_logs_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from orchestrator.worker import WorkerCrashed

    created = upload(client, "scene.png", image_bytes("PNG")).json()

    def crash(**_: object) -> None:
        raise WorkerCrashed("Inference worker exited with code -9 while running test")

    monkeypatch.setattr(
        model_router, "get", lambda _: SimpleNamespace(version="test", infer=crash)
    )

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/api/analyze",
            json={"scene_id": created["scene_id"], "question": "What is visible?"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Live model inference is unavailable."}
    records = [record for record in caplog.records if record.name == services.logger.name]
    assert [record.levelno for record in records] == [logging.ERROR]
    assert "worker crashed" in records[0].getMessage()


def test_log_excerpt_truncates_and_flattens_user_text() -> None:
    assert services.log_excerpt("short question") == "short question"
    assert services.log_excerpt(None) == ""
    long_text = "line one\n" + "x" * 200
    excerpt = services.log_excerpt(long_text)
    assert "\n" not in excerpt
    assert len(excerpt) == services.LOG_EXCERPT_CHARS + 3
    assert excerpt.endswith("...")
