"""GET /api/scenes upload listing and GET /api/scenes/uploads/{scene_id}."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import backend.services as services
import orchestrator.trace as trace_store
from backend.main import app
from backend.scene_pack import scene_catalog

pytestmark = pytest.mark.usefixtures("ready_providers")

UPLOAD_KEYS = {
    "scene_id",
    "filename",
    "format",
    "width",
    "height",
    "modality",
    "sensor",
    "acquisition_time",
    "has_native_raster",
    "georeferenced",
    "uploaded_at",
}


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    monkeypatch.setattr(trace_store, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(services, "INGESTED_SCENE_DIR", tmp_path / "scenes")
    monkeypatch.setattr(services, "INGESTED_RASTER_DIR", tmp_path / "rasters")
    monkeypatch.setattr(services, "SCENE_MANIFEST_DIR", tmp_path / "manifests")
    yield
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def png_bytes(size: tuple[int, int] = (5, 3)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(output, format="PNG")
    return output.getvalue()


def tiff_bytes(*, georeferenced: bool = True, size: tuple[int, int] = (6, 4)) -> bytes:
    width, height = size
    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 2,
        "dtype": "uint16",
        "nodata": 0,
    }
    if georeferenced:
        profile.update(crs="EPSG:32643", transform=from_origin(500000, 2000000, 10, 10))
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            for band in (1, 2):
                values = np.arange(width * height).reshape(height, width) + band
                dataset.write(values.astype("uint16"), band)
        return memory.read()


def upload(client: TestClient, filename: str, data: bytes, **metadata: str) -> dict:
    response = client.post(
        "/api/scenes",
        files={"file": (filename, data, "application/octet-stream")},
        data=metadata,
    )
    assert response.status_code == 201, response.text
    return response.json()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    return parsed


def test_listing_is_empty_without_uploads_and_curated_scenes_unchanged(
    client: TestClient,
) -> None:
    response = client.get("/api/scenes")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"version", "scenes", "uploads"}
    assert body["uploads"] == []
    curated = scene_catalog()
    assert body["version"] == curated["version"]
    assert body["scenes"] == curated["scenes"]


def test_uploads_are_listed_newest_first_with_contract_fields(client: TestClient) -> None:
    png = upload(client, "field.png", png_bytes((5, 3)), modality="optical", sensor="PlanetScope")
    tif = upload(
        client,
        "../geo.tif",
        tiff_bytes(size=(6, 4)),
        modality="multispectral",
        sensor="S2",
        acquisition_timestamp="2026-01-02T03:04:05Z",
    )

    body = client.get("/api/scenes").json()
    uploads = body["uploads"]
    assert [item["scene_id"] for item in uploads] == [tif["scene_id"], png["scene_id"]]
    assert all(set(item) == UPLOAD_KEYS for item in uploads)
    assert all(re.fullmatch(r"scene_[0-9a-f]{32}", item["scene_id"]) for item in uploads)
    assert parse_utc(uploads[0]["uploaded_at"]) >= parse_utc(uploads[1]["uploaded_at"])
    assert body["scenes"] == scene_catalog()["scenes"]

    newest, oldest = uploads
    assert newest["filename"] == "geo.tif"
    assert newest["format"] == "TIFF"
    assert (newest["width"], newest["height"]) == (6, 4)
    assert newest["modality"] == "multispectral"
    assert newest["sensor"] == "S2"
    assert newest["acquisition_time"] == "2026-01-02T03:04:05+00:00"
    assert newest["has_native_raster"] is True
    assert newest["georeferenced"] is True

    assert oldest == {
        "scene_id": png["scene_id"],
        "filename": "field.png",
        "format": "PNG",
        "width": 5,
        "height": 3,
        "modality": "optical",
        "sensor": "PlanetScope",
        "acquisition_time": None,
        "has_native_raster": False,
        "georeferenced": False,
        "uploaded_at": oldest["uploaded_at"],
    }

    manifest = json.loads(
        (services.SCENE_MANIFEST_DIR / f"{png['scene_id']}.json").read_text(encoding="utf-8")
    )
    assert parse_utc(manifest["uploaded_at"]) == parse_utc(oldest["uploaded_at"])


def test_ungeoreferenced_tiff_is_not_marked_georeferenced(client: TestClient) -> None:
    created = upload(client, "plain.tif", tiff_bytes(georeferenced=False))

    scene = client.get(f"/api/scenes/uploads/{created['scene_id']}").json()

    assert scene["format"] == "TIFF"
    assert scene["has_native_raster"] is True
    assert scene["georeferenced"] is False
    assert scene["modality"] == "unknown"
    assert scene["sensor"] is None


def test_native_raster_size_wins_over_downsampled_preview(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "MAX_PREVIEW_DIMENSION", 3)
    created = upload(client, "big.tif", tiff_bytes(size=(6, 4)))

    scene = client.get(f"/api/scenes/uploads/{created['scene_id']}").json()

    assert (scene["width"], scene["height"]) == (6, 4)


def test_single_upload_endpoint_matches_listing_entry(client: TestClient) -> None:
    created = upload(client, "one.png", png_bytes())

    response = client.get(f"/api/scenes/uploads/{created['scene_id']}")

    assert response.status_code == 200
    assert response.json() == client.get("/api/scenes").json()["uploads"][0]


@pytest.mark.parametrize(
    "scene_id",
    [
        "scene_" + "0" * 32,  # well-formed but unknown
        "scene_123",
        "SCENE_" + "a" * 32,
        "loveda-golden-vqa",
        "..%2Fmanifests",
    ],
)
def test_single_upload_endpoint_404s_for_unknown_or_malformed_ids(
    client: TestClient, scene_id: str
) -> None:
    upload(client, "present.png", png_bytes())

    response = client.get(f"/api/scenes/uploads/{scene_id}")

    assert response.status_code == 404


def test_corrupt_or_orphaned_manifests_are_skipped_with_a_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    good = upload(client, "good.png", png_bytes())
    orphan = upload(client, "orphan.png", png_bytes())
    (services.INGESTED_SCENE_DIR / f"{orphan['scene_id']}.png").unlink()
    corrupt_id = "scene_" + "c" * 32
    (services.SCENE_MANIFEST_DIR / f"{corrupt_id}.json").write_text("{not json", encoding="utf-8")
    invalid_id = "scene_" + "d" * 32
    (services.SCENE_MANIFEST_DIR / f"{invalid_id}.json").write_text(
        json.dumps({"version": "1.0", "scene_id": invalid_id}), encoding="utf-8"
    )
    (services.SCENE_MANIFEST_DIR / "notes.json").write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="backend.services"):
        response = client.get("/api/scenes")

    assert response.status_code == 200
    assert [item["scene_id"] for item in response.json()["uploads"]] == [good["scene_id"]]
    warned = " ".join(record.getMessage() for record in caplog.records)
    for skipped in (orphan["scene_id"], corrupt_id, invalid_id):
        assert skipped in warned
    for skipped in (orphan["scene_id"], corrupt_id, invalid_id):
        assert client.get(f"/api/scenes/uploads/{skipped}").status_code == 404


def test_legacy_manifest_without_uploaded_at_uses_file_mtime(client: TestClient) -> None:
    created = upload(client, "legacy.png", png_bytes())
    path = services.SCENE_MANIFEST_DIR / f"{created['scene_id']}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["uploaded_at"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    stamp = datetime(2025, 5, 6, 7, 8, 9, tzinfo=timezone.utc).timestamp()
    os.utime(path, (stamp, stamp))

    scene = client.get(f"/api/scenes/uploads/{created['scene_id']}").json()

    assert parse_utc(scene["uploaded_at"]) == datetime(2025, 5, 6, 7, 8, 9, tzinfo=timezone.utc)


def test_listing_is_capped_to_the_most_recent_uploads(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = [upload(client, f"n{index}.png", png_bytes()) for index in range(4)]
    monkeypatch.setattr(services, "MAX_LISTED_UPLOADS", 2)

    uploads = client.get("/api/scenes").json()["uploads"]

    assert [item["scene_id"] for item in uploads] == [
        created[3]["scene_id"],
        created[2]["scene_id"],
    ]


def test_missing_manifest_directory_lists_no_uploads(client: TestClient) -> None:
    assert not services.SCENE_MANIFEST_DIR.exists()

    assert client.get("/api/scenes").json()["uploads"] == []
