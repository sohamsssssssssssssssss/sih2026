import json
import re
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.services as services
import orchestrator.trace as trace_store
from backend.main import app
from backend.routes import analyze as analyze_routes


@pytest.fixture(autouse=True)
def isolated_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    monkeypatch.setattr(trace_store, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(services, "INGESTED_SCENE_DIR", tmp_path / "scenes")
    yield
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def image_bytes(format_name: str, size: tuple[int, int] = (4, 3)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(output, format=format_name)
    return output.getvalue()


def upload(client: TestClient, filename: str, data: bytes) -> object:
    return client.post(
        "/api/scenes",
        files={"file": (filename, data, "application/octet-stream")},
    )


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ready", "mode": "offline-first"}


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


def test_golden_analysis_falls_back_to_exact_cache(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_gpu(**_: object) -> dict:
        raise RuntimeError("Qwen2.5-VL inference requires a CUDA GPU")

    monkeypatch.setattr(services, "route", no_gpu)
    response = client.post(
        "/api/analyze",
        json={
            "scene_id": services.GOLDEN_SCENE_ID,
            "question": services.GOLDEN_QUESTION,
            "sensor": "LoveDA",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Yes"
    assert payload["execution_mode"] == "cached_result"
    assert payload["trace"]["params"]["results_artifact"] == services.RESULTS_RELATIVE_PATH


def test_unmatched_query_never_fabricates(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "route", lambda **_: (_ for _ in ()).throw(RuntimeError("no GPU")))
    response = client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": "Invent an answer", "sensor": "LoveDA"},
    )
    assert response.status_code == 422
    assert "No answer was generated" in response.json()["detail"]


def test_trace_history_and_verification(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services, "route", lambda **_: (_ for _ in ()).throw(RuntimeError("no GPU")))
    client.post(
        "/api/analyze",
        json={"scene_id": services.GOLDEN_SCENE_ID, "question": services.GOLDEN_QUESTION, "sensor": "LoveDA"},
    )
    history = client.get("/api/traces").json()
    assert history["count"] == 1
    verification = client.post("/api/traces/verify").json()
    assert verification == {"verified": True, "message": "Chain verified (1 records)"}


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
    assert response.json()["answer"] == "uploaded scene analyzed"
    assert routed["image_paths"] == [
        str(services.INGESTED_SCENE_DIR / f"{created['scene_id']}.png")
    ]
    assert routed["params"] == {
        "scene_id": created["scene_id"],
        "sensor": None,
        "execution_mode": "live",
    }


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

    assert response.status_code == 422
    assert "no exact committed result matches" in response.json()["detail"]
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
