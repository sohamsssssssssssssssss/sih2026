import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.services as services
import orchestrator.trace as trace_store
from backend.main import app


@pytest.fixture(autouse=True)
def isolated_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    monkeypatch.setattr(trace_store, "TRACE_PATH", tmp_path / "trace.jsonl")
    yield
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
