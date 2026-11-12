"""API count routing is exercised with injected CPU worker responses."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend import services
from models.base import ModelReadiness
from models.grounding_dino import GroundingDINOModel
from orchestrator import router, trace


@pytest.mark.parametrize("available", [True, False])
def test_count_api_evidence_or_explicit_fallback(monkeypatch, ready_providers, available):
    monkeypatch.setenv("SATQUERY_VQA_ADAPTER", "/synthetic/adapter")
    monkeypatch.setattr(GroundingDINOModel, "readiness", lambda self: ModelReadiness(
        available, None if available else "CUDA_UNAVAILABLE", None if available else "No CUDA"))
    def infer(model, images, question, timeout):
        if available:
            assert model == "grounding-dino-swint" and question == "buildings"
            return {"answer": "matches", "evidence": [{"type": "bounding_box", "confidence": 0.9,
                "coordinate_space": "normalized_xyxy", "coordinates": [0, 0, 0.5, 0.5]}]}
        assert model == "qwen2.5vl-3b" and question == "How many buildings are there?"
        return {"answer": "8", "evidence": []}
    monkeypatch.setattr(router.inference_worker, "run", infer)
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={"scene_id": services.GOLDEN_SCENE_ID,
            "question": "How many buildings are there?", "execution_mode": "live"})
    assert response.status_code == 200, response.text
    body = response.json()
    if available:
        assert body["answer"] == "1" and body["count"] == 1
        assert body["count_status"] == "detected"
        assert body["count_thresholds"]["nms_iou"] == 0.5
        assert body["evidence"][0]["type"] == "bounding_box"
        assert body["evidence"][-1]["thresholds"] == body["count_thresholds"]
        assert body["trace"]["params"]["count_thresholds"] == body["count_thresholds"]
    else:
        assert body["answer"] == "8"
        assert body["unverified_count"] == "detector_unavailable"
        assert body["count_status"] == "detector_unavailable"
        assert "unverified_count: detector_unavailable" in body["notice"]
        assert body["model"]["name"] == "qwen2.5vl-3b"
        assert body["trace"]["params"]["unverified_count"] == "detector_unavailable"
        assert body["evidence"] == []
    assert trace.verify_chain()[0]
