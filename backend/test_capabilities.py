import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def test_capabilities_serves_committed_evidence():
    path = Path(__file__).resolve().parents[1] / "docs/evidence/capabilities.json"
    expected = json.loads(path.read_text())
    response = TestClient(app).get("/capabilities")
    assert response.status_code == 200
    assert response.json()["evidence"] == expected
    assert {"capabilities", "tools"} <= response.json().keys()
    assert len({row["name"] for row in expected}) == len(expected) == 10
    assert {row["status"] for row in expected} == {"live", "experimental", "off", "in-progress"}
