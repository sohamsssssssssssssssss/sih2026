"""POST /api/sentinel1/pairs against a mocked Sentinel Hub; no network, no real credentials."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

import backend.sentinel1 as s1
from backend.main import app
from backend.routes import sentinel1 as sentinel1_routes
from backend.test_sentinel1 import PATNA_AOI, FakeSentinelHub, acquisition, runtime_dirs  # noqa: F401

URL = "/api/sentinel1/pairs"
SECRET = "top-secret"
WAIT_SECONDS = 10
EXPLICIT = {
    "bbox": list(PATNA_AOI),
    "before": {"start": "2024-07-25", "end": "2024-08-02"},
    "after": {"start": "2024-08-15", "end": "2024-08-24"},
}
PATNA_EVENT = {
    "event_id": "patna-2024",
    "aoi": {"bbox": list(PATNA_AOI)},
    "before": EXPLICIT["before"],
    "after": EXPLICIT["after"],
}


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSE_CLIENT_ID", "id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", SECRET)


def use_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(
        sentinel1_routes, "cdse_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )


def no_cdse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"CDSE must not be called: {request.url}")


def flood_hub() -> FakeSentinelHub:
    return FakeSentinelHub(before=[acquisition(1)], after=[acquisition(13)])


def test_event_pair_is_fetched_and_listed_with_the_uploads(monkeypatch, runtime_dirs, credentials) -> None:
    monkeypatch.setattr(s1, "load_events", lambda: [PATNA_EVENT])
    use_transport(monkeypatch, flood_hub())
    client = TestClient(app)

    response = client.post(URL, json={"event_id": "patna-2024"})

    assert response.status_code == 201
    body = response.json()
    assert body["pair_group"] == "patna-2024"
    assert body["before"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240801T001234"
    assert body["after"]["acquisition_id"] == "S1A_IW_GRDH_1SDV_20240813T001234"
    uploads = {scene["scene_id"]: scene for scene in client.get("/api/scenes").json()["uploads"]}
    assert {body["before"]["scene_id"], body["after"]["scene_id"]} <= set(uploads)
    assert uploads[body["after"]["scene_id"]]["modality"] == "sar"
    assert body["optical_check"]["verdict"] == "no_acquisition"  # passed through, not dropped


def test_explicit_pair_searches_inclusive_whole_day_windows(monkeypatch, runtime_dirs, credentials) -> None:
    hub, windows = flood_hub(), []

    def recording(request: httpx.Request) -> httpx.Response:
        if str(request.url) == s1.CATALOG_URL:
            windows.append(json.loads(request.content)["datetime"])
        return hub(request)

    use_transport(monkeypatch, recording)

    response = TestClient(app).post(URL, json={**EXPLICIT, "pair_group": "patna-explicit"})

    assert response.status_code == 201
    assert response.json()["pair_group"] == "patna-explicit"
    assert "2024-07-25T00:00:00Z/2024-08-03T00:00:00Z" in windows
    assert "2024-08-15T00:00:00Z/2024-08-25T00:00:00Z" in windows


@pytest.mark.parametrize(
    "body",
    [
        {**EXPLICIT, "event_id": "kosi-2024"},
        {"event_id": "kosi-2024", "bbox": EXPLICIT["bbox"]},
        {},
        {"pair_group": "kosi"},
        {"bbox": EXPLICIT["bbox"], "before": EXPLICIT["before"]},
        {**EXPLICIT, "bbox": [85.0, 25.6, 85.004]},
        {"event_id": "kosi-2024", "pairgroup": "typo"},
    ],
)
def test_request_must_be_exactly_one_form(body, monkeypatch, credentials) -> None:
    use_transport(monkeypatch, no_cdse)

    assert TestClient(app).post(URL, json=body).status_code == 422


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"event_id": "atlantis-2099"}, "kosi-2024"),
        ({**EXPLICIT, "before": {"start": "2024-07-32", "end": "2024-08-02"}}, "2024-07-32"),
        ({**EXPLICIT, "before": EXPLICIT["after"], "after": EXPLICIT["before"]}, "ordered"),
        ({**EXPLICIT, "bbox": [85.2, 25.6, 85.0, 25.7]}, "west < east"),
        ({**EXPLICIT, "bbox": [85.0, 25.0, 86.0, 26.0]}, "5000"),
    ],
)
def test_invalid_request_values_are_422_before_any_cdse_call(body, message, monkeypatch, credentials) -> None:
    use_transport(monkeypatch, no_cdse)

    response = TestClient(app).post(URL, json=body)

    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_missing_credentials_is_503_without_the_secret(monkeypatch) -> None:
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.setenv("CDSE_CLIENT_SECRET", SECRET)
    use_transport(monkeypatch, no_cdse)

    response = TestClient(app).post(URL, json=EXPLICIT)

    assert response.status_code == 503
    assert "CDSE_CLIENT_ID" in response.json()["detail"]
    assert SECRET not in response.text


def test_upstream_rejection_is_502_and_never_echoes_the_secret(monkeypatch, credentials, caplog) -> None:
    # A misbehaving upstream that echoes the submitted token form back in its error.
    use_transport(monkeypatch, lambda request: httpx.Response(401, text=request.content.decode()))

    with caplog.at_level(logging.DEBUG):
        response = TestClient(app).post(URL, json=EXPLICIT)

    assert response.status_code == 502
    assert "401" in response.json()["detail"]
    assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_second_fetch_while_one_runs_is_409(monkeypatch, runtime_dirs, credentials) -> None:
    hub, started, release = flood_hub(), threading.Event(), threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        started.set()
        assert release.wait(WAIT_SECONDS)
        return hub(request)

    use_transport(monkeypatch, slow)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(TestClient(app).post, URL, json=EXPLICIT)
        assert started.wait(WAIT_SECONDS)
        second = TestClient(app).post(URL, json=EXPLICIT)
        release.set()

    assert second.status_code == 409
    assert first.result().status_code == 201
