"""CORS origins: localhost defaults plus exact deployment origins from the environment."""

import importlib

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.main import CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS, cors_origins

DEPLOYED = "https://satquery.example.org"


def preflight(client: TestClient, origin: str):
    return client.options(
        "/api/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )


@pytest.fixture
def reloaded_main(monkeypatch: pytest.MonkeyPatch):
    """Rebuild the app under a given environment, then restore the default app."""

    def build(value: str | None):
        if value is None:
            monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)
        else:
            monkeypatch.setenv(CORS_ORIGINS_ENV, value)
        return importlib.reload(main)

    yield build
    monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)
    importlib.reload(main)


@pytest.mark.parametrize("raw", [None, "", " , ,"])
def test_unset_or_blank_keeps_only_localhost_defaults(raw):
    assert cors_origins(raw) == list(DEFAULT_CORS_ORIGINS)


def test_configured_origins_are_appended_after_defaults_without_duplicates():
    raw = f" {DEPLOYED} ,http://203.0.113.7:8080,{DEPLOYED}/,http://localhost:3000"
    assert cors_origins(raw) == [
        *DEFAULT_CORS_ORIGINS,
        DEPLOYED,
        "http://203.0.113.7:8080",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "*",
        "https://*.example.org",
        "satquery.example.org",
        "ftp://satquery.example.org",
        "https://satquery.example.org/app",
        "https://user@satquery.example.org",
        "https://satquery.example.org?x=1",
    ],
)
def test_wildcards_and_non_origins_are_rejected(bad):
    with pytest.raises(ValueError, match=CORS_ORIGINS_ENV):
        cors_origins(f"{DEPLOYED},{bad}")


def test_default_app_allows_localhost_and_refuses_unknown_origin(reloaded_main):
    client = TestClient(reloaded_main(None).app)
    allowed = preflight(client, "http://localhost:3000")
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    # Any localhost port keeps working for local development.
    assert preflight(client, "http://127.0.0.1:3001").status_code == 200

    refused = preflight(client, DEPLOYED)
    assert refused.status_code == 400
    assert "access-control-allow-origin" not in refused.headers


def test_env_origin_is_allowed_exactly_and_never_as_wildcard(reloaded_main):
    client = TestClient(reloaded_main(DEPLOYED).app)
    allowed = preflight(client, DEPLOYED)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == DEPLOYED

    simple = client.get("/api/health", headers={"Origin": DEPLOYED})
    assert simple.headers["access-control-allow-origin"] == DEPLOYED

    for other in ("https://evil.example.org", "https://satquery.example.org.evil.test", "http://satquery.example.org"):
        response = client.get("/api/health", headers={"Origin": other})
        assert "access-control-allow-origin" not in response.headers
    # Localhost defaults survive alongside the configured origin.
    assert preflight(client, "http://localhost:3000").status_code == 200


def test_malformed_env_fails_app_startup(reloaded_main):
    with pytest.raises(ValueError, match=CORS_ORIGINS_ENV):
        reloaded_main("*")
