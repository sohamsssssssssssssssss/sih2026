"""Offline probe contract tests: mock HTTP only, no credentials or stored scenes."""

import json
import httpx
import pytest
from shapely.geometry import box, mapping

from scripts import probe_s1_provenance as probe
from backend import sentinel1 as s1
from backend.test_sentinel1 import geotiff, tar_response, userdata_json


def catalog_feature(scene="before"):
    event = next(event for event in json.loads(s1.FLOOD_RUNS_PATH.read_text())["events"]
                 if event["event_id"] == "kosi-2024")
    return {"id": event[scene]["product_id"], "collection": s1.COLLECTION,
            "geometry": mapping(box(86, 25, 87, 27)), "properties": {
                "datetime": event[scene]["acquired_at"], "sat:orbit_state": "ascending",
                "sat:relative_orbit": 85, "sar:instrument_mode": "IW",
                "s1:polarization": "DV", "s1:resolution": "HIGH",
            }}


def transport(feature, calls, *, wrong_source=False, status=200, scene="before"):
    def handler(request):
        calls.append(str(request.url))
        if str(request.url) == s1.TOKEN_URL:
            return httpx.Response(200, json={"access_token": "offline-access-value"})
        if str(request.url) == s1.CATALOG_URL:
            body = json.loads(request.content)
            assert body["ids"] == [s1.load_expected_products("kosi-2024")[scene == "after"]]
            return httpx.Response(200, json={"features": [feature]})
        assert str(request.url) == s1.PROCESS_URL
        assert request.headers["Accept"] == "application/x-tar"
        body = json.loads(request.content)
        assert body["input"]["data"][0]["processing"]["backCoeff"] == "GAMMA0_TERRAIN"
        if status != 200:
            return httpx.Response(status, content=b"provider error")
        bounds = body["input"]["bounds"]
        raster = geotiff(body["output"]["width"], body["output"]["height"], bounds["bbox"],
                         int(bounds["properties"]["crs"].rsplit("/", 1)[1]), 5)
        product_id = "S1A_WRONG" if wrong_source else feature["id"]
        return httpx.Response(200, headers={"Content-Type": "application/x-tar"},
                              content=tar_response(raster, userdata_json(product_id)))
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("bad", ["id", "time", "orbit", "mode", "footprint", "missing_geometry"])
def test_preflight_rejects_before_process(bad):
    feature = catalog_feature()
    if bad == "id":
        feature["id"] = "S1A_WRONG"
    elif bad == "time":
        feature["properties"]["datetime"] = "2024-09-25T12:21:46Z"
    elif bad == "orbit":
        feature["properties"]["sat:relative_orbit"] = 12
    elif bad == "mode":
        feature["properties"]["sar:instrument_mode"] = "EW"
    elif bad == "footprint":
        feature["geometry"] = mapping(box(0, 0, 1, 1))
    else:
        feature.pop("geometry")
    calls = []
    with httpx.Client(transport=transport(feature, calls)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["verdict"] == "NO-GO"
    assert report["process_requests"] == 0
    assert s1.PROCESS_URL not in calls
    stage = "footprint" if bad in {"footprint", "missing_geometry"} else "product selection"
    assert report["failure_stage"] == stage
    assert report["http_status"] == 200
    assert report["retryable"] is False


def test_missing_credentials_sends_no_requests():
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("HTTP forbidden"))) as client:
        report = probe.probe(client, None, None)
    assert report["process_requests"] == 0
    assert report["failure_stage"] == "credentials"
    assert report["http_status"] is None
    assert report["error_category"] == "missing_credentials"
    assert report["retryable"] is False


@pytest.mark.parametrize("wrong_source,status", [(False, 200), (True, 200), (False, 500)])
def test_one_process_request_and_no_ingestion(monkeypatch, wrong_source, status):
    import backend.optical_clouds as optical

    def forbidden(*args, **kwargs):
        pytest.fail("Ingestion, pair acquisition, and optical processing are forbidden")

    monkeypatch.setattr(s1, "ingest_scene", forbidden)
    monkeypatch.setattr(s1, "_ingest", forbidden)
    monkeypatch.setattr(s1, "fetch_pair", forbidden)
    monkeypatch.setattr(optical, "check_optical_clouds", forbidden)
    calls = []
    with httpx.Client(transport=transport(catalog_feature(), calls, wrong_source=wrong_source, status=status)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert calls.count(s1.PROCESS_URL) == report["process_requests"] == 1
    assert report["http_status"] == status
    encoded = json.dumps(report)
    assert all(secret not in encoded for secret in
               ("offline-client-value", "offline-secret-value", "offline-access-value"))
    if wrong_source or status != 200:
        assert report["verdict"] == "NO-GO"
        assert "raster" not in report
    else:
        assert report["verdict"] == "GO"
        assert report["raster"]["bands"] == 3
        assert report["raster"]["crs"] == "EPSG:32645"
        assert report["request_grid"]["resolution_m"] == 10
        assert len(report["response_sha256"]) == 64
        assert set(report["member_sha256"]) == {"default.tif", "userdata.json"}
        assert "service_metadata" not in report


@pytest.mark.parametrize("endpoint,stage", [(s1.TOKEN_URL, "OAuth"), (s1.CATALOG_URL, "catalogue"),
                                           (s1.PROCESS_URL, "Process")])
@pytest.mark.parametrize("status,category,retryable", [
    (400, "bad_request", False), (401, "authentication_rejected", False),
    (403, "access_denied", False), (429, "rate_limited", True), (503, "http_error", True),
])
def test_safe_http_diagnostics(endpoint, stage, status, category, retryable):
    calls = []
    base = transport(catalog_feature(), calls)
    sensitive = "offline-client-value offline-secret-value offline-access-value Authorization provider-private"

    def handler(request):
        if str(request.url) == endpoint:
            return httpx.Response(status, text=sensitive, headers={"Content-Type": sensitive})
        return base.handle_request(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
        assert client.event_hooks == {"request": [], "response": []}
    assert report["failure_stage"] == stage
    assert report["http_status"] == status
    assert report["error_category"] == category
    assert report["retryable"] is retryable
    assert report["process_requests"] == (stage == "Process")
    assert not any(word in json.dumps(report) for word in sensitive.split())


@pytest.mark.parametrize("endpoint,stage", [(s1.TOKEN_URL, "OAuth"), (s1.CATALOG_URL, "catalogue")])
@pytest.mark.parametrize("kind,category", [(httpx.ReadTimeout, "timeout"),
                                          (httpx.ConnectError, "transport_error")])
def test_safe_transport_diagnostics(endpoint, stage, kind, category):
    base = transport(catalog_feature(), [])

    def handler(request):
        if str(request.url) == endpoint:
            raise kind("Authorization provider-private offline-secret-value", request=request)
        return base.handle_request(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert (report["failure_stage"], report["http_status"], report["error_category"], report["retryable"]) == (
        stage, None, category, True)
    assert report["process_requests"] == 0
    assert "provider-private" not in json.dumps(report)


@pytest.mark.parametrize("payload", [b"provider-private", b"[]", b"{}"])
def test_invalid_oauth_response_is_safe(payload):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload))) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["failure_stage"] == "OAuth"
    assert report["http_status"] == 200
    assert report["error_category"] == "invalid_oauth_response"
    assert report["retryable"] is False
    assert "provider-private" not in json.dumps(report)


@pytest.mark.parametrize("operation,stage", [("load_expected_products", "product selection"),
                                            ("utm_grid", "grid")])
def test_local_preflight_error_never_prints_exception(monkeypatch, operation, stage):
    def fail(*args):
        raise ValueError("provider-private Authorization offline-secret-value")

    monkeypatch.setattr(s1, operation, fail)
    with httpx.Client(transport=transport(catalog_feature(), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["failure_stage"] == stage
    assert report["error_category"] == "local_error"
    assert report["http_status"] is None
    assert report["retryable"] is False
    assert report["process_requests"] == 0
    assert "provider-private" not in json.dumps(report)


@pytest.mark.parametrize("bad,category", [
    ("structure", "tar_structure_invalid"), ("eof", "tar_eof_invalid"),
    ("member", "tar_member_invalid"), ("missing_member", "tar_member_invalid"),
    ("empty_member", "tar_member_invalid"), ("json", "userdata_json_invalid"),
    ("duplicate_json", "userdata_json_invalid"), ("orbit", "source_provenance_invalid"),
    ("tiles", "source_provenance_invalid"), ("source", "source_provenance_invalid"),
    ("decode", "raster_decode_failed"), ("dimensions", "raster_contract_mismatch"),
    ("crs", "raster_contract_mismatch"), ("dtype", "raster_contract_mismatch"),
    ("bands", "raster_contract_mismatch"), ("transform", "raster_contract_mismatch"),
])
def test_process_validation_diagnostics(bad, category, capsys):
    import io
    import tarfile
    from backend.test_sentinel1 import tar_members
    from rasterio.io import MemoryFile

    marker = "offline-secret-value offline-access-value Authorization provider-private"
    feature = catalog_feature()
    base = transport(feature, [])

    def handler(request):
        if str(request.url) != s1.PROCESS_URL:
            return base.handle_request(request)
        body = json.loads(request.content)
        width, height = body["output"]["width"], body["output"]["height"]
        bounds = body["input"]["bounds"]["bbox"]
        if bad == "transform":
            bounds = [value + 10 for value in bounds]
        raster = geotiff(width + (bad == "dimensions"), height, bounds,
                         32644 if bad == "crs" else 32645, 0,
                         dtype="uint16" if bad == "dtype" else "float32")
        if bad == "bands":
            with MemoryFile(raster) as source, source.open() as dataset:
                pixels, profile = dataset.read()[:2], dict(dataset.profile)
            profile["count"] = 2
            with MemoryFile() as target:
                with target.open(**profile) as output:
                    output.write(pixels)
                raster = target.read()
        userdata = userdata_json(feature["id"])
        if bad == "json":
            userdata = marker.encode()
        elif bad == "duplicate_json":
            userdata = b'{"orbits": [], "orbits": []}'
        elif bad == "orbit":
            userdata = json.dumps({"orbits": [], "private": marker}).encode()
        elif bad == "tiles":
            userdata = json.dumps({"orbits": [{"tiles": []}], "private": marker}).encode()
        elif bad == "source":
            userdata = userdata_json("S1A_WRONG")
        elif bad == "decode":
            raster = marker.encode()
        members = [("default.tif", raster), ("userdata.json", userdata)]
        if bad == "member":
            members.append((marker, b"private"))
        elif bad == "missing_member":
            members.pop()
        elif bad == "empty_member":
            members[0] = ("default.tif", b"")
        content = tar_members(members)
        if bad == "structure":
            content = marker.encode()
        elif bad == "eof":
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
                list(archive)
                content = content[:archive.offset] + b"\0" * 512
        return httpx.Response(200, content=content, headers={"Content-Type": "application/x-tar"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["verdict"] == "NO-GO"
    assert report["process_requests"] == 1
    assert report["http_status"] == 200
    assert report["failure_stage"] == "Process"
    assert report["validation_stage"] == probe.VALIDATION_STAGES[category]
    assert report["error_category"] == category
    assert report["retryable"] is False
    assert len(report["response_sha256"]) == 64
    encoded = json.dumps(report) + str(capsys.readouterr())
    assert not any(word in encoded for word in marker.split())
    assert "raster" not in report


@pytest.mark.parametrize("code", ["tar_eof_invalid", "Authorization provider-private"])
def test_exception_diagnostic_allowlist(monkeypatch, code):
    def fail(*args, **kwargs):
        raise s1.CDSEError("offline-secret-value Authorization provider-private",
                           diagnostic_category=code)

    monkeypatch.setattr(s1, "_parse_tar_response", fail)
    with httpx.Client(transport=transport(catalog_feature(), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["error_category"] == (code if code in probe.VALIDATION_STAGES else "process_validation_failed")
    assert "provider-private" not in json.dumps(report)
    assert "offline-secret-value" not in json.dumps(report)


@pytest.fixture
def capture_home(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.Path, "home", lambda: tmp_path)
    return tmp_path / ".local/share/satquery/probe-captures"


def test_capture_disabled_by_default(capture_home):
    with httpx.Client(transport=transport(catalog_feature(), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["verdict"] == "GO"
    assert "capture" not in report
    assert not capture_home.exists()


@pytest.mark.parametrize("failure", [None, "tar", "provenance", "raster"])
def test_opt_in_capture_precedes_validation(capture_home, monkeypatch, failure, capsys):
    import hashlib
    import stat

    base = transport(catalog_feature(), [], wrong_source=failure == "provenance")
    bodies = []

    def handler(request):
        response = base.handle_request(request)
        if str(request.url) == s1.PROCESS_URL:
            if failure == "tar":
                response = httpx.Response(200, content=b"invalid archive", headers={"Content-Type": "application/x-tar"})
            bodies.append(response.content)
        return response

    parser = s1._parse_tar_response

    def check_saved(content, **kwargs):
        files = list(capture_home.iterdir())
        assert len(files) == 1
        assert files[0].read_bytes() == content
        return parser(content, **kwargs)

    monkeypatch.setattr(s1, "_parse_tar_response", check_saved)
    if failure == "raster":
        def invalid_raster(*args):
            raise s1.CDSEError("offline-secret-value Authorization provider-private",
                               diagnostic_category="raster_decode_failed")
        monkeypatch.setattr(s1, "_on_grid", invalid_raster)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", capture=True)
    assert report["verdict"] == ("GO" if failure is None else "NO-GO")
    saved = probe.Path(report["capture"]["path"])
    assert saved.parent == capture_home
    assert saved.read_bytes() == bodies[0]
    assert report["capture"] == {"path": str(saved), "byte_count": len(bodies[0]),
                                  "sha256": hashlib.sha256(bodies[0]).hexdigest()}
    assert stat.S_IMODE(capture_home.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    encoded = json.dumps(report) + str(capsys.readouterr())
    for private in ("offline-client-value", "offline-secret-value", "offline-access-value",
                    "Authorization", "provider-private", "invalid archive"):
        assert private not in encoded
        if private != "invalid archive":
            assert private.encode() not in saved.read_bytes()


def test_capture_size_limit_before_directory_creation(capture_home, monkeypatch):
    monkeypatch.setattr(s1, "MAX_PROCESS_BYTES", 4)
    with pytest.raises(s1.CDSEError):
        probe.capture_response(b"12345")
    assert not capture_home.exists()
    result = probe.capture_response(b"1234")
    assert probe.Path(result["path"]).read_bytes() == b"1234"


def test_capture_never_overwrites_and_tightens_directory(capture_home, monkeypatch):
    import stat
    import uuid

    capture_home.mkdir(parents=True, mode=0o755)
    capture_home.chmod(0o755)
    first = probe.capture_response(b"first")
    second = probe.capture_response(b"second")
    assert first["path"] != second["path"]
    assert stat.S_IMODE(capture_home.stat().st_mode) == 0o700
    fixed = uuid.UUID(int=0)
    monkeypatch.setattr(probe.uuid, "uuid4", lambda: fixed)
    existing = capture_home / f"process-{fixed.hex}.tar"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        probe.capture_response(b"replacement")
    assert existing.read_bytes() == b"keep"
    assert probe.Path(first["path"]).read_bytes() == b"first"


@pytest.mark.parametrize("status,oversized", [(500, False), (200, True)])
def test_http_errors_and_oversized_responses_not_captured(capture_home, monkeypatch, status, oversized):
    base = transport(catalog_feature(), [])
    if oversized:
        monkeypatch.setattr(s1, "MAX_PROCESS_BYTES", 4)

    def handler(request):
        if str(request.url) == s1.PROCESS_URL:
            return httpx.Response(status, content=b"provider-private offline-secret-value",
                                  headers={"Content-Type": "application/x-tar"})
        return base.handle_request(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", capture=True)
    assert report["verdict"] == "NO-GO"
    assert "capture" not in report
    assert not capture_home.exists()
    assert "provider-private" not in json.dumps(report)


def test_capture_refuses_symlink_directory(capture_home, tmp_path):
    target = tmp_path / "other"
    target.mkdir()
    capture_home.parent.mkdir(parents=True)
    capture_home.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        probe.capture_response(b"body")
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("arguments,enabled", [([], False), (["--capture-response"], True)])
def test_capture_cli_flag(monkeypatch, arguments, enabled):
    monkeypatch.setattr(probe.os, "environ", {})

    def fake_probe(client, client_id, client_secret, *, capture=False, scene="before", full_grid=False):
        assert client_id is None and client_secret is None
        assert capture is enabled
        assert scene == "before"
        return {"verdict": "GO"}

    monkeypatch.setattr(probe, "probe", fake_probe)
    assert probe.main(arguments) == 0


def test_capture_write_failure_is_safe_and_stops_before_validation(monkeypatch):
    def fail(content):
        raise OSError("offline-secret-value Authorization provider-private")

    monkeypatch.setattr(probe, "capture_response", fail)
    monkeypatch.setattr(s1, "_parse_tar_response", lambda *args, **kwargs: pytest.fail("Validation must follow capture"))
    with httpx.Client(transport=transport(catalog_feature(), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", capture=True)
    assert report["verdict"] == "NO-GO"
    assert report["error_category"] == "response_capture_failed"
    assert report["validation_stage"] == "response capture"
    assert "capture" not in report
    assert "provider-private" not in json.dumps(report)


def test_probe_reports_missing_eof_compatibility():
    import io
    import tarfile

    base = transport(catalog_feature(), [])

    def handler(request):
        response = base.handle_request(request)
        if str(request.url) == s1.PROCESS_URL:
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as archive:
                list(archive)
                content = response.content[:archive.offset]
            return httpx.Response(200, content=content, headers={"Content-Type": "application/x-tar"})
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    assert report["verdict"] == "GO"
    assert report["tar_eof_mode"] == "missing_accepted"
    assert report["provenance_validation"].startswith("PASS")


@pytest.mark.parametrize("wrong_source", [False, True])
def test_after_full_grid_capture_and_no_ingestion(capture_home, monkeypatch, wrong_source):
    import backend.optical_clouds as optical

    def forbidden(*args, **kwargs):
        pytest.fail("No pair fetching, ingestion, optical, or BEFORE processing")

    for name in ("fetch_pair", "ingest_scene", "_ingest"):
        monkeypatch.setattr(s1, name, forbidden)
    monkeypatch.setattr(optical, "check_optical_clouds", forbidden)
    feature = catalog_feature("after")
    calls, process_bodies = [], []
    base = transport(feature, calls, scene="after", wrong_source=wrong_source)

    def handler(request):
        if str(request.url) == s1.PROCESS_URL:
            process_bodies.append(json.loads(request.content))
        return base.handle_request(request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", scene="after", capture=True)
    assert report["scene"] == "after"
    assert report["expected_product_id"] == feature["id"]
    assert "20241006T122147" in feature["id"]
    assert calls.count(s1.PROCESS_URL) == report["process_requests"] == 1
    body = process_bodies[0]
    assert body["output"]["width"] == 749
    assert body["output"]["height"] == 1649
    assert body["input"]["bounds"]["bbox"] == [432970., 2867120., 440460., 2883610.]
    assert body["input"]["bounds"]["properties"]["crs"].endswith("/32645")
    assert body["input"]["data"][0]["dataFilter"]["timeRange"] == {
        "from": "2024-10-06T12:11:47Z", "to": "2024-10-06T12:31:47Z"}
    saved = probe.Path(report["capture"]["path"])
    assert report["response_size_bytes"] == saved.stat().st_size == report["capture"]["byte_count"]
    assert report["capture"]["sha256"] == report["response_sha256"]
    if wrong_source:
        assert report["verdict"] == "NO-GO"
        assert report["error_category"] == "source_provenance_invalid"
        assert "raster" not in report
    else:
        assert report["verdict"] == "GO"
        assert report["source_product_ids"] == [feature["id"]]
        assert report["provenance_validation"].startswith("PASS")
        assert (report["raster"]["width"], report["raster"]["height"], report["raster"]["crs"]) == (749, 1649, "EPSG:32645")
        assert report["raster"]["valid_pixels"] == report["raster"]["total_pixels"] == 749 * 1649
        assert report["raster"]["valid_fraction"] == 1.0
    assert not any(secret in json.dumps(report) for secret in
                   ("offline-client-value", "offline-secret-value", "offline-access-value"))


@pytest.mark.parametrize("bad", ["before_id", "timestamp", "orbit", "footprint", "grid"])
def test_after_preflight_fails_without_process(monkeypatch, bad):
    feature = catalog_feature("after")
    if bad == "before_id":
        feature["id"] = catalog_feature()["id"]
    elif bad == "timestamp":
        feature["properties"]["datetime"] = "2024-10-05T12:21:47Z"
    elif bad == "orbit":
        feature["properties"]["sat:relative_orbit"] = 86
    elif bad == "footprint":
        # Covers the small default probe, but excludes much of the full grid.
        feature["geometry"] = mapping(box(86.36, 25.99, 86.38, 26.01))
    else:
        original = s1.utm_grid
        monkeypatch.setattr(s1, "utm_grid", lambda bbox: s1.Grid(32645, original(bbox).bounds, 750, 1649))
    calls = []
    with httpx.Client(transport=transport(feature, calls, scene="after")) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", scene="after")
    assert report["verdict"] == "NO-GO"
    assert report["process_requests"] == 0
    assert s1.PROCESS_URL not in calls


def test_default_before_grid_unchanged():
    calls = []
    with httpx.Client(transport=transport(catalog_feature(), calls)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    bbox, _, _ = s1.event_request("kosi-2024")
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    grid = s1.utm_grid((lon - .002, lat - .002, lon + .002, lat + .002))
    assert report["scene"] == "before"
    assert (report["request_grid"]["width"], report["request_grid"]["height"]) == (grid.width, grid.height)
    assert report["expected_product_id"] == catalog_feature()["id"]


def test_valid_pixel_coverage_uses_mask(monkeypatch):
    import numpy as np
    original = s1._on_grid

    def partial_mask(content, grid):
        profile, pixels = original(content, grid)
        pixels[2] = 0
        pixels[2, 0] = 1
        pixels[2, 1, 0] = -1
        pixels[2, 1, 1] = np.nan
        return profile, pixels

    monkeypatch.setattr(s1, "_on_grid", partial_mask)
    with httpx.Client(transport=transport(catalog_feature(), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value")
    raster = report["raster"]
    assert raster["valid_pixels"] == raster["width"]
    assert raster["valid_fraction"] == 1 / raster["height"]


def test_after_cli_option(monkeypatch):
    monkeypatch.setattr(probe.os, "environ", {})

    def fake_probe(client, client_id, client_secret, *, capture, scene, full_grid):
        assert client_id is None and client_secret is None
        assert scene == "after"
        assert full_grid is False
        assert capture is True
        return {"verdict": "GO"}

    monkeypatch.setattr(probe, "probe", fake_probe)
    assert probe.main(["--scene", "after", "--capture-response"]) == 0


def test_invalid_scene_never_sends_request():
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("HTTP forbidden"))) as client:
        report = probe.probe(client, "offline", "offline", scene="invalid")
    assert report["verdict"] == "NO-GO"
    assert report["process_requests"] == 0
    assert report["failure_stage"] == "product selection"


@pytest.mark.parametrize("wrong_source", [False, True])
def test_full_grid_before_capture_and_provenance(capture_home, monkeypatch, wrong_source):
    def forbidden(*args, **kwargs):
        pytest.fail("Ingestion and pair fetching forbidden")

    for name in ("fetch_pair", "_ingest", "ingest_scene"):
        monkeypatch.setattr(s1, name, forbidden)
    with httpx.Client(transport=transport(catalog_feature(), [], wrong_source=wrong_source)) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", full_grid=True, capture=True)
    assert report["scene"] == "before"
    assert report["expected_product_id"] == s1.load_expected_products("kosi-2024")[0]
    assert report["process_requests"] == 1
    assert report["request_grid"] == {"width": 749, "height": 1649, "epsg": 32645,
                                       "bounds": (432970., 2867120., 440460., 2883610.), "resolution_m": 10}
    saved = probe.Path(report["capture"]["path"])
    assert saved.stat().st_size == report["response_size_bytes"]
    assert report["capture"]["sha256"] == report["response_sha256"]
    if wrong_source:
        assert report["verdict"] == "NO-GO"
        assert report["error_category"] == "source_provenance_invalid"
    else:
        assert report["verdict"] == "GO"
        assert report["source_product_ids"] == [catalog_feature()["id"]]
        assert report["raster"]["transform"] == [10., 0., 432970., 0., -10., 2883610., 0., 0., 1.]
        assert report["raster"]["crs"] == "EPSG:32645"


def test_full_grid_before_after_request_consistency():
    requests, reports = [], []
    for scene, full_grid in (("before", True), ("after", False), ("after", True)):
        base = transport(catalog_feature(scene), [], scene=scene)

        def handler(request):
            if str(request.url) == s1.PROCESS_URL:
                requests.append(json.loads(request.content))
            return base.handle_request(request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            reports.append(probe.probe(client, "offline-client-value", "offline-secret-value",
                                       scene=scene, full_grid=full_grid))
    assert all(report["verdict"] == "GO" for report in reports)
    before, after, explicit_after = requests
    assert after == explicit_after
    assert before["input"]["bounds"] == after["input"]["bounds"]
    assert before["output"] == after["output"]
    assert before["evalscript"] == after["evalscript"] == s1.EVALSCRIPT
    before_data, after_data = before["input"]["data"][0], after["input"]["data"][0]
    assert before_data["processing"] == after_data["processing"]
    assert before_data["type"] == after_data["type"]
    for key in ("acquisitionMode", "polarization", "resolution", "orbitDirection"):
        assert before_data["dataFilter"][key] == after_data["dataFilter"][key]
    assert before_data["dataFilter"]["timeRange"] == {
        "from": "2024-09-24T12:11:46Z", "to": "2024-09-24T12:31:46Z"}
    assert reports[0]["raster"]["transform"] == reports[1]["raster"]["transform"]
    assert s1.SAR_BANDS == ("VV", "VH", "dataMask")
    assert 'return [s.VV, s.VH, s.dataMask && !s.shadowMask ? 1 : 0]' in before["evalscript"]


def test_full_grid_before_rejects_after_catalogue_product():
    with httpx.Client(transport=transport(catalog_feature("after"), [])) as client:
        report = probe.probe(client, "offline-client-value", "offline-secret-value", full_grid=True)
    assert report["verdict"] == "NO-GO"
    assert report["failure_stage"] == "product selection"
    assert report["process_requests"] == 0


def test_full_grid_before_cli(monkeypatch):
    monkeypatch.setattr(probe.os, "environ", {})

    def fake_probe(client, client_id, client_secret, *, capture, scene, full_grid):
        assert client_id is None and client_secret is None
        assert scene == "before"
        assert capture is True and full_grid is True
        return {"verdict": "GO"}

    monkeypatch.setattr(probe, "probe", fake_probe)
    assert probe.main(["--full-grid", "--capture-response"]) == 0
