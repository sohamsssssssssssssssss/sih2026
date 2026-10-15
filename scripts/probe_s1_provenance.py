"""One in-memory Kosi Process probe: small BEFORE by default, optional full grid; never ingests imagery."""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape

from backend import sentinel1 as s1


# Fixed labels only: shared validators attach codes without exposing their messages.
VALIDATION_STAGES = {
    "tar_structure_invalid": "TAR structure",
    "tar_eof_invalid": "TAR EOF",
    "tar_member_invalid": "TAR members",
    "userdata_json_invalid": "userdata JSON",
    "source_provenance_invalid": "source-product provenance",
    "raster_decode_failed": "raster decoding",
    "raster_contract_mismatch": "raster dimensions/CRS/bands",
}


def capture_response(content):
    """Persist only the bounded successful Process body, with exclusive private creation."""
    if len(content) > s1.MAX_PROCESS_BYTES:
        raise s1.CDSEError("Capture exceeds Process byte limit")
    directory = Path.home() / ".local" / "share" / "satquery" / "probe-captures"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(directory_fd, 0o700)
        name = f"process-{uuid.uuid4().hex}.tar"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(fd, "wb") as output:
                os.fchmod(output.fileno(), 0o600)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            os.unlink(name, dir_fd=directory_fd)
            raise
    finally:
        os.close(directory_fd)
    return {"path": str(directory / name), "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def preflight(client, token, set_stage=lambda stage, **kwargs: None, *, scene="before", full_grid=False):
    set_stage("product selection")
    event = next(event for event in json.loads(s1.FLOOD_RUNS_PATH.read_text())["events"]
                 if event["event_id"] == "kosi-2024")
    pinned = s1.load_expected_products("kosi-2024")[scene == "after"]
    acquired_at = datetime.fromisoformat(event[scene]["acquired_at"].replace("Z", "+00:00"))
    set_stage("grid")
    bbox, _, _ = s1.event_request("kosi-2024")
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    use_full_grid = full_grid or scene == "after"
    grid = s1.utm_grid(bbox if use_full_grid else
                       (lon - .002, lat - .002, lon + .002, lat + .002))
    if use_full_grid and (
            {"epsg": grid.epsg, "width": grid.width, "height": grid.height} != event["grid"]
            or max(grid.width, grid.height) > s1.MAX_SIDE_PIXELS):
        raise s1.CDSEError("Full event grid must match the pinned grid and fit one Process request")
    set_stage("catalogue")
    page = s1._json(s1._post(client, s1.CATALOG_URL, token=token, json={
        "collections": [s1.COLLECTION], "ids": [pinned], "limit": 2,
    }))
    set_stage("product selection", preserve_status=True)
    features = page.get("features")
    if not isinstance(features, list) or len(features) != 1 or (page.get("context") or {}).get("next"):
        raise s1.CDSEError("Catalogue must return exactly one pinned product")
    feature = features[0]
    acquisition = s1._acquisition(feature)
    properties = feature.get("properties", {})
    if (feature.get("collection") != s1.COLLECTION or acquisition.product_id != pinned
            or acquisition.acquired_at != acquired_at
            or acquisition.orbit_state != event["orbit_direction"]
            or acquisition.relative_orbit != event["relative_orbit"]
            or properties.get("sar:instrument_mode") != "IW"
            or properties.get("s1:polarization") != "DV"
            or properties.get("s1:resolution") != "HIGH"):
        raise s1.CDSEError("Catalogue metadata does not match pinned acquisition")
    set_stage("footprint", preserve_status=True)
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise s1.CDSEError("Catalogue product lacks a polygon footprint")
    footprint = shape(geometry)
    # Check the snapped request grid, not just the original geographic AOI.
    set_stage("grid", preserve_status=True)
    request_bounds = transform_bounds(f"EPSG:{grid.epsg}", "EPSG:4326", *grid.bounds, densify_pts=21)
    set_stage("footprint", preserve_status=True)
    if not footprint.is_valid or footprint.is_empty or not footprint.contains(box(*request_bounds)):
        raise s1.CDSEError("Snapped AOI is not wholly inside the product footprint")
    return acquisition, grid


def probe(client, client_id, client_secret, *, capture=False, scene="before", full_grid=False):
    report = {"verdict": "NO-GO", "process_requests": 0}
    stage = "credentials"
    status = None

    def set_stage(value, *, preserve_status=False):
        nonlocal stage, status
        stage = value
        if not preserve_status:
            status = None

    def record_request(request):
        if str(request.url) == s1.PROCESS_URL:
            if report["process_requests"]:
                raise s1.CDSEError("Second Process request forbidden")
            report["process_requests"] += 1

    def record_response(response):
        nonlocal status
        if str(response.request.url) in {s1.TOKEN_URL, s1.CATALOG_URL, s1.PROCESS_URL}:
            status = response.status_code
        if str(response.request.url) == s1.PROCESS_URL:
            report["http_status"] = response.status_code
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            report["content_type"] = media_type if media_type in {"application/x-tar", "application/tar"} else "other"

    client.event_hooks["request"].append(record_request)
    client.event_hooks["response"].append(record_response)
    try:
        if scene not in {"before", "after"}:
            set_stage("product selection")
            raise s1.CDSEError("Scene must be before or after")
        report["scene"] = scene
        if not client_id or not client_secret:
            raise s1.CDSEError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in this terminal")
        set_stage("OAuth")
        token = s1.access_token(client, client_id, client_secret)
        acquisition, grid = preflight(client, token, set_stage, scene=scene, full_grid=full_grid)
        report["expected_product_id"] = acquisition.product_id
        report["request_grid"] = {"width": grid.width, "height": grid.height,
                                  "epsg": grid.epsg, "bounds": grid.bounds, "resolution_m": s1.PIXEL_SIZE_M}
        set_stage("Process")
        body = s1._process_body(acquisition, grid)
        report["processing"] = body["input"]["data"][0]["processing"]
        content = s1._process_request(client, token, body)
        report["response_size_bytes"] = len(content)
        report["response_sha256"] = hashlib.sha256(content).hexdigest()
        if capture:
            report["validation_stage"] = "response capture"
            report["capture"] = capture_response(content)
        report["validation_stage"] = "TAR structure"
        tiff, userdata_bytes = s1._parse_tar_response(content, allow_missing_eof=True, diagnostics=report)
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            report["tar_members"] = [{"name": member.name, "size": member.size} for member in archive]
        report["member_sha256"] = {"default.tif": hashlib.sha256(tiff).hexdigest(),
                                   "userdata.json": hashlib.sha256(userdata_bytes).hexdigest()}
        report["validation_stage"] = "userdata JSON"
        userdata = s1._parse_userdata(userdata_bytes, acquisition.product_id)
        report["source_product_ids"] = sorted({tile["sentinel1ProductId"]
            for orbit in userdata["orbits"] for tile in orbit["tiles"]})
        report["provenance_validation"] = "PASS: complete source set equals exact pinned ID"
        report["validation_stage"] = "raster decoding"
        profile, pixels = s1._on_grid(tiff, grid)
        valid_pixels = int((pixels[2] > 0).sum())
        total_pixels = grid.width * grid.height
        report["raster"] = {"width": profile["width"], "height": profile["height"],
                            "bands": profile["count"], "dtype": profile["dtype"],
                            "crs": str(profile["crs"]), "transform": list(profile["transform"]),
                            "valid_pixels": valid_pixels, "total_pixels": total_pixels,
                            "valid_fraction": valid_pixels / total_pixels}
        report.pop("validation_stage", None)
        report["verdict"] = "GO"
    except Exception as exc:
        # Only fixed labels and numeric status escape; never inspect exception text.
        category = {
            "credentials": "missing_credentials", "OAuth": "invalid_oauth_response",
            "catalogue": "invalid_catalogue_response", "product selection": "product_validation_failed",
            "footprint": "footprint_validation_failed", "grid": "grid_construction_failed",
            "Process": "process_validation_failed",
        }[stage]
        retryable = False
        cause = exc.__cause__ or exc
        if isinstance(cause, httpx.TimeoutException):
            category, retryable = "timeout", True
        elif isinstance(cause, httpx.TransportError):
            category, retryable = "transport_error", True
        elif status is not None and status != 200:
            category = {400: "bad_request", 401: "authentication_rejected",
                        403: "access_denied", 429: "rate_limited"}.get(status, "http_error")
            retryable = status in {408, 429, 500, 502, 503, 504}
        elif not isinstance(exc, s1.CDSEError):
            category = "local_error"
        if report.get("validation_stage") == "response capture":
            category = "response_capture_failed"
        diagnostic = exc.diagnostic_category if isinstance(exc, s1.CDSEError) else None
        if stage == "Process" and diagnostic in VALIDATION_STAGES:
            category = diagnostic
            report["validation_stage"] = VALIDATION_STAGES[diagnostic]
        report["failure_stage"] = stage
        report["http_status"] = status
        report["error_category"] = category
        report["retryable"] = retryable
        report["error_type"] = "CDSEError" if isinstance(exc, s1.CDSEError) else "Exception"
        report["error"] = "Probe stopped; provider body and exception details withheld"
        # Failure diagnostics must not retain provider-derived identities or metadata.
        for key in ("source_product_ids", "service_metadata", "tar_members"):
            report.pop(key, None)
        if report["process_requests"]:
            report.setdefault("provenance_validation", "NOT PASSED")
    finally:
        client.event_hooks["request"].remove(record_request)
        client.event_hooks["response"].remove(record_response)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("before", "after"), default="before",
                        help="before: existing small probe; after: full configured Kosi grid")
    parser.add_argument("--full-grid", action="store_true",
                        help="Use the full configured Kosi grid for BEFORE; AFTER already uses it")
    parser.add_argument("--capture-response", action="store_true",
                        help="Save the bounded successful Process body privately outside the repository")
    args = parser.parse_args(argv)
    with httpx.Client(transport=httpx.HTTPTransport(retries=0), follow_redirects=False) as client:
        report = probe(client, os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET"),
                       capture=args.capture_response, scene=args.scene, full_grid=args.full_grid)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
