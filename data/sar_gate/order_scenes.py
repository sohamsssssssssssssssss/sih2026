"""Search recent Sentinel-1 GRD scenes over India and submit HyP3 RTC jobs."""

import argparse
import json
import netrc
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SAR_ROOT = ROOT / "data" / "sar_gate"
JOBS_PATH = SAR_ROOT / "jobs.json"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

FIXED_LOCATIONS = (
    {"name": "mumbai_coastal", "label": "Mumbai coastal", "lat": 19.05, "lon": 72.85},
    {
        "name": "maharashtra_farmland",
        "label": "Maharashtra farmland",
        "lat": 19.9,
        "lon": 75.3,
    },
    {
        "name": "western_ghats_forest",
        "label": "Western Ghats forest",
        "lat": 17.9,
        "lon": 73.5,
    },
    {"name": "konkan_coast", "label": "Konkan coast", "lat": 16.7, "lon": 73.3},
)
FLAT_INLAND_CANDIDATES = (
    {"name": "flat_inland_plain", "label": "Kutch flat inland plain", "lat": 23.3, "lon": 70.3},
    {
        "name": "flat_inland_plain",
        "label": "Madhya Pradesh flat inland plain",
        "lat": 23.5,
        "lon": 78.5,
    },
    {
        "name": "flat_inland_plain",
        "label": "Rajasthan flat inland plain",
        "lat": 27.0,
        "lon": 75.5,
    },
)


def require_earthdata_credentials() -> bool:
    path = Path.home() / ".netrc"
    try:
        credentials = netrc.netrc(str(path)).authenticators(EARTHDATA_HOST)
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        credentials = None
    if credentials and credentials[0] and credentials[2]:
        return True
    print("Earthdata credentials are required; no usable ~/.netrc entry was found.", file=sys.stderr)
    print("NASA Earthdata signup: https://urs.earthdata.nasa.gov/users/new", file=sys.stderr)
    print("ASF Vertex account/linking: https://search.asf.alaska.edu/", file=sys.stderr)
    print(
        "Then add: machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD",
        file=sys.stderr,
    )
    print("Run: chmod 600 ~/.netrc", file=sys.stderr)
    return False


def print_authentication_help() -> None:
    print(
        "Earthdata/ASF authentication failed. Check the credentials in ~/.netrc and "
        "make sure the ASF Vertex account is linked to Earthdata.",
        file=sys.stderr,
    )
    print("NASA Earthdata account: https://urs.earthdata.nasa.gov/", file=sys.stderr)
    print("ASF Vertex account/linking: https://search.asf.alaska.edu/", file=sys.stderr)
    print("Run: chmod 600 ~/.netrc", file=sys.stderr)


def _acquisition_time(result: Any) -> datetime:
    value = result.properties.get("startTime") or result.properties.get("stopTime")
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def search_latest(location: dict[str, Any], start: datetime, end: datetime) -> Any | None:
    import asf_search as asf

    results = asf.geo_search(
        platform=asf.PLATFORM.SENTINEL1,
        beamMode=asf.BEAMMODE.IW,
        processingLevel=asf.PRODUCT_TYPE.GRD_HD,
        polarization=asf.POLARIZATION.VV_VH,
        intersectsWith=f"POINT({location['lon']} {location['lat']})",
        start=start,
        end=end,
        maxResults=100,
    )
    return max(results, key=_acquisition_time) if results else None


def select_scenes(start: datetime, end: datetime) -> list[tuple[dict[str, Any], Any]]:
    selected = []
    for location in FIXED_LOCATIONS:
        result = search_latest(location, start, end)
        if result is None:
            raise RuntimeError(f"No dual-pol Sentinel-1 GRD scene found for {location['label']}")
        selected.append((location, result))

    inland_options = []
    for candidate in FLAT_INLAND_CANDIDATES:
        result = search_latest(candidate, start, end)
        if result is not None:
            inland_options.append((candidate, result))
    if not inland_options:
        raise RuntimeError("No dual-pol Sentinel-1 GRD scene found for any flat inland candidate")
    selected.append(max(inland_options, key=lambda pair: _acquisition_time(pair[1])))
    return selected


def save_manifest(payload: dict[str, Any]) -> None:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(JOBS_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")
    if not require_earthdata_credentials():
        return 2
    if JOBS_PATH.exists():
        print(f"Existing order manifest found: {JOBS_PATH}")
        print("Delete or archive it deliberately before placing a new set of RTC orders.")
        return 0

    import hyp3_sdk
    from hyp3_sdk.exceptions import AuthenticationError

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    selected = select_scenes(start, end)
    try:
        hyp3 = hyp3_sdk.HyP3()
    except AuthenticationError:
        print_authentication_help()
        return 2
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_start": start.isoformat(),
        "search_end": end.isoformat(),
        "jobs": [],
    }
    for location, result in selected:
        granule = str(result.properties["sceneName"])
        batch = hyp3.submit_rtc_job(
            granule=granule,
            name=f"sih26167-sar-{location['name']}",
            radiometry="gamma0",
            resolution=30,
            scale="power",
            speckle_filter=False,
            include_rgb=False,
        )
        job = batch[0]
        record = {
            **location,
            "granule": granule,
            "acquisition_start": result.properties.get("startTime"),
            "job_id": job.job_id,
            "status": job.status_code,
        }
        payload["jobs"].append(record)
        save_manifest(payload)
        print(
            f"SUBMITTED {location['label']}: granule={granule} "
            f"job_id={job.job_id} status={job.status_code}"
        )

    print(
        "RTC processing takes 20-90 min per scene, check status via hyp3_sdk job.refresh() "
        "or the Vertex web UI. Re-run download_scenes.py once jobs show COMPLETE."
    )
    print("For this repository, the download entry point is: python3 data/sar_gate/process_scenes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
