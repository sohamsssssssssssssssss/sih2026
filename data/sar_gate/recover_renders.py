"""Rebuild a lost data/sar_gate/jobs.json from the live ASF HyP3 account.

The five RTC jobs were ordered with data/sar_gate/order_scenes.py under the
account that owns ~/.netrc. The order manifest (data/sar_gate/jobs.json) is
intentionally gitignored (see .gitignore: raw products, job IDs, and renders
are unversioned), so a fresh checkout cannot run process_scenes.py without
re-deriving it. This script recovers the manifest without re-ordering or
inventing provenance:

- Job identity comes from the account itself: jobs are matched by the exact
  "sih26167-sar-" name prefix that order_scenes.py submits, never guessed.
- Scene locations come only from the committed FIXED_LOCATIONS in
  order_scenes.py. flat_inland_plain was selected at order time from the
  three committed FLAT_INLAND_CANDIDATES by newest acquisition; that choice
  is not recorded anywhere, so this script re-derives it by replicating
  order_scenes.py's own search_latest selection at each candidate and
  requiring the recovered product's truncated granule name to match
  exactly one candidate.
- Granule names come from the HyP3 product filenames; acquisition times come
  from the ASF API record of the recovered granule.

If any job, product, or unique location cannot be resolved, the script
reports the blocker and exits nonzero without writing a partial manifest.
Rendering itself is left entirely to the committed data/sar_gate/
process_scenes.py; this script never touches imagery.
"""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SAR_ROOT = ROOT / "data" / "sar_gate"
JOBS_PATH = SAR_ROOT / "jobs.json"
ORDER_SCENES_PATH = SAR_ROOT / "order_scenes.py"
JOB_NAME_PREFIX = "sih26167-sar-"
PRODUCT_NAME_PATTERN = re.compile(
    r"^(?P<truncated>S1[ABCD]_IW_(?P<start>\d{8}T\d{6})_(?P<tail>[A-Z]{3}))_RTC30_G_gpuned_[0-9A-F]{4}\.zip$"
)
SEARCH_DAYS_BACK = 60  # order_scenes.py default --days window


def load_order_scenes() -> Any:
    """Import the committed order_scenes.py so location constants have one source of truth."""
    spec = importlib.util.spec_from_file_location("sar_gate_order_scenes", ORDER_SCENES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import committed order scenes module: {ORDER_SCENES_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truncated_granule(filename: str) -> tuple[str, str] | None:
    """Return (truncated granule, start timestamp) from a HyP3 RTC product filename."""
    match = PRODUCT_NAME_PATTERN.match(filename)
    if match is None:
        return None
    return match.group("truncated"), match.group("start")


def granule_matches(full_name: str, truncated: str) -> bool:
    """Check an ASF granule name against the HyP3 truncated form.

    HyP3 truncates granule names to platform, beam mode, start timestamp, and a
    HyP3-derived mission token (e.g. DVR) that is not present in the ASF name,
    so identity is pinned by platform + beam + exact start timestamp; the start
    timestamp is unique per Sentinel-1 acquisition.
    """
    parts = truncated.split("_")
    full_parts = full_name.split("_")
    if len(parts) != 4 or len(full_parts) < 5:
        return False
    return (
        full_parts[0] == parts[0]
        and full_parts[1] == parts[1]
        and full_parts[4] == parts[2]
    )


def search_scenes_at(location: dict[str, Any], start: datetime, end: datetime) -> list[Any]:
    """Same ASF search parameters as order_scenes.py, returning every hit."""
    import asf_search as asf

    return list(
        asf.geo_search(
            platform=asf.PLATFORM.SENTINEL1,
            beamMode=asf.BEAMMODE.IW,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            polarization=asf.POLARIZATION.VV_VH,
            intersectsWith=f"POINT({location['lon']} {location['lat']})",
            start=start,
            end=end,
            maxResults=100,
        )
    )


def search_latest(location: dict[str, Any], start: datetime, end: datetime) -> Any | None:
    """Replicate order_scenes.py's newest-scene selection for one location."""
    results = search_scenes_at(location, start, end)

    def acquisition_time(result: Any) -> datetime:
        value = result.properties.get("startTime") or result.properties.get("stopTime")
        if not value:
            return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    return max(results, key=acquisition_time) if results else None


def resolve_flat_inland(
    candidates: tuple[dict[str, Any], ...],
    target_truncated: str,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, Any], Any] | None:
    """Find the one committed candidate whose order-time selection matches the recovered product."""
    matches: list[tuple[dict[str, Any], Any]] = []
    for candidate in candidates:
        latest = search_latest(candidate, start, end)
        if latest is None:
            continue
        granule = str(latest.properties.get("sceneName", ""))
        if granule_matches(granule, target_truncated):
            matches.append((candidate, latest))
    if len(matches) != 1:
        return None
    return matches[0]


def recover() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    order_scenes = load_order_scenes()
    committed_locations = {entry["name"]: entry for entry in order_scenes.FIXED_LOCATIONS}
    flat_candidates = order_scenes.FLAT_INLAND_CANDIDATES

    import hyp3_sdk

    try:
        hyp3 = hyp3_sdk.HyP3()
    except Exception as exc:  # hyp3_sdk raises AuthenticationError subclasses; fail closed either way
        print(f"HyP3 authentication failed: {exc}", file=sys.stderr)
        return 2

    batch = hyp3.find_jobs()
    jobs = [job for job in batch.jobs if job.name and job.name.startswith(JOB_NAME_PREFIX)]
    print(f"Found {len(jobs)} HyP3 jobs with name prefix '{JOB_NAME_PREFIX}'")

    grouped: dict[str, list[Any]] = {}
    for job in jobs:
        grouped.setdefault(job.name, []).append(job)

    blockers: list[str] = []
    records: list[dict[str, Any]] = []
    request_times: list[datetime] = []
    for name in sorted(grouped):
        matches = grouped[name]
        if len(matches) != 1:
            blockers.append(f"{name}: expected exactly one account job, found {len(matches)}")
            continue
        job = matches[0]
        scene_name = name[len(JOB_NAME_PREFIX) :]
        if not job.succeeded():
            blockers.append(f"{name}: job status is {job.status_code}, not SUCCEEDED")
            continue
        files = [f.get("filename", "") for f in job.files]
        products = [f for f in files if f.endswith(".zip")]
        if len(products) != 1:
            blockers.append(f"{name}: expected one product zip, found {products}")
            continue
        product = products[0]
        parsed = truncated_granule(product)
        if parsed is None:
            blockers.append(f"{name}: product filename does not match the HyP3 RTC naming convention: {product}")
            continue
        truncated, acquisition_start_token = parsed
        request_time = job.request_time
        if request_time is None:
            blockers.append(f"{name}: job has no recorded request time")
            continue
        if request_time.tzinfo is None:
            request_time = request_time.replace(tzinfo=timezone.utc)
        request_times.append(request_time)

        record: dict[str, Any] = {
            "name": scene_name,
            "label": scene_name.replace("_", " ").title(),
            "granule": truncated,
            "acquisition_start_token": acquisition_start_token,
            "job_id": job.job_id,
            "status": job.status_code,
            "recovered_from": "hyp3_account",
        }

        if scene_name in committed_locations:
            location = committed_locations[scene_name]
            record.update(lat=location["lat"], lon=location["lon"], label=location["label"])
        elif scene_name == "flat_inland_plain":
            window_end = request_time
            window_start = window_end - timedelta(days=SEARCH_DAYS_BACK)
            resolved = resolve_flat_inland(flat_candidates, truncated, window_start, window_end)
            if resolved is None:
                blockers.append(
                    f"{name}: the recovered product ({truncated}) does not uniquely match the "
                    "committed FLAT_INLAND_CANDIDATES selection; refusing to guess its location"
                )
                continue
            location, result = resolved
            record.update(lat=location["lat"], lon=location["lon"], label=location["label"])
            record["granule_full"] = str(result.properties.get("sceneName", ""))
            start_time = result.properties.get("startTime")
            if start_time:
                record["acquisition_start"] = str(start_time)
        else:
            blockers.append(f"{name}: no committed location and no re-derivation rule for scene '{scene_name}'")
            continue

        records.append(record)
        print(
            f"RECOVERED {record['label']}: job_id={record['job_id']} "
            f"granule={truncated} lat={record['lat']} lon={record['lon']}"
        )

    expected_names = {entry["name"] for entry in order_scenes.FIXED_LOCATIONS} | {
        candidate["name"] for candidate in flat_candidates
    }
    for scene_name in sorted(expected_names):
        if not any(record["name"] == scene_name for record in records):
            blockers.append(f"{scene_name}: no '{JOB_NAME_PREFIX}{scene_name}' job in the HyP3 account")

    if blockers:
        print("\nRecovery blockers (nothing was written):", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 1

    earliest = min(request_times)
    latest = max(request_times)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recovered_by": "data/sar_gate/recover_renders.py",
        "search_start": earliest.isoformat(),
        "search_end": latest.isoformat(),
        "jobs": records,
    }
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(JOBS_PATH)
    print(f"\nWrote {len(records)} recovered jobs to {JOBS_PATH}")
    print("Next step (committed pipeline): python3 data/sar_gate/process_scenes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(recover())
