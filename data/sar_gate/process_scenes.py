"""Download completed HyP3 RTC products and render fixed-scale SAR false color."""

import argparse
import json
import netrc
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAR_ROOT = ROOT / "data" / "sar_gate"
JOBS_PATH = SAR_ROOT / "jobs.json"
RAW_ROOT = SAR_ROOT / "raw"
RENDERED_ROOT = SAR_ROOT / "rendered"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
DB_MIN = -25.0
DB_MAX = 5.0


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


def gamma0_to_db(gamma0: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(gamma0, 1e-6, None))


def _box_sum(array: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(array, pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def lee_filter(image: np.ndarray, size: int = 7) -> np.ndarray:
    """Hand-rolled Lee filter with reflect padding and NaN-aware local moments."""
    if size != 7:
        raise ValueError("This gate uses the required 7x7 Lee filter")
    valid = np.isfinite(image)
    values = np.where(valid, image, 0.0).astype(np.float64)
    counts = np.maximum(_box_sum(valid.astype(np.float64), size), 1.0)
    local_mean = _box_sum(values, size) / counts
    local_second = _box_sum(values * values, size) / counts
    local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
    supported = valid & (counts > 1)
    noise_variance = float(np.median(local_variance[supported])) if np.any(supported) else 0.0
    weight = np.maximum(local_variance - noise_variance, 0.0) / np.maximum(
        local_variance, 1e-12
    )
    filtered = local_mean + weight * (values - local_mean)
    return np.where(valid, filtered, np.nan).astype(np.float32)


def normalize_fixed(image_db: np.ndarray) -> np.ndarray:
    clipped = np.clip(image_db, DB_MIN, DB_MAX)
    normalized = (clipped - DB_MIN) / (DB_MAX - DB_MIN)
    return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)


def read_power(path: Path, max_dimension: int) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = min(1.0, max_dimension / max(dataset.height, dataset.width))
        height = max(1, round(dataset.height * scale))
        width = max(1, round(dataset.width * scale))
        data = dataset.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
            masked=True,
        )
    return np.asarray(data.filled(np.nan), dtype=np.float32)


def safe_extract(archive: Path, destination: Path) -> None:
    marker = destination / f".{archive.stem}.extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (destination / member.filename).resolve().is_relative_to(resolved):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def find_polarization_tif(root: Path, polarization: str) -> Path:
    pattern = re.compile(rf"(?:^|[_-]){polarization}(?:$|[_-])", re.IGNORECASE)
    matches = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() == ".tif" and pattern.search(path.stem)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {polarization} GeoTIFF under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def render_job(record: dict[str, Any], product_root: Path, max_dimension: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vv_path = find_polarization_tif(product_root, "VV")
    vh_path = find_polarization_tif(product_root, "VH")
    vv_power = read_power(vv_path, max_dimension)
    vh_power = read_power(vh_path, max_dimension)
    if vv_power.shape != vh_power.shape:
        raise ValueError(f"VV/VH dimensions differ: {vv_power.shape} vs {vh_power.shape}")
    vv_db = lee_filter(gamma0_to_db(vv_power), size=7)
    vh_db = lee_filter(gamma0_to_db(vh_power), size=7)
    rgb = np.stack(
        [
            normalize_fixed(vv_db),
            normalize_fixed(vh_db),
            normalize_fixed(vv_db - vh_db),
        ],
        axis=-1,
    )
    value_min = float(np.min(rgb))
    value_max = float(np.max(rgb))
    value_std = float(np.std(rgb))
    if not np.isfinite(rgb).all() or value_max - value_min < 0.05 or value_std < 0.01:
        raise ValueError(
            f"Degenerate render for {record['name']}: min={value_min:.4f}, "
            f"max={value_max:.4f}, std={value_std:.4f}"
        )
    RENDERED_ROOT.mkdir(parents=True, exist_ok=True)
    output = RENDERED_ROOT / f"{record['name']}.png"
    plt.imsave(output, rgb)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"PNG render was not created: {output}")
    print(
        f"RENDERED {record['label']}: {output} shape={rgb.shape} "
        f"range=[{value_min:.4f}, {value_max:.4f}] std={value_std:.4f}"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dimension", type=int, default=2048)
    args = parser.parse_args()
    if args.max_dimension < 256:
        parser.error("--max-dimension must be at least 256")
    if not require_earthdata_credentials():
        return 2
    if not JOBS_PATH.is_file():
        print(f"Missing {JOBS_PATH}; run python3 data/sar_gate/order_scenes.py first", file=sys.stderr)
        return 2

    import hyp3_sdk
    from hyp3_sdk.exceptions import AuthenticationError

    manifest = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    try:
        hyp3 = hyp3_sdk.HyP3()
    except AuthenticationError:
        print_authentication_help()
        return 2
    rendered = []
    incomplete = []
    for record in manifest["jobs"]:
        job = hyp3.get_job_by_id(record["job_id"])
        print(f"JOB {record['label']}: id={job.job_id} status={job.status_code}")
        if not job.succeeded():
            incomplete.append((record["label"], job.status_code))
            continue
        job_root = RAW_ROOT / record["name"]
        product_root = job_root / "product"
        product_tifs = (
            [path for path in product_root.rglob("*") if path.suffix.lower() == ".tif"]
            if product_root.exists()
            else []
        )
        direct_tifs = (
            [path for path in job_root.glob("*.tif") if path.is_file()]
            if job_root.exists()
            else []
        )
        if direct_tifs and not product_tifs:
            product_root = job_root
        if not product_tifs and not direct_tifs:
            archives = [path for path in job_root.glob("*.zip") if path.is_file()]
            downloaded = [] if archives else job.download_files(location=job_root)
            for path in [*archives, *downloaded]:
                if path.suffix.lower() == ".zip":
                    safe_extract(path, product_root)
            newly_downloaded_tifs = [
                path for path in downloaded if path.suffix.lower() == ".tif"
            ]
            if newly_downloaded_tifs:
                product_root = job_root
            if not archives and not downloaded:
                raise RuntimeError(f"No files downloaded for completed job {job.job_id}")
        rendered.append(render_job(record, product_root, args.max_dimension))

    if incomplete:
        print("RTC jobs not yet complete:")
        for label, status in incomplete:
            print(f"  {label}: {status}")
        print(
            "RTC processing takes 20-90 min per scene. Check with hyp3.refresh(job) or "
            "https://search.asf.alaska.edu/ and rerun this script."
        )
    print(f"Rendered {len(rendered)}/{len(manifest['jobs'])} completed scenes")
    return 0 if len(rendered) == len(manifest["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
