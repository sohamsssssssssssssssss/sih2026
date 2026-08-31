"""Idempotently download and extract real imagery for the resolution ladder."""

import argparse
import hashlib
import math
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
LOVEDA_FILES = {
    "Train.zip": (
        "https://zenodo.org/records/5706578/files/Train.zip?download=1",
        "de2b196043ed9b4af1690b3f9a7d558f",
    ),
    "Val.zip": (
        "https://zenodo.org/records/5706578/files/Val.zip?download=1",
        "84cae2577468ff0b5386758bb386d31d",
    ),
}
DOTA_PAGE = "https://captain-whu.github.io/DOTA/dataset.html"
DOTA_GDRIVE_FOLDERS = (
    "https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK",
    "https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224",
)
DOWNLOAD_SEGMENTS = 8


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream integrity checksum, not security use
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "sih26167-ladder/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming {destination.name} at {offset:,} bytes")
    else:
        print(f"Downloading {destination.name}")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            if offset and response.status != 206:
                offset = 0
                partial.unlink(missing_ok=True)
            mode = "ab" if offset else "wb"
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    except urllib.error.URLError as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise
        print(f"Python HTTPS failed ({exc}); retrying securely with segmented curl")
        _curl_segmented(curl, url, partial)
    partial.replace(destination)


def _curl_segmented(curl: str, url: str, partial: Path) -> None:
    head = subprocess.run(
        [curl, "--silent", "--show-error", "--location", "--head", url],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lengths = [
        line.split(":", 1)[1].strip()
        for line in head.splitlines()
        if line.lower().startswith("content-length:")
    ]
    if not lengths:
        raise RuntimeError("Zenodo did not provide Content-Length")
    total = int(lengths[-1])
    chunk = math.ceil(total / DOWNLOAD_SEGMENTS)
    first_segment = partial.with_suffix(partial.suffix + ".000")
    if partial.exists() and not first_segment.exists():
        partial.replace(first_segment)

    def fetch(index: int) -> Path:
        start = index * chunk
        end = min(total - 1, start + chunk - 1)
        segment = partial.with_suffix(partial.suffix + f".{index:03d}")
        expected = end - start + 1
        current = segment.stat().st_size if segment.exists() else 0
        if current == expected:
            return segment
        if current > expected:
            segment.unlink()
            current = 0
        command = [
            curl,
            "--fail",
            "--location",
            "--retry",
            "5",
            "--silent",
            "--show-error",
            "--range",
            f"{start + current}-{end}",
            "--output",
            str(segment) if current == 0 else "/dev/stdout",
            url,
        ]
        if current:
            with segment.open("ab") as output:
                subprocess.run(command, check=True, stdout=output)
        else:
            subprocess.run(command, check=True)
        if segment.stat().st_size != expected:
            raise RuntimeError(f"Incomplete segment {index} for {partial.name}")
        return segment

    print(f"Fetching {total:,} bytes in {DOWNLOAD_SEGMENTS} resumable segments")
    with ThreadPoolExecutor(max_workers=DOWNLOAD_SEGMENTS) as executor:
        segments = list(executor.map(fetch, range(DOWNLOAD_SEGMENTS)))
    joining = partial.with_suffix(partial.suffix + ".joining")
    with joining.open("wb") as output:
        for segment in segments:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if joining.stat().st_size != total:
        raise RuntimeError(f"Combined download has incorrect size for {partial.name}")
    joining.replace(partial)
    for segment in segments:
        segment.unlink()


def extract_once(archive: Path, destination: Path) -> None:
    marker = destination / f".{archive.stem}.extracted"
    if marker.exists():
        print(f"Already extracted: {archive.name}")
        return
    print(f"Extracting {archive.name} into {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def acquire_loveda() -> int:
    destination = RAW_ROOT / "loveda"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_md5) in LOVEDA_FILES.items():
        archive = destination / filename
        download(url, archive)
        actual_md5 = md5(archive)
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
            )
        extract_once(archive, destination)
    return 0


def acquire_dota() -> int:
    destination = RAW_ROOT / "dota"
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.rglob("*.png")):
        print(f"DOTA imagery already present under {destination}")
        return 0
    gdown = shutil.which("gdown")
    if gdown:
        try:
            for folder_url in DOTA_GDRIVE_FOLDERS:
                subprocess.run(
                    [gdown, "--folder", folder_url, "--output", str(destination)],
                    check=True,
                    timeout=120,
                )
            if any(destination.rglob("*.png")):
                print(f"Downloaded DOTA imagery under {destination}")
                return 0
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"gdown could not acquire DOTA automatically: {exc}", file=sys.stderr)
    print(
        "DOTA v2 requires Google Drive confirmation or credentials. Download the images "
        f"manually from {DOTA_PAGE}, place archives/files in {destination}, then rerun."
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("loveda", "dota"), default="loveda")
    args = parser.parse_args()
    try:
        return acquire_loveda() if args.dataset == "loveda" else acquire_dota()
    except (
        OSError,
        subprocess.SubprocessError,
        urllib.error.URLError,
        zipfile.BadZipFile,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"DATA ACQUISITION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
