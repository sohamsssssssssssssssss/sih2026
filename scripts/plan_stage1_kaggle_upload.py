"""Plan the Kaggle upload of the Stage-1 train shards (loop 9, step 4).

No upload happens here and no archive is created: this loop has no Kaggle
credentials and the hard rules forbid one. The planner reads the verified shard
manifest, packs shards into parts of at most `--part-bytes`, and writes one
SHA256 list file per part so the eventual uploader can check what landed.

Kaggle's per-dataset and per-file size limits were NOT checked in this loop: they
are recorded as UNVERIFIED rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bigearthnet_shards import sha256_file  # noqa: E402

DEFAULT_PART_BYTES = 5 * 1000 ** 3


def plan(shard_dir: Path, manifest_path: Path, *, part_bytes: int = DEFAULT_PART_BYTES,
         dataset_name: str = "satquery-stage1-train-shards") -> dict:
    manifest = json.loads(manifest_path.read_text())
    records = sorted(manifest.get("shards", []), key=lambda record: record["index"])

    parts: list[dict] = []
    current: dict | None = None
    for record in records:
        npz = shard_dir / record["npz"]
        sidecar = shard_dir / record["sidecar"]
        for path in (npz, sidecar):
            if not path.is_file():
                raise FileNotFoundError(f"Shard file missing; run the verifier first: {path}")
        member_bytes = npz.stat().st_size + sidecar.stat().st_size
        if current is not None and current["bytes"] + member_bytes > part_bytes:
            parts.append(current)
            current = None
        if current is None:
            index = len(parts) + 1
            current = {"part": f"{dataset_name}-part{index:02d}", "bytes": 0,
                       "first_shard": record["index"], "last_shard": record["index"],
                       "shards": []}
        current["bytes"] += member_bytes
        current["last_shard"] = record["index"]
        current["shards"].append({
            "index": record["index"], "npz": record["npz"], "sidecar": record["sidecar"],
            "npz_bytes": record["npz_bytes"], "npz_sha256": record["npz_sha256"],
            "rows": record["rows"],
        })
    if current is not None:
        parts.append(current)

    total_bytes = sum(record["npz_bytes"] for record in records)
    return {
        "kind": "stage-1 Kaggle upload plan (loop 9, step 4) -- plan only, nothing uploaded",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "split_name": manifest.get("split_name"),
        "seed": manifest.get("seed"),
        "device": "CPU (py3.11.15 .venv-stage1)",
        "dtype": manifest.get("dtype"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "dataset_name": dataset_name,
        "shard_dir": str(shard_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": manifest.get("status"),
        "part_bytes_limit": part_bytes,
        "part_bytes_limit_is_a_project_choice": True,
        "kaggle_dataset_size_limit_bytes": None,
        "kaggle_limits_verified": False,
        "kaggle_limits_note": (
            "Kaggle dataset and per-file size limits were NOT retrieved or checked in this "
            "loop; the 5 GB part size is a project choice, not a verified Kaggle limit."),
        "upload_performed": False,
        "archives_created": False,
        "archives_note": (
            "No tar/zip part was assembled: that would need another ~24 GB of local disk and "
            "hours of deflate. Each part lists the exact shard files it must contain."),
        "measured_total_bytes": total_bytes,
        "measured_total_shards": len(records),
        "measured_total_rows": sum(record["rows"] for record in records),
        "measured_sidecar_bytes": sum(
            (shard_dir / record["sidecar"]).stat().st_size for record in records),
        "part_count": len(parts),
        "largest_part_bytes": max((part["bytes"] for part in parts), default=0),
        "parts": [{"part": part["part"], "bytes": part["bytes"],
                   "first_shard": part["first_shard"], "last_shard": part["last_shard"],
                   "shard_count": len(part["shards"]),
                   "sha256_list_file": f"{part['part']}.sha256.txt",
                   "shards": part["shards"]} for part in parts],
    }


def write_sha256_lists(report: dict, shard_dir: Path, out_dir: Path) -> list[str]:
    """One sha256sum-compatible list per part, over the shard files it must contain."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for part in report["parts"]:
        lines = []
        for shard in part["shards"]:
            for name in (shard["npz"], shard["sidecar"]):
                digest = sha256_file(shard_dir / name)
                lines.append(f"{digest}  {name}")
        path = out_dir / part["sha256_list_file"]
        path.write_text("\n".join(lines) + "\n")
        written.append(str(path))
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument("--dataset-name", default="satquery-stage1-train-shards")
    parser.add_argument("--sha256-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = plan(args.shards, args.manifest, part_bytes=args.part_bytes,
                  dataset_name=args.dataset_name)
    report["sha256_list_files"] = write_sha256_lists(report, args.shards, args.sha256_dir)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for part in report["parts"]:
        print(f"{part['part']} shards {part['first_shard']:05d}-{part['last_shard']:05d} "
              f"({part['shard_count']} shards, {part['bytes'] / 1e9:.2f} GB) -> "
              f"{part['sha256_list_file']}")
    print(f"measured total {report['measured_total_bytes'] / 1e9:.2f} GB in "
          f"{report['part_count']} parts; uploaded={report['upload_performed']}; "
          f"kaggle_limits_verified={report['kaggle_limits_verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())