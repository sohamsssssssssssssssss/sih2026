"""Append-only experiment registry: one immutable JSON record per experiment.

Records live in eval/registry/<experiment_id>.json and are never overwritten.
Every field in REQUIRED_FIELDS must be present; use null (None) for a field
that genuinely does not apply (e.g. adapter_revision for a base-model run),
never omit it.
"""

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "eval" / "registry"
STATUSES = frozenset({"PASSED", "FAILED", "INCONCLUSIVE"})
ID_PATTERN = re.compile(r"^SQ-\d{8}-\d{3}$")
REQUIRED_FIELDS = (
    "experiment_id",
    "title",
    "dataset",
    "dataset_version",
    "dataset_checksum",
    "split",
    "model_id",
    "model_revision",
    "adapter_revision",
    "seed",
    "hyperparameters",
    "command",
    "started_at",
    "ended_at",
    "metrics",
    "artifact_paths",
    "status",
    "notes",
)
DEFAULT_PACKAGES = (
    "torch", "transformers", "peft", "accelerate", "bitsandbytes",
    "qwen-vl-utils", "numpy", "rasterio",
)


def repository_state(root: Path = ROOT) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "url": git("remote", "get-url", "origin"),
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def environment(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }


def write_record(record: dict, registry_dir: Path = REGISTRY_DIR) -> Path:
    """Validate and write a record; refuses to overwrite an existing one.

    `repository` and `environment` are captured automatically unless the
    caller supplies them (e.g. a record transcribed from a remote Kaggle run,
    where the local machine is not the hardware that produced the result).
    """
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"experiment record missing fields: {missing}")
    if not ID_PATTERN.match(record["experiment_id"]):
        raise ValueError(f"bad experiment_id {record['experiment_id']!r}; want SQ-YYYYMMDD-XXX")
    if record["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    path = registry_dir / f"{record['experiment_id']}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; experiment records are never overwritten")
    full = {
        **record,
        "repository": record.get("repository") or repository_state(),
        "environment": record.get("environment") or environment(),
    }
    registry_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(full, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
