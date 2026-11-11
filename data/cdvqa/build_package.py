"""Assemble the CDVQA pilot package from repo sources (no images, no weights).

Run: python3.11 data/cdvqa/build_package.py <UTC-timestamp> [prefix]
Writes packages/<prefix>-<ts>/ with package-manifest.json (default prefix: cdvqa-g0d). The zip is made separately.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "data/cdvqa/package"
SOURCES = {
    "scripts/cdvqa_pilot.py": "scripts/cdvqa_pilot.py",
    "scripts/train_remote_sensing_adapter.py": "scripts/train_remote_sensing_adapter.py",
    "scripts/stage1_runtime.py": "scripts/stage1_runtime.py",
    "training/__init__.py": "training/__init__.py",
    "training/remote_sensing.py": "training/remote_sensing.py",
    "eval/__init__.py": "eval/__init__.py",
    "eval/cdvqa_pairs.py": "eval/cdvqa_pairs.py",
    "eval/test_cdvqa_pairs.py": "eval/test_cdvqa_pairs.py",
    "eval/test_cdvqa_pilot.py": "eval/test_cdvqa_pilot.py",
}
MANIFESTS = ["train-8k.v1.jsonl", "train-8k.v2.jsonl", "eval-test1-400.v1.jsonl", "eval-test2-400.v1.jsonl", "provenance.json"]
TEMPLATE_FILES = ["pilot-config.json", "configs/model_artifacts.json", "requirements-target.txt",
                  "requirements-cpu-preflight.txt", "pilot.py", "KAGGLE.md", "README.md"]


def build(timestamp, note, prefix="cdvqa-g0d"):
    out = REPO / "packages" / f"{prefix}-{timestamp}"
    if out.exists():
        raise FileExistsError(out)
    for dst, src in SOURCES.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / src, out / dst)
    for name in TEMPLATE_FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATE / name, out / name)
    for name in MANIFESTS:
        (out / "data/manifests/cdvqa").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "data/manifests/cdvqa" / name, out / "data/manifests/cdvqa" / name)
    for name in ("KAGGLE.md", "README.md"):
        text = (out / name).read_text().replace("cdvqa-g0d-TIMESTAMP", out.name)  # placeholder in templates
        (out / name).write_text(text)
    config = json.loads((out / "pilot-config.json").read_text())
    config["package"] = out.name
    (out / "pilot-config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "package-manifest.json")
    sha = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "package-manifest.json").write_text(json.dumps({
        "package": out.name, "payload_file_count": len(sha),
        "payload_bytes": sum(p.stat().st_size for p in files), "payload_sha256": sha, "notes": note},
        indent=2, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    prefix = sys.argv[2] if len(sys.argv) > 2 else "cdvqa-g0d"
    print(build(sys.argv[1], "CDVQA two-image pilot package; no images, no weights", prefix))
