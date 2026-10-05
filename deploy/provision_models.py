"""Provision SatQuery model artifacts into one host directory, then verify offline.

This is the only step that touches the network. It is run once, explicitly, by
an operator; the running service never downloads anything (backend/services.py
forces HF_HUB_OFFLINE=1). Run it inside the GPU image so the library versions
match the ones that will read the files:

    docker run --rm --user "$(id -u):$(id -g)" \
      -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
      -v /srv/satquery/models:/models \
      satquery-backend:gpu python deploy/provision_models.py --dest /models

Resulting layout (matches docker-compose.gpu.yml and configs/model_artifacts.json):

    <dest>/qwen2.5-vl-3b-instruct/       SATQUERY_QWEN_MODEL_DIR
    <dest>/groundingdino_swint_ogc.pth   SATQUERY_GROUNDING_CHECKPOINT
    <dest>/huggingface/hub/              HF_HUB_CACHE (bert-base-uncased, the text
                                         encoder named by Grounding DINO's config)
    <dest>/PROVENANCE.json               resolved revisions and SHA-256 digests

`--verify-only` skips downloads and re-runs the offline checks.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_DIR = "qwen2.5-vl-3b-instruct"
DINO_REPO = "ShilongLiu/GroundingDINO"
DINO_FILE = "groundingdino_swint_ogc.pth"
# Exactly this repo id: groundingdino/config/GroundingDINO_SwinT_OGC.py sets
# text_encoder_type = "bert-base-uncased", and the offline cache lookup is keyed
# by the id as written (models--bert-base-uncased), not by a redirect target.
BERT_REPO = "bert-base-uncased"
BERT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_env(dest: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_CACHE=str(dest / "huggingface" / "hub"),
        SATQUERY_QWEN_MODEL_DIR=str(dest / QWEN_DIR),
        SATQUERY_GROUNDING_CHECKPOINT=str(dest / DINO_FILE),
        PYTHONPATH=str(ROOT),
    )
    return env


def download(dest: Path, args: argparse.Namespace) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    # Under `docker run --user <host uid>` the image's HOME is not writable;
    # keep huggingface_hub's own state (token, xet cache) next to the weights.
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        home.mkdir(parents=True, exist_ok=True)
        writable = os.access(home, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        os.environ["HF_HOME"] = str(dest / ".hf-home")
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    hub_cache = dest / "huggingface" / "hub"

    qwen_sha = api.model_info(QWEN_REPO, revision=args.qwen_revision).sha
    print(f"Qwen {QWEN_REPO}@{qwen_sha} -> {dest / QWEN_DIR}", flush=True)
    snapshot_download(QWEN_REPO, revision=qwen_sha, local_dir=dest / QWEN_DIR)

    dino_sha = api.model_info(DINO_REPO, revision=args.dino_revision).sha
    print(f"Grounding DINO {DINO_REPO}/{DINO_FILE}@{dino_sha} -> {dest}", flush=True)
    hf_hub_download(DINO_REPO, DINO_FILE, revision=dino_sha, local_dir=dest)

    bert_sha = api.model_info(BERT_REPO, revision=args.bert_revision).sha
    print(f"Text encoder {BERT_REPO}@{bert_sha} -> {hub_cache}", flush=True)
    # Downloaded with the default revision name so the cache's refs/main points
    # at it; that is what an offline from_pretrained("bert-base-uncased") reads.
    snapshot_download(BERT_REPO, revision=args.bert_revision, cache_dir=hub_cache, allow_patterns=BERT_FILES)

    qwen_files = sorted(p for p in (dest / QWEN_DIR).iterdir() if p.is_file() and p.suffix in {".safetensors", ".json"})
    return {
        "provisioned_at": datetime.now(timezone.utc).isoformat(),
        "qwen": {"repo": QWEN_REPO, "revision": qwen_sha, "files": {p.name: sha256(p) for p in qwen_files}},
        "grounding_dino": {"repo": DINO_REPO, "file": DINO_FILE, "revision": dino_sha, "sha256": sha256(dest / DINO_FILE)},
        "text_encoder": {"repo": BERT_REPO, "revision": bert_sha},
    }


VERIFY = r"""
import json, sys
from models.artifacts import validate_artifact
failures = []
for provider in ("qwen2.5vl-3b", "grounding-dino-swint"):
    status = validate_artifact(provider)
    print(f"{provider}: available={status.available} reason={status.reason_code} path={status.path}")
    if not status.available:
        failures.append(f"{provider}: {status.reason_code}: {status.detail}")
try:
    from transformers import AutoTokenizer, BertModel
    AutoTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    print("bert-base-uncased: loads offline from HF_HUB_CACHE")
except Exception as exc:
    failures.append(f"bert-base-uncased: {type(exc).__name__}: {exc}")
if failures:
    print("\n".join(["FAILED:"] + failures))
    sys.exit(1)
print("All model artifacts resolve offline.")
"""


def verify(dest: Path) -> int:
    """Check offline in a fresh process, with exactly the service's env vars."""
    return subprocess.run([sys.executable, "-c", VERIFY], env=runtime_env(dest), cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--qwen-revision", default=None, help="commit/branch/tag; default: main")
    parser.add_argument("--dino-revision", default=None, help="commit/branch/tag; default: main")
    parser.add_argument("--bert-revision", default=None, help="commit/branch/tag; default: main")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    dest = args.dest.resolve()
    if not args.verify_only:
        if os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0", "false", "False"):
            parser.error("downloads need network: run with -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0")
        provenance = download(dest, args)
        (dest / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {dest / 'PROVENANCE.json'}")
    return verify(dest)


if __name__ == "__main__":
    raise SystemExit(main())
