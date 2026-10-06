"""Lightweight GPU runtime verification.

Checks dependency and configuration state without loading models or
performing inference. Designed to run on a future GPU machine or the
current CPU-only development host.

DO NOT import or invoke any model weights; only check availability,
configuration files, and environment variables.
"""
import json
import os
import sys
from pathlib import Path

from models.artifacts import load_manifest, validate_artifact


def check_qwen_dependency():
    """Check if qwen-vl-utils is importable."""
    try:
        import qwen_vl_utils  # noqa: F401
        return "READY"
    except ImportError:
        return "MISSING"


def check_qwen_artifact_config():
    """Check if the Qwen artifact config in the manifest is valid.

    This does NOT check whether model files are cached; it only verifies
    that the manifest's model_id matches the expected model ID and that
    the manifest structure is correct.
    """
    try:
        manifest = load_manifest()
        spec = manifest["providers"]["qwen2.5vl-3b"]
        if spec.get("model_id") != "Qwen/Qwen2.5-VL-3B-Instruct":
            return "INVALID"
        # Verify required fields exist and are non-empty strings
        common = ("model_id", "local_path_env", "default_local_path", "revision")
        for key in common:
            if not isinstance(spec.get(key), str) or not spec[key]:
                return "INVALID"
        return "VALID"
    except Exception:
        return "INVALID"


def check_grounding_dependency():
    """Check if groundingdino is importable."""
    try:
        import groundingdino  # noqa: F401
        return "READY"
    except ImportError:
        return "MISSING"


def check_grounding_config():
    """Check if SATQUERY_GROUNDING_CONFIG environment variable is set
    or the default config path exists."""
    if os.environ.get("SATQUERY_GROUNDING_CONFIG"):
        return "READY"
    # Check default config path: <repo_root>/groundingdino/config/GroundingDINO_SwinT_OGC.py
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    if default_config.is_file():
        return "READY"
    return "MISSING"


def check_grounding_checkpoint():
    """Check if the Grounding DINO checkpoint path exists.

    Checks SATQUERY_GROUNDING_CHECKPOINT env var first, then the
    default local path from the manifest.
    """
    # Check env var first
    env_path = os.environ.get("SATQUERY_GROUNDING_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return "READY"
    # Check default local path from manifest
    try:
        manifest = load_manifest()
        spec = manifest["providers"]["grounding-dino-swint"]
        default_path = Path(spec["default_local_path"])
        if default_path.is_file():
            return "READY"
    except Exception:
        pass
    return "MISSING"


def check_bert_asset():
    """Check if bert-base-uncased is in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache("bert-base-uncased", "config.json")
        if isinstance(cached, str):
            return "READY"
    except Exception:
        pass
    # Also check if the dir structure exists
    cache_dir = Path("/root/.cache/huggingface/hub/models--bert-base-uncase")
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        return "READY"
    return "MISSING"


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "AVAILABLE"
    except ImportError:
        pass
    return "UNAVAILABLE"


def main():
    report = {
        "qwen_dependency": check_qwen_dependency(),
        "qwen_artifact_config": check_qwen_artifact_config(),
        "grounding_dependency": check_grounding_dependency(),
        "grounding_config": check_grounding_config(),
        "grounding_checkpoint": check_grounding_checkpoint(),
        "bert_asset": check_bert_asset(),
        "cuda": check_cuda(),
    }
    # Print human-readable summary
    for key, value in report.items():
        print(f"{key}: {value}")
    # Also print JSON for machine reading
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
