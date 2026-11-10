"""Offline model artifact lookup; never loads weights or contacts the network."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "model_artifacts.json"


@dataclass(frozen=True)
class ArtifactStatus:
    available: bool
    reason_code: str | None = None
    detail: str | None = None
    path: Path | None = None
    identity: str | None = None


def load_manifest(path: Path | None = None) -> dict:
    path = path or MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or set(data.get("providers", {})) != {"qwen2.5vl-3b", "grounding-dino-swint"}:
        raise ValueError("Invalid model artifact manifest")
    for name, spec in data["providers"].items():
        common = ("model_id", "local_path_env", "default_local_path", "revision", "sha256")
        extra = ("required_files",) if name == "qwen2.5vl-3b" else ("architecture", "checkpoint")
        if any(not isinstance(spec.get(key), str) or not spec[key] for key in common[:3]):
            raise ValueError(f"Invalid artifact configuration for {name}")
        if any(key not in spec for key in ("revision", "sha256", *extra)):
            raise ValueError(f"Incomplete artifact configuration for {name}")
        if spec["revision"] is not None and not isinstance(spec["revision"], str):
            raise ValueError(f"Invalid revision for {name}")
        if spec["sha256"] is not None and (not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64):
            raise ValueError(f"Invalid checksum for {name}")
        if name == "qwen2.5vl-3b" and (not isinstance(spec["required_files"], list) or not spec["required_files"] or any(not isinstance(f, str) or Path(f).name != f for f in spec["required_files"])):
            raise ValueError("Invalid Qwen required files")
        if name == "grounding-dino-swint" and any(not isinstance(spec[k], str) or Path(spec[k]).name != spec[k] for k in extra):
            raise ValueError("Invalid Grounding DINO configuration")
    return data


def validate_artifact(provider: str, *, model_id: str | None = None) -> ArtifactStatus:
    try:
        spec = load_manifest()["providers"][provider]
        local_model_dir = Path(model_id) if provider == "qwen2.5vl-3b" and model_id and Path(model_id).is_dir() else None
        if model_id is not None and local_model_dir is None and model_id != spec["model_id"]:
            return ArtifactStatus(False, "ARTIFACT_CONFIG_INVALID", "Model ID differs from artifact manifest.")
        override = os.environ.get(spec["local_path_env"])
        configured = local_model_dir or (Path(override) if override else Path(spec["default_local_path"]))
        if local_model_dir or override or configured.exists():
            path = configured
        else:
            from huggingface_hub import snapshot_download, try_to_load_from_cache
            if provider == "qwen2.5vl-3b":
                path = Path(snapshot_download(spec["model_id"], revision=spec["revision"], local_files_only=True))
            else:
                cached = try_to_load_from_cache(spec["model_id"], spec["checkpoint"], revision=spec["revision"])
                path = Path(cached) if isinstance(cached, str) else configured
        if provider == "qwen2.5vl-3b":
            required = spec["required_files"]
            missing = [name for name in required if not (path / name).is_file()]
            if not missing and json.loads((path / "config.json").read_text(encoding="utf-8")).get("model_type") != "qwen2_5_vl":
                raise ValueError("Qwen config has unexpected model type")
            index = path / "model.safetensors.index.json"
            if index.is_file():
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                if not isinstance(weight_map, dict) or not weight_map or any(Path(f).name != f for f in weight_map.values()):
                    raise ValueError("Invalid Qwen weight index")
                missing.extend(f for f in set(weight_map.values()) if not (path / f).is_file() or (path / f).stat().st_size == 0)
            elif not any((path / f).is_file() and (path / f).stat().st_size > 0 for f in ("model.safetensors", "pytorch_model.bin")):
                missing.append("model weights")
            if missing:
                return ArtifactStatus(False, "ARTIFACT_UNAVAILABLE", f"Missing Qwen files: {', '.join(sorted(missing))}.")
        elif path.name != spec["checkpoint"]:
            return ArtifactStatus(False, "ARTIFACT_CONFIG_INVALID", "Grounding DINO checkpoint filename differs from manifest.")
        elif not path.is_file() or path.stat().st_size == 0:
            return ArtifactStatus(False, "ARTIFACT_UNAVAILABLE", "Grounding DINO checkpoint is unavailable; configure SATQUERY_GROUNDING_CHECKPOINT.")
        return ArtifactStatus(True, path=path.resolve(), identity=f"{spec['model_id']}@{spec['revision'] or 'unverified-revision'}")
    except (KeyError, TypeError, ValueError) as exc:
        return ArtifactStatus(False, "ARTIFACT_CONFIG_INVALID", str(exc))
    except Exception:
        return ArtifactStatus(False, "ARTIFACT_UNAVAILABLE", "Local model artifact could not be resolved.")
