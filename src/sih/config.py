"""Config loading with single-level inheritance.

Every run is a YAML file in configs/. No notebook-only runs, ever — that rule
is what lets six people on six Kaggle accounts run each other's experiments.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name_or_path: str) -> dict[str, Any]:
    """Load a config by name ('s0_baseline_qwen3b') or path, resolving `extends`."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path.removesuffix('.yaml')}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No config at {name_or_path} or {p}")

    with open(p) as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(parent), cfg)

    cfg.setdefault("name", p.stem)
    cfg["_path"] = str(p)
    return cfg


def dump_config(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return yaml.safe_dump(clean, sort_keys=False)
