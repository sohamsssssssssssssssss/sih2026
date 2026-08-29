"""Configuration-driven training entry point."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.registry import get  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = get(config["model_name"])
    train_method = getattr(model, "train", None)
    if train_method is None:
        print(f"Model '{config['model_name']}' has no training implementation; config loaded successfully.")
        return 0
    train_method(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
