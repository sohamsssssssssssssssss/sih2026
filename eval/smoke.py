"""Fifty-step end-to-end training smoke test."""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TileSample, validate_tile_sample  # noqa: E402


def run(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise AssertionError("config must load as a mapping")

    channels, height, width = 4, 8, 8
    tile: TileSample = {
        "id": "synthetic-0",
        "optical": np.zeros((channels, height, width), dtype=np.float32),
        "sar": np.zeros((3, height, width), dtype=np.float32),
        "optical_t2": None,
        "gsd": 10.0,
        "sensor": "synthetic",
        "meta": {},
    }
    validate_tile_sample(tile)
    inputs = torch.ones((int(config["batch_size"]), channels, height, width)) / math.sqrt(
        channels * height * width
    )
    targets = torch.zeros((int(config["batch_size"]), 1))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=int(config["batch_size"]))
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (int(config["batch_size"]), channels, height, width)
    assert batch_y.shape == (int(config["batch_size"]), 1)

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(channels * height * width, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["lr"]))
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        output = model(batch_x)
        assert torch.isfinite(output).all(), "forward pass returned a non-finite tensor"
        loss = torch.nn.functional.mse_loss(output, batch_y)
        assert torch.isfinite(loss), "loss is not finite"
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None, "backward pass produced a missing gradient"
            assert torch.isfinite(parameter.grad).all(), "backward pass produced NaN/Inf gradients"
        optimizer.step()
        losses.append(loss.item())
    assert math.isfinite(losses[-1]) and losses[-1] < losses[0], "loss did not decrease over 50 steps"

    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(channels * height * width, 1))
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        assert torch.isfinite(reloaded(batch_x)).all(), "reloaded checkpoint failed inference"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED: 50 steps, finite decreasing loss, backward and checkpoint reload succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
