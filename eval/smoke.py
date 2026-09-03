"""Fifty-step end-to-end training smoke test.

Default run is CPU-only and checks the training loop itself: a batch flows
forward and backward, the loss is finite and decreasing, and a checkpoint
round-trips. That runs anywhere and stays the default.

--gpu adds hardware gates for the accelerator the job will actually use. They
exist because the toy Linear model above passes happily on any device and will
not catch a config that is invalid for the GPU underneath it — bf16 on a T4,
say, which is Turing (sm_75) and has no bf16 support. Finding that out 30
minutes into an allocated session is the expensive way to learn it.
"""

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


# Ampere (sm_80) is the floor for bf16 and FlashAttention-2. T4 is sm_75.
MIN_CAPABILITY_BF16 = 8
MIN_CAPABILITY_FLASH_ATTENTION = 8
# Fraction of total VRAM the loaded weights may occupy before the run is
# considered too tight to survive activations.
MAX_WEIGHT_VRAM_FRACTION = 0.9


def check_gpu(config: dict) -> None:
    """Gate the config against the accelerator actually present.

    Raises AssertionError naming the fix. Config keys are optional: a config
    that does not request bf16 or flash_attention_2 simply skips those gates.
    """
    if not torch.cuda.is_available():
        raise AssertionError(
            "--gpu requested but CUDA is unavailable; run without --gpu on CPU"
        )

    device = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device} sm_{major}{minor} {total_gb:.1f}GB")

    precision = str(config.get("precision", "fp16")).lower()
    if precision == "bf16" and major < MIN_CAPABILITY_BF16:
        raise AssertionError(
            f"config requests bf16 but {device} is sm_{major}{minor}; "
            f"bf16 needs sm_{MIN_CAPABILITY_BF16}0+. Use fp16."
        )

    attention = str(config.get("attn_implementation", "sdpa")).lower()
    if attention == "flash_attention_2" and major < MIN_CAPABILITY_FLASH_ATTENTION:
        raise AssertionError(
            f"config requests flash_attention_2 but {device} is sm_{major}{minor}; "
            f"it needs sm_{MIN_CAPABILITY_FLASH_ATTENTION}0+. Use sdpa."
        )

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    if allocated_gb > total_gb * MAX_WEIGHT_VRAM_FRACTION:
        raise AssertionError(
            f"weights already occupy {allocated_gb:.1f}GB of {total_gb:.1f}GB; "
            "no headroom for activations. Lower image_size or batch size."
        )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"GPU gates passed: precision={precision}, attn={attention}, "
        f"weights={allocated_gb:.1f}GB, peak={peak_gb:.1f}GB, budget={total_gb:.1f}GB"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also gate the config against the present accelerator "
        "(bf16/flash-attention capability, VRAM headroom)",
    )
    args = parser.parse_args()
    try:
        run(args.config)
        if args.gpu:
            with args.config.open(encoding="utf-8") as handle:
                check_gpu(yaml.safe_load(handle) or {})
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    passed = "50 steps, finite decreasing loss, backward and checkpoint reload succeeded"
    print(f"SMOKE TEST PASSED: {passed}" + (" + GPU gates" if args.gpu else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
