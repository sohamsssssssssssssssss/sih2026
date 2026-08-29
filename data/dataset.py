"""Canonical tile sample schema and validation."""

from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

Sensor = Literal["S2", "S1", "cartosat", "risat", "dota", "synthetic"]


class TileSample(TypedDict):
    id: str
    optical: NDArray[np.float32]  # float32[C,H,W]
    sar: NDArray[np.float32] | None  # float32[3,H,W] or None
    optical_t2: NDArray[np.float32] | None  # float32[C,H,W] or None
    gsd: float
    sensor: Sensor
    meta: dict[str, Any]


def validate_tile_sample(sample: TileSample) -> TileSample:
    """Validate required metadata and tensor-like array shapes on load."""
    if sample.get("gsd") is None:
        raise ValueError("Tile sample is missing required field: gsd")
    if sample.get("sensor") is None:
        raise ValueError("Tile sample is missing required field: sensor")
    if sample["sensor"] not in {"S2", "S1", "cartosat", "risat", "dota", "synthetic"}:
        raise ValueError(f"Unsupported sensor: {sample['sensor']}")
    if sample["optical"].dtype != np.float32 or sample["optical"].ndim != 3:
        raise ValueError("optical must be float32[C,H,W]")
    if sample["sar"] is not None and (
        sample["sar"].dtype != np.float32
        or sample["sar"].ndim != 3
        or sample["sar"].shape[0] != 3
    ):
        raise ValueError("sar must be float32[3,H,W] or None")
    if sample["optical_t2"] is not None and (
        sample["optical_t2"].dtype != np.float32 or sample["optical_t2"].ndim != 3
    ):
        raise ValueError("optical_t2 must be float32[C,H,W] or None")
    return sample


def load_tile(record: TileSample) -> TileSample:
    """Loader stub; replace storage access without changing its output contract."""
    return validate_tile_sample(record)
