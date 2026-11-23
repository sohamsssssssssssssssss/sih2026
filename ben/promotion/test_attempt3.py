import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import torch

from ben import landcover as lc, transfer as tr
from ben.promotion.attempt3_g4 import INTEG, align
from ben.promotion.attempt3_groups import OLD, assign
from ben.promotion.attempt3_train import indices, train_one
from ben.promotion.attempt3_score import comparison, ece


OLD_COUNTS = {name: [50, 60, 70, 80, 90] for group in OLD for name in group}
NEW_COUNTS = {f"new-{i:02d}": [50, 60, 70, 80, 90] for i in range(10)}
GROUPS = assign(OLD_COUNTS, NEW_COUNTS)["groups"]


def test_label_split_is_invariant_to_input_order_and_breaks_exact_ties_by_name():
    actual = assign(OLD_COUNTS, NEW_COUNTS)
    assert actual == assign(dict(reversed(list(OLD_COUNTS.items()))), dict(reversed(list(NEW_COUNTS.items()))))
    assert actual["worst_prevalence_gap"] == 0
    assert [actual["groups"][g]["regions"] for g in "CDE"] == [
        tuple(list(NEW_COUNTS)[:3]), tuple(list(NEW_COUNTS)[3:6]), tuple(list(NEW_COUNTS)[6:])
    ]
    impossible = {name: [0, 60, 70, 80, 90] for name in NEW_COUNTS}
    assert assign(OLD_COUNTS, impossible)["status"] == "not run"


@pytest.mark.parametrize("sensor", ["S1", "S2"])
def test_g4_grid_and_app_normalization(sensor):
    channels = {"S1": 2, "S2": 4}[sensor]
    constants = {"S1": [-15, -20], "S2": [1200, 2300, 3400, 4500]}[sensor]
    band = np.broadcast_to(np.array(constants, dtype=np.float32)[:, None, None], (channels, 128, 128))
    patch = SimpleNamespace(utm_x=480000, utm_y=2100000, utm_epsg=32644)
    transform = from_origin(patch.utm_x - 640, patch.utm_y + 640, 10, 10)
    with MemoryFile() as buf:
        with buf.open(driver="GTiff", crs="EPSG:32644", transform=transform,
                      count=channels, width=128, height=128, dtype="float32", nodata=np.nan) as dataset:
            dataset.write(band)
            dataset.update_tags(cloud_pct="9.9", satquery_provenance=json.dumps({
                "item_id": "synthetic", "acquisition_datetime": "2021-03-10T00:00:00Z"
            }))
        chip, record = align(buf.read(), patch, sensor)
    assert chip.shape == (channels, 120, 120)
    assert np.array_equal(chip[:, 0, 0], constants)
    assert len(record["aligned_pixels_sha256"]) == 64

    spec = importlib.util.spec_from_file_location("integration_landcover", INTEG / "ben/landcover.py")
    integ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integ)
    inputs = {sensor.lower(): chip[None]}
    stats = {sensor.lower(): {"scale": 1 / 10000, "mean": [0.1] * channels, "std": [2] * channels,
                              "clip_lo": [-40] * channels, "clip_hi": [-5] * channels}}
    idx, device = np.array([0]), torch.device("cpu")
    assert torch.equal(lc.make_batch(inputs, stats, idx, device), integ.make_batch(inputs, stats, idx, device))


@pytest.mark.parametrize("fold", list(GROUPS))
@pytest.mark.parametrize("mode, channels", [("s1", 2), ("s2", 4)])
def test_tiny_cpu_fold(fold, mode, channels):
    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    meta = pd.DataFrame({"region": [r for g in GROUPS.values() for r in g["regions"] for _ in range(5)]})
    cls = list(dict.fromkeys(c for members in tr.COARSE.values() for c in members))
    members, _ = tr.coarse_members(cls)
    net = lc.build_model(channels, len(cls))
    checkpoint = {"mode": mode, "inputs": (mode,), "in_channels": channels, "classes": cls,
                  "state_dict": net.state_dict()}
    arrays = {mode: rng.integers(100, 1000, size=(len(meta), channels, 32, 32)).astype(np.float32)}
    yy = ((np.arange(len(meta))[:, None] + np.arange(len(members))) % 3 != 0).astype(np.float32)
    held_out = set(GROUPS[fold]["regions"])
    train, val, test = indices(meta, held_out)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert set(meta.iloc[test].region) == held_out
    assert not set(meta.iloc[np.r_[train, val]].region) & held_out
    weights, scores = train_one(checkpoint, arrays, yy, train, val, test, torch.device("cpu"),
                               epochs=1, batch_size=32)
    assert weights["best_epoch"] == 1
    assert scores.shape == (len(test), 5) and np.isfinite(scores).all()

def test_later_cpu_gates_handle_boundary_bins_and_paired_control():
    scores = np.array([[0.0], [0.1], [1.0]])
    labels = np.array([[0], [0], [1]])
    assert ece(scores, labels) == pytest.approx(0.1 / 3)
    y = np.tile(np.array([[0], [1]], dtype=np.uint8), (50, 1))
    folds = np.repeat(list("ABCDE"), 20)
    good, bad = y.astype(float), 1.0 - y
    measured = comparison(y, good, bad, folds)
    assert measured["valid_draws"] == 2000
    assert measured["pass"] and measured["ci95"][0] > 0
    assert not comparison(y, good, good, folds)["pass"]
