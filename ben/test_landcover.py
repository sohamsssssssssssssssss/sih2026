"""Synthetic CPU tests for ben/landcover.py. No real data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")
pytest.importorskip("torchvision")

from ben import landcover  # noqa: E402

CLASSES = ["Arable land", "Coniferous forest", "Inland waters", "Marine waters", "Urban fabric"]
REQUIRED_KEYS = {"mode", "seed", "config", "epochs_run", "best_epoch", "n", "macro_map", "classes_skipped_no_test_positive",
                 "micro_f1", "macro_f1", "per_class", "prior_baseline", "bootstrap", "control", "manifest_sha256", "code",
                 "runtime_s", "torch_version", "device", "test_evaluations", "preprocess_stats", "files", "history"}


def make_pack(root, n_split=(60, 10, 20)):
    rng = np.random.default_rng(0)
    n = sum(n_split)
    split = np.repeat(["train", "validation", "test"], n_split)
    labels = (rng.random((n, len(CLASSES))) < 0.4).astype(np.uint8)
    labels[:len(CLASSES) * 2] |= np.tile(np.eye(len(CLASSES), dtype=np.uint8), (2, 1))  # every class present in train
    labels[n_split[0]:n_split[0] + len(CLASSES)] |= np.eye(len(CLASSES), dtype=np.uint8)  # and in validation
    labels[-len(CLASSES):] |= np.eye(len(CLASSES), dtype=np.uint8)  # and in test
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "s1.npy", rng.normal(-15, 4, (n, 2, 120, 120)).astype(np.float16))
    np.save(root / "s2_10m.npy", rng.integers(1, 6000, (n, 4, 120, 120)).astype(np.uint16))
    np.save(root / "s2_20m.npy", rng.integers(1, 6000, (n, 6, 60, 60)).astype(np.uint16))
    np.save(root / "labels_multihot.npy", labels)
    (root / "classes.json").write_text(json.dumps(CLASSES))
    pd.DataFrame({"idx": np.arange(n), "patch_id": [f"p{i}" for i in range(n)], "split": split,
                  "country": "X"}).to_parquet(root / "selection.parquet", index=False)
    entries = [{"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
               for f in sorted(root.iterdir())]
    (root / "MANIFEST.json").write_text(json.dumps(entries))
    return root


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One 1-epoch CPU run per mode on one synthetic pack; test-split predict calls are counted."""
    base = tmp_path_factory.mktemp("ben")
    pack = make_pack(base / "pack")
    mp = pytest.MonkeyPatch()
    mp.setattr(landcover, "BOOTSTRAP", 30)
    calls = []
    original = landcover.predict

    def counting(model, arrays, stats, idx, device, batch_size):
        calls.append(len(idx))
        return original(model, arrays, stats, idx, device, batch_size)
    mp.setattr(landcover, "predict", counting)
    out = base / "out"
    results = {}
    for mode in ("s1", "s2", "s1s2"):
        calls.clear()
        results[mode] = (landcover.main(["--pack", str(pack), "--mode", mode, "--control", "shuffle", "--epochs", "1",
                                         "--batch-size", "32", "--out-dir", str(out), "--progress", str(out / "progress.txt")]),
                         list(calls))
    yield pack, out, results
    mp.undo()


def test_manifest_mismatch_stops(tmp_path):
    pack = make_pack(tmp_path / "pack", (10, 5, 5))
    landcover.verify_manifest(pack)
    labels = np.load(pack / "labels_multihot.npy"); labels[0, 0] ^= 1; np.save(pack / "labels_multihot.npy", labels)
    with pytest.raises(ValueError, match="STOP: sha256 mismatch for labels_multihot.npy"):
        landcover.verify_manifest(pack)
    with pytest.raises(ValueError, match="STOP"):
        landcover.main(["--pack", str(pack), "--mode", "s1", "--epochs", "1", "--out-dir", str(tmp_path / "o")])
    assert not (tmp_path / "o" / "ben-landcover-s1.json").exists()


def test_each_mode_runs_one_epoch_with_expected_shapes_and_keys(runs):
    pack, out, results = runs
    for mode, (result, _) in results.items():
        assert REQUIRED_KEYS <= set(result), REQUIRED_KEYS - set(result)
        assert result["config"]["in_channels"] == {"s1": 2, "s2": 4, "s1s2": 6}[mode]
        assert result["epochs_run"] == 1 and result["best_epoch"] == 1
        assert result["n"] == {"train": 60, "validation": 10, "test": 20}
        assert np.load(out / result["files"]["test_logits"]).shape == (20, len(CLASSES))
        assert list(np.load(out / result["files"]["test_idx"])) == list(range(70, 90))
        saved = json.loads((out / f"ben-landcover-{mode}.json").read_text())
        assert REQUIRED_KEYS <= set(saved) and saved["manifest_sha256"] == hashlib.sha256((pack / "MANIFEST.json").read_bytes()).hexdigest()
        assert len(saved["per_class"]) == len(CLASSES) and 0 <= saved["macro_map"] <= 1
        lo, hi = saved["bootstrap"]["macro_map_ci95"]
        assert lo <= hi
        assert set(saved["preprocess_stats"]) == set({"s1": ["s1"], "s2": ["s2"], "s1s2": ["s1", "s2"]}[mode])
    assert "epoch 1/1" in (out / "progress.txt").read_text()


def test_test_split_is_evaluated_exactly_once(runs):
    _, _, results = runs
    for result, calls in results.values():
        # one validation predict per epoch, one final validation predict (thresholds), one test predict
        assert calls.count(20) == 1 and result["test_evaluations"] == 1
        assert calls == [10, 10, 20]


def test_shuffle_control_reports_model_prior_and_shuffled(runs):
    _, _, results = runs
    for result, _ in results.values():
        control = result["control"]
        assert control["model_macro_map"] == result["macro_map"]
        assert control["prior_baseline_macro_map"] == result["prior_baseline"]["macro_map"]
        assert 0 <= control["macro_map"] <= 1


def test_preprocess_uses_train_only_stats():
    arrays = {"s1": np.zeros((4, 2, 3, 3), np.float16), "s2": np.full((4, 4, 3, 3), 10000, np.uint16)}
    arrays["s1"][2:] = 50  # non-train rows must not move the stats
    arrays["s2"][2:] = 0
    stats = landcover.train_stats(arrays, np.array([0, 1]))
    assert stats["s1"]["mean"] == [0.0, 0.0] and stats["s1"]["clip_hi"] == [0.0, 0.0]
    assert stats["s2"]["mean"] == [1.0] * 4


def test_compare_is_paired_and_runs_from_saved_files(runs, tmp_path):
    _, out, _ = runs
    result = landcover.main(["--compare", str(out / "ben-landcover-s1s2.json"), str(out / "ben-landcover-s2.json"),
                             "--output", str(tmp_path / "diff.json")])
    assert result["a"] == "s1s2" and result["b"] == "s2"
    assert result["diff"] == pytest.approx(result["a_macro_map"] - result["b_macro_map"])
    assert result["diff_ci95"][0] <= result["diff_ci95"][1]
    same = landcover.compare(out / "ben-landcover-s2.json", out / "ben-landcover-s2.json")
    assert same["diff"] == 0 and same["diff_ci95"] == [0.0, 0.0]
    assert json.loads((tmp_path / "diff.json").read_text())["b"] == "s2"


def test_gpu_cell_contract():
    text = (Path(landcover.__file__).parent / "run_landcover_gpu.txt").read_text()
    assert text.splitlines()[0] == "%%bash"
    for needle in ('set -euo pipefail', "trap ", 'find /kaggle/input -type f -path "*/ben/landcover.py"',
                   "find /kaggle/input -name MANIFEST.json", "export PYTHONPATH", "/kaggle/tmp", "--control shuffle",
                   "--compare", 'echo "CELL DONE"', "for MODE in s1 s2 s1s2"):
        assert needle in text, needle
    assert text.rstrip().endswith('echo "CELL DONE"')
