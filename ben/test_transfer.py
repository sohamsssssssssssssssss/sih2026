"""Synthetic CPU tests for ben/transfer.py and checkpoint saving in ben/landcover.py. No real data."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")
pytest.importorskip("torchvision")

from ben import landcover, transfer  # noqa: E402
from ben.test_landcover import make_pack  # noqa: E402

BEN19 = ["Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest", "Coastal wetlands",
         "Complex cultivation patterns", "Coniferous forest", "Industrial or commercial units", "Inland waters",
         "Inland wetlands", "Land principally occupied by agriculture, with significant areas of natural vegetation",
         "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
         "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops", "Transitional woodland, shrub",
         "Urban fabric"]
COARSE = ["trees", "shrub_grass", "cropland", "built", "water", "wetland"]


def make_wc_pack(root, per_region=8):
    rng = np.random.default_rng(1)
    regions = [("punjab", "india"), ("kerala", "india"), ("vienna", "europe")]
    n = per_region * len(regions)
    root.mkdir(parents=True)
    frac = rng.dirichlet(np.ones(len(COARSE)), n).astype(np.float32)
    labels = (frac >= 0.05).astype(np.uint8)
    labels[::4] = 1  # every class positive in every region
    np.save(root / "s1.npy", rng.normal(-10, 4, (n, 2, 120, 120)).astype(np.float16))
    np.save(root / "s2_10m.npy", rng.integers(1, 5000, (n, 4, 120, 120)).astype(np.uint16))
    np.save(root / "fractions.npy", frac)
    np.save(root / "labels_coarse.npy", labels)
    pd.DataFrame({"idx": np.arange(n), "region": np.repeat([r for r, _ in regions], per_region),
                  "group": np.repeat([g for _, g in regions], per_region)}).to_parquet(root / "meta.parquet", index=False)
    (root / "classes.json").write_text(json.dumps({"coarse_classes": COARSE, "present_fraction": 0.05}))
    entries = [{"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
               for f in sorted(root.iterdir())]
    (root / "MANIFEST.json").write_text(json.dumps(entries))
    return root


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("transfer")
    ben = make_pack(base / "ben")
    out = base / "runs"
    mp = pytest.MonkeyPatch()
    mp.setattr(landcover, "BOOTSTRAP", 20); mp.setattr(transfer, "BOOT", 20); mp.setattr(transfer, "MIN_POSITIVES", 1)
    for mode in ("s1", "s2"):
        landcover.main(["--pack", str(ben), "--mode", mode, "--epochs", "1", "--batch-size", "32",
                        "--out-dir", str(out), "--save-checkpoint"])
    wc = make_wc_pack(base / "wc")
    result = transfer.main(["--wc-pack", str(wc), "--runs", str(out), "--modes", "s1", "s2", "--output", str(out / "transfer.json")])
    yield ben, out, wc, result
    mp.undo()


def test_ben19_to_coarse_mapping_is_complete_and_disjoint():
    members, unmapped = transfer.coarse_members(BEN19)
    assert list(members) == COARSE and unmapped == ["Beaches, dunes, sands"]
    flat = [i for v in members.values() for i in v]
    assert len(flat) == len(set(flat)) == 18
    assert [BEN19[i] for i in members["water"]] == ["Inland waters", "Marine waters"]


def test_coarse_score_is_max_member_sigmoid():
    logits = np.array([[0.0, 2.0, -1.0], [-3.0, -2.0, 4.0]])
    scores = transfer.to_coarse(logits, {"a": [0, 1], "b": [2]})
    sig = 1 / (1 + np.exp(-logits))
    assert np.allclose(scores, np.stack([sig[:, :2].max(1), sig[:, 2]], 1))


def test_checkpoint_holds_best_weights_and_ben_train_stats(runs):
    ben, out, _, _ = runs
    run = json.loads((out / "ben-landcover-s1.json").read_text())
    ckpt = torch.load(out / run["checkpoint"], map_location="cpu")
    assert ckpt["preprocess_stats"] == run["preprocess_stats"] and ckpt["best_epoch"] == run["best_epoch"]
    model = landcover.build_model(ckpt["in_channels"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    arrays, _, _, splits, _ = landcover.load_pack(ben, "s1")
    logits = landcover.predict(model, arrays, ckpt["preprocess_stats"], splits["test"], torch.device("cpu"), 32)
    assert np.allclose(logits, np.load(out / run["files"]["test_logits"]), atol=1e-5)
    # --compare still works with a checkpoint beside the logits
    assert landcover.compare(out / "ben-landcover-s1.json", out / "ben-landcover-s1.json")["diff"] == 0


def test_transfer_eval_reports_groups_controls_and_level_shift(runs):
    _, out, _, result = runs
    saved = json.loads((out / "transfer.json").read_text())
    for mode in ("s1", "s2"):
        res = saved["modes"][mode]
        # synthetic BEN uses 5 of the 19 classes -> only 4 coarse classes have members
        assert res["coarse_classes"] == ["trees", "cropland", "built", "water"]
        assert set(res["groups"]) == {"ben_test", "india", "europe", "punjab", "kerala", "vienna"}
        assert res["groups"]["india"]["n"] == 16 and res["groups"]["europe"]["n"] == 8
        for g in res["groups"].values():
            assert {"macro_map", "macro_map_ci95", "prevalence_macro_map", "shuffled_macro_map", "per_class_ap"} <= set(g)
        assert set(res["differences"]) == {"europe-ben_test", "india-europe", "india-ben_test"}
        assert set(res["groups_common"]) == {"ben_test", "india", "europe"}
        assert all(d["classes"] == res["common_classes"] for d in res["differences"].values())
        chans = {r["channel"] for r in res["level_shift"]}
        assert chans == ({"VV", "VH"} if mode == "s1" else {"B02", "B03", "B04", "B08"})
        assert np.load(out / f"wc-{mode}-logits.npy").shape == (24, 5)
    assert "level shift" in transfer.summary(result)


def test_wc_manifest_mismatch_stops(tmp_path):
    wc = make_wc_pack(tmp_path / "wc", per_region=2)
    transfer.verify_wc_manifest(wc)
    np.save(wc / "fractions.npy", np.zeros((6, 6), np.float32))
    with pytest.raises(ValueError, match="STOP: sha256 mismatch for fractions.npy"):
        transfer.verify_wc_manifest(wc)


def test_level_shift_uses_checkpoint_stats():
    stats = {"s1": {"mean": [-12.0, -19.0], "std": [5.0, 5.0]}}
    arrays = {"s1": np.full((4, 2, 2, 2), -6.0, np.float16)}
    rows = transfer.level_shift(arrays, stats, np.array(["a", "a", "b", "b"]))
    vv = [r for r in rows if r["channel"] == "VV"]
    assert [r["shift_in_ben_std"] for r in vv] == [1.2, 1.2] and all(r["flag_gt_1std"] for r in vv)


def test_transfer_cell_contract():
    text = (Path(transfer.__file__).parent / "run_transfer_gpu.txt").read_text()
    assert text.splitlines()[0] == "%%bash" and text.rstrip().endswith('echo "CELL DONE"')
    for needle in ("set -euo pipefail", "trap ", 'find /kaggle/input -type f -path "*/ben/landcover.py"',
                   "s2_20m.npy", "fractions.npy", "--save-checkpoint", "ben/transfer.py", "export PYTHONPATH",
                   "for MODE in s1 s2 s1s2", "0.5598", "0.6254", "0.6283"):
        assert needle in text, needle
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"/kaggle/input/[A-Za-z0-9]", code)


def test_macro_map_uses_fixed_class_set_with_min_positives(monkeypatch):
    monkeypatch.setattr(transfer, "BOOT", 50); monkeypatch.setattr(transfer, "MIN_POSITIVES", 10)
    rng = np.random.default_rng(0)
    y = np.zeros((40, 3), np.float32); y[:20, 0] = 1; y[::2, 1] = 1; y[0, 2] = 1  # class 2 has 1 positive
    s = rng.random((40, 3))
    m, boot = transfer.group_metrics(y, s)
    assert m["classes_used"] == [0, 1] and m["positives"] == [20, 20, 1]
    assert m["macro_map"] == pytest.approx(np.mean([landcover.average_precision_score(y[:, c], s[:, c]) for c in (0, 1)]))
    assert m["macro_map_ci95"][0] <= m["macro_map"] <= m["macro_map_ci95"][1] and len(boot) == 50
