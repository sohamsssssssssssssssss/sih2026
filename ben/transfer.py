"""Evaluate BEN-trained land-cover checkpoints on the WorldCover 2021 transfer pack (India + Europe control).

BEN-19 scores are collapsed to 6 coarse WorldCover classes (coarse score = max sigmoid over member classes).
Inputs are normalised with the BEN TRAIN stats stored in each checkpoint, never with WorldCover stats.

  python ben/transfer.py --wc-pack WC_DIR --runs RUN_DIR --modes s1 s2 s1s2 --output OUT.json
RUN_DIR holds ben-landcover-<mode>.json, -checkpoint.pt and -test-{logits,labels}.npy from landcover.py.
"""
import argparse
import json
from pathlib import Path
import time

import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python ben/transfer.py` without PYTHONPATH
    sys.path.insert(0, str(ROOT))
from ben import landcover as lc  # noqa: E402

SEED, BOOT = 17, 1000
MIN_POSITIVES = 10  # a coarse class enters a group's macro mAP only with >= 10 positives in that group
COARSE = {  # BEN-19 (reBEN) class names -> WorldCover-derived coarse classes. "Beaches, dunes, sands" is unmapped.
    "trees": ["Broad-leaved forest", "Coniferous forest", "Mixed forest", "Agro-forestry areas"],
    "shrub_grass": ["Transitional woodland, shrub", "Natural grassland and sparsely vegetated areas",
                    "Moors, heathland and sclerophyllous vegetation", "Pastures"],
    "cropland": ["Arable land", "Permanent crops", "Complex cultivation patterns",
                 "Land principally occupied by agriculture, with significant areas of natural vegetation"],
    "built": ["Urban fabric", "Industrial or commercial units"],
    "water": ["Inland waters", "Marine waters"],
    "wetland": ["Inland wetlands", "Coastal wetlands"],
}
WC_REQUIRED = ("s1.npy", "s2_10m.npy", "fractions.npy", "labels_coarse.npy", "meta.parquet", "classes.json")


def verify_wc_manifest(pack):
    manifest = pack / "MANIFEST.json"
    entries = {e["file"]: e["sha256"] for e in json.loads(manifest.read_text())}
    for name in WC_REQUIRED:
        if name not in entries:
            raise ValueError(f"STOP: {name} missing from WorldCover MANIFEST.json")
    for name, digest in entries.items():
        if lc.sha256(pack / name) != digest:
            raise ValueError(f"STOP: sha256 mismatch for {name}")
    return lc.sha256(manifest)


def coarse_members(classes):
    """Index lists of BEN classes per coarse class; coarse classes with no member present in `classes` are dropped."""
    pos = {c: i for i, c in enumerate(classes)}
    members = {k: [pos[c] for c in v if c in pos] for k, v in COARSE.items()}
    mapped = {c for v in COARSE.values() for c in v}
    return {k: v for k, v in members.items() if v}, sorted(set(classes) - mapped)


def coarse_logits(logits, members):
    """Shared max-member pooling, differentiable for torch adaptation logits."""
    if isinstance(logits, torch.Tensor):
        return torch.stack([logits[:, idx].max(1).values for idx in members.values()], 1)
    return np.stack([logits[:, idx].max(1) for idx in members.values()], 1)


def to_coarse(logits, members):
    prob = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    return coarse_logits(prob, members)


def load_wc(pack):
    """Load the same WorldCover arrays, labels and metadata for transfer and adaptation."""
    meta = pd.read_parquet(pack / "meta.parquet")
    info = json.loads((pack / "classes.json").read_text())
    labels = np.load(pack / "labels_coarse.npy").astype(np.float32)
    raw = {"s1": np.load(pack / "s1.npy", mmap_mode="r"),
           "s2": np.load(pack / "s2_10m.npy", mmap_mode="r")}
    return raw, labels, meta, info


def group_metrics(y, scores, rng_seed=SEED):
    """Coarse macro mAP + CI, per-class AP, prevalence baseline (constant score), shuffled-image control.

    The macro average uses a FIXED class set per group: classes with >= MIN_POSITIVES positives in the group.
    Otherwise rare classes enter/leave the average across bootstrap resamples and the CI is not about one quantity.
    """
    pos = y.sum(0)
    keep = pos >= MIN_POSITIVES
    per_all = lc.macro_map(y, scores)[2]
    if not keep.any():
        nan = float("nan")
        return {"n": int(len(y)), "macro_map": nan, "macro_map_ci95": [nan, nan], "classes_used": [], "min_positives": MIN_POSITIVES,
                "prevalence_macro_map": nan, "shuffled_macro_map": nan, "per_class_ap": [None if np.isnan(a) else float(a) for a in per_all],
                "positives": [int(v) for v in pos]}, np.full(BOOT, np.nan)
    yk, sk = y[:, keep], scores[:, keep]
    perm = np.random.default_rng(rng_seed).permutation(len(y))
    boot = lc.bootstrap_maps(yk, sk, len(y), BOOT)
    return {"n": int(len(y)), "macro_map": lc.macro_map(yk, sk)[0], "macro_map_ci95": lc.ci95(boot),
            "classes_used": [int(i) for i in np.flatnonzero(keep)], "min_positives": MIN_POSITIVES,
            "prevalence_macro_map": lc.macro_map(yk, np.broadcast_to(yk.mean(0), yk.shape))[0],
            "shuffled_macro_map": lc.macro_map(yk, sk[perm])[0],
            "per_class_ap": [None if np.isnan(a) else float(a) for a in per_all], "positives": [int(v) for v in pos]}, boot


def level_shift(arrays, stats, regions):
    """Per region and channel: mean/std of raw inputs vs the checkpoint's BEN train stats; flag |dmean| > 1 BEN std."""
    rows = []
    for key, chans in (("s1", ["VV", "VH"]), ("s2", ["B02", "B03", "B04", "B08"])):
        if key not in stats:
            continue
        scale = 1.0 if key == "s1" else stats[key]["scale"]
        x = arrays[key].astype(np.float32) * scale
        for region in sorted(set(regions)):
            sel = x[regions == region]
            for c, ch in enumerate(chans):
                mean, std = float(sel[:, c].mean()), float(sel[:, c].std())
                bm, bs = stats[key]["mean"][c], stats[key]["std"][c]
                rows.append({"region": region, "channel": ch, "mean": mean, "std": std, "ben_mean": bm, "ben_std": bs,
                             "shift_in_ben_std": (mean - bm) / bs, "flag_gt_1std": abs(mean - bm) > bs})
    return rows


def evaluate(args):
    t0 = time.time()
    wc, runs = Path(args.wc_pack), Path(args.runs)
    wc_manifest = verify_wc_manifest(wc)
    raw, y_wc_all, meta, wc_info = load_wc(wc)
    wc_classes = wc_info["coarse_classes"]
    regions, groups = meta.region.to_numpy(), meta.group.to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"wc_manifest_sha256": wc_manifest, "coarse_present_fraction": wc_info["present_fraction"], "modes": {},
           "regions": sorted(set(regions)), "n_per_region": {r: int((regions == r).sum()) for r in sorted(set(regions))}}
    for mode in args.modes:
        run = json.loads((runs / f"ben-landcover-{mode}.json").read_text())
        ckpt = torch.load(runs / run["checkpoint"], map_location="cpu")
        members, unmapped = coarse_members(ckpt["classes"])
        cols = [wc_classes.index(k) for k in members]
        y_wc = y_wc_all[:, cols]
        model = lc.build_model(ckpt["in_channels"], len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"]); model.to(device)
        arrays = {k: raw[k] for k in ckpt["inputs"]}
        logits = lc.predict(model, arrays, ckpt["preprocess_stats"], np.arange(len(meta)), device, args.batch_size)
        np.save(Path(args.output).with_name(f"wc-{mode}-logits.npy"), logits)
        scores = to_coarse(logits, members)
        # In-domain anchor: the same model's saved BEN test logits/labels, collapsed to the same coarse classes.
        ben_logits = np.load(runs / run["files"]["test_logits"])
        ben_y19 = np.load(runs / run["files"]["test_labels"])
        ben_y = np.stack([ben_y19[:, idx].max(1) for idx in members.values()], 1).astype(np.float32)
        res = {"coarse_classes": list(members), "unmapped_ben_classes": unmapped, "ben_test_macro_map_19": run["macro_map"],
               "best_epoch": run["best_epoch"], "groups": {}}
        boots = {}
        res["groups"]["ben_test"], boots["ben_test"] = group_metrics(ben_y, to_coarse(ben_logits, members))
        for name, mask in [("india", groups == "india"), ("europe", groups == "europe")] + [(r, regions == r) for r in sorted(set(regions))]:
            res["groups"][name], boots[name] = group_metrics(y_wc[mask], scores[mask])
        # Differences use ONE class set shared by the three pooled groups (>= MIN_POSITIVES in each), so they compare
        # the same quantity. Unpaired bootstrap: the groups are independent samples.
        pooled = {"ben_test": (ben_y, to_coarse(ben_logits, members)), "india": (y_wc[groups == "india"], scores[groups == "india"]),
                  "europe": (y_wc[groups == "europe"], scores[groups == "europe"])}
        common = np.all([yy.sum(0) >= MIN_POSITIVES for yy, _ in pooled.values()], axis=0)
        res["common_classes"] = [k for k, c in zip(members, common) if c]
        res["groups_common"], cboots = {}, {}
        for g, (yy, ss) in pooled.items():
            res["groups_common"][g], cboots[g] = group_metrics(yy[:, common], ss[:, common])
        res["differences"] = {}
        for a, b in (("europe", "ben_test"), ("india", "europe"), ("india", "ben_test")):
            res["differences"][f"{a}-{b}"] = {"diff": res["groups_common"][a]["macro_map"] - res["groups_common"][b]["macro_map"],
                                              "ci95": lc.ci95(cboots[a] - cboots[b]), "classes": res["common_classes"],
                                              "note": "unpaired bootstrap on the common class set; class prevalence still differs"}
        res["level_shift"] = level_shift(arrays, ckpt["preprocess_stats"], regions)
        out["modes"][mode] = res
        print(f"[{mode}] BEN test coarse mAP {res['groups']['ben_test']['macro_map']:.4f} | Europe-WC "
              f"{res['groups']['europe']['macro_map']:.4f} | India-WC {res['groups']['india']['macro_map']:.4f}", flush=True)
    out["runtime_s"] = time.time() - t0
    out["code"] = {"git_sha": lc.git_sha(), "transfer_py_sha256": lc.sha256(Path(__file__)),
                   "landcover_py_sha256": lc.sha256(Path(lc.__file__))}
    out["torch_version"], out["device"] = torch.__version__, str(device)
    Path(args.output).write_text(json.dumps(_finite(out), indent=2, allow_nan=False) + "\n")
    return out


def _finite(o):
    """NaN (e.g. mAP of a group with no positives) -> None so the JSON is strict."""
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summary(out):
    lines = []
    for mode, res in out["modes"].items():
        lines.append(f"== {mode} (coarse classes {res['coarse_classes']}; unmapped {res['unmapped_ben_classes']})")
        lines.append(f"{'group':<12} {'n':>5} {'mAP':>7} {'CI95':>17} {'prev':>7} {'shuffled':>9} classes_used")
        for g, m in res["groups"].items():
            lines.append(f"{g:<12} {m['n']:>5} {m['macro_map']:7.4f} [{m['macro_map_ci95'][0]:.4f}, {m['macro_map_ci95'][1]:.4f}] "
                         f"{m['prevalence_macro_map']:7.4f} {m['shuffled_macro_map']:9.4f} {[res['coarse_classes'][i] for i in m['classes_used']]}")
        lines.append(f"  common class set for differences: {res['common_classes']}")
        for g, m in res["groups_common"].items():
            lines.append(f"  common {g:<9} mAP {m['macro_map']:.4f} [{m['macro_map_ci95'][0]:.4f}, {m['macro_map_ci95'][1]:.4f}] "
                         f"prev {m['prevalence_macro_map']:.4f} shuffled {m['shuffled_macro_map']:.4f}")
        for k, d in res["differences"].items():
            lines.append(f"  {k}: {d['diff']:+.4f} CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    shift = {(r["region"], r["channel"]): r for res in out["modes"].values() for r in res["level_shift"]}
    lines.append("level shift (region mean - BEN train mean) / BEN std; * = |shift| > 1")
    chans = sorted({c for _, c in shift}, key=["VV", "VH", "B02", "B03", "B04", "B08"].index)
    lines.append(f"{'region':<12} " + " ".join(f"{c:>7}" for c in chans))
    for region in out["regions"]:
        cells = [shift.get((region, c)) for c in chans]
        lines.append(f"{region:<12} " + " ".join(f"{r['shift_in_ben_std']:+6.2f}{'*' if r['flag_gt_1std'] else ' '}" if r else "      -" for r in cells))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wc-pack", required=True); p.add_argument("--runs", required=True)
    p.add_argument("--modes", nargs="+", default=["s1", "s2", "s1s2"], choices=["s1", "s2", "s1s2"])
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = evaluate(args)
    print(summary(out))
    return out


if __name__ == "__main__":
    main()
