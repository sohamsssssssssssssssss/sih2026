"""BigEarthNet pilot multi-label land-cover baseline: radar (S1) vs optical (S2) vs both.

Measurement only. Trains a from-scratch ResNet-18 per mode, picks the epoch by VALIDATION macro mAP,
evaluates TEST exactly once, and reports metrics computed from the saved test logits.

  python ben/landcover.py --pack DIR --mode s1 --control shuffle --out-dir OUT
  python ben/landcover.py --compare OUT/ben-landcover-s1s2.json OUT/ben-landcover-s2.json --output diff.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import torch
from torch import nn
import torchvision

SEED = 17
BOOTSTRAP = 1000
MODES = {"s1": ("s1",), "s2": ("s2",), "s1s2": ("s1", "s2")}
SPLITS = ("train", "validation", "test")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(pack):
    """Every .npy/.json in MANIFEST.json (plus selection.parquet) must match its sha256, else stop."""
    manifest = pack / "MANIFEST.json"
    entries = {e["file"]: e["sha256"] for e in json.loads(manifest.read_text())}
    needed = [f for f in entries if f.endswith((".npy", ".json"))] + ["selection.parquet"]
    for name in ("s1.npy", "s2_10m.npy", "labels_multihot.npy", "classes.json", "selection.parquet"):
        if name not in entries:
            raise ValueError(f"STOP: {name} missing from MANIFEST.json")
    for name in needed:
        actual = sha256(pack / name)
        if actual != entries[name]:
            raise ValueError(f"STOP: sha256 mismatch for {name}: {actual} != {entries[name]}")
    return sha256(manifest)


def load_pack(pack, mode):
    sel = pd.read_parquet(pack / "selection.parquet", columns=["idx", "patch_id", "split"])
    if list(sel.idx) != list(range(len(sel))):
        raise ValueError("STOP: selection.parquet idx is not 0..N-1 in order")
    splits = {s: sel.index[sel.split == s].to_numpy() for s in SPLITS}
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError(f"STOP: empty split in {dict((k, len(v)) for k, v in splits.items())}")
    arrays = {"s1": np.load(pack / "s1.npy", mmap_mode="r"), "s2": np.load(pack / "s2_10m.npy", mmap_mode="r")}
    arrays = {k: np.ascontiguousarray(arrays[k]) for k in MODES[mode]}  # pack is ~3.5 GB raw; fits Kaggle RAM
    labels = np.load(pack / "labels_multihot.npy")
    classes = json.loads((pack / "classes.json").read_text())
    n = len(sel)
    for k, a in arrays.items():
        if a.shape[0] != n or a.shape[2:] != (120, 120):
            raise ValueError(f"STOP: {k} shape {a.shape} does not match N={n}")
    if labels.shape != (n, len(classes)):
        raise ValueError(f"STOP: labels shape {labels.shape} != ({n}, {len(classes)})")
    return arrays, labels.astype(np.float32), classes, splits, sel


def train_stats(arrays, train_idx):
    """Per-channel stats from TRAIN only. S1: 0.5/99.5 percentile clip then mean/std. S2: /10000 then mean/std."""
    stats = {}
    if "s1" in arrays:
        x = arrays["s1"][train_idx].astype(np.float32)
        lo = np.percentile(x, 0.5, axis=(0, 2, 3)); hi = np.percentile(x, 99.5, axis=(0, 2, 3))
        x = np.clip(x, lo[None, :, None, None], hi[None, :, None, None])
        stats["s1"] = {"clip_lo": lo.tolist(), "clip_hi": hi.tolist(),
                       "mean": x.mean(axis=(0, 2, 3), dtype=np.float64).tolist(), "std": x.std(axis=(0, 2, 3), dtype=np.float64).tolist()}
    if "s2" in arrays:
        x = arrays["s2"][train_idx].astype(np.float32) / 10000
        stats["s2"] = {"scale": 1 / 10000, "mean": x.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
                       "std": x.std(axis=(0, 2, 3), dtype=np.float64).tolist()}
    return stats


def make_batch(arrays, stats, idx, device):
    parts = []
    for key in ("s1", "s2"):
        if key not in arrays:
            continue
        x = torch.from_numpy(arrays[key][idx].astype(np.float32)).to(device)
        s = stats[key]
        col = lambda v: torch.tensor(v, dtype=torch.float32, device=device)[None, :, None, None]
        if key == "s1":
            x = torch.maximum(torch.minimum(x, col(s["clip_hi"])), col(s["clip_lo"]))
        else:
            x = x * s["scale"]
        parts.append((x - col(s["mean"])) / col(s["std"]))
    return torch.cat(parts, 1)


def augment(x, gen):
    """Per-sample random flips and 90-degree rotations (the 8 dihedral transforms)."""
    k = torch.randint(0, 8, (x.shape[0],), generator=gen).to(x.device)
    out = x.clone()
    for t in range(8):
        m = k == t
        if m.any():
            y = torch.rot90(x[m], t % 4, (2, 3))
            out[m] = torch.flip(y, (3,)) if t >= 4 else y
    return out


def build_model(in_ch, n_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


@torch.no_grad()
def predict(model, arrays, stats, idx, device, batch_size):
    model.eval()
    out = []
    for i in range(0, len(idx), batch_size):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(make_batch(arrays, stats, idx[i:i + batch_size], device)).float().cpu())
    return torch.cat(out).numpy()


def macro_map(y, scores):
    keep = y.sum(0) > 0
    per_class = np.full(y.shape[1], np.nan)
    for c in np.flatnonzero(keep):
        per_class[c] = average_precision_score(y[:, c], scores[:, c])
    return float(np.nanmean(per_class)), int((~keep).sum()), per_class


def val_thresholds(y_val, p_val):
    """Per-class probability threshold maximising validation F1; 0.5 for classes with no val positives."""
    th = np.full(y_val.shape[1], 0.5)
    for c in range(y_val.shape[1]):
        if y_val[:, c].sum() == 0:
            continue
        prec, rec, t = precision_recall_curve(y_val[:, c], p_val[:, c])
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        th[c] = float(t[int(np.argmax(f1))])
    return th


def bootstrap_maps(y, scores, n_test, reps=None):
    rng, reps = np.random.default_rng(SEED), reps or BOOTSTRAP
    return np.array([macro_map(y[s], scores[s])[0] for s in (rng.integers(0, n_test, n_test) for _ in range(reps))])


def ci95(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log(progress, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if progress:
        with open(progress, "a") as fh:
            fh.write(line + "\n")


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False


def run(args):
    t0 = time.time()
    seed_all()
    pack, out_dir = Path(args.pack), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = verify_manifest(pack)
    arrays, labels, classes, splits, sel = load_pack(pack, args.mode)
    tr, va, te = (splits[s] for s in SPLITS)
    stats = train_stats(arrays, tr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = sum(arrays[k].shape[1] for k in arrays)
    model = build_model(in_ch, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    rng, gen = np.random.default_rng(SEED), torch.Generator().manual_seed(SEED)
    y_tr, y_va, y_te = labels[tr], labels[va], labels[te]
    best = {"map": -1.0, "epoch": None, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = tr[rng.permutation(len(tr))]
        total, seen = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i:i + args.batch_size])
            x = augment(make_batch(arrays, stats, idx, device), gen)
            y = torch.from_numpy(labels[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.detach()) * len(idx); seen += len(idx)
        sched.step()
        val_map = macro_map(y_va, predict(model, arrays, stats, va, device, args.batch_size))[0]
        history.append({"epoch": epoch, "train_loss": total / seen, "val_macro_map": val_map})
        if val_map > best["map"]:
            best = {"map": val_map, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        log(args.progress, f"[{args.mode}] epoch {epoch}/{args.epochs} loss {total / seen:.4f} val mAP {val_map:.4f} "
                           f"best {best['map']:.4f}@{best['epoch']} elapsed {time.time() - t0:.0f}s")
    model.load_state_dict(best["state"])
    val_logits = predict(model, arrays, stats, va, device, args.batch_size)
    test_logits = predict(model, arrays, stats, te, device, args.batch_size)  # the only TEST evaluation
    test_evaluations = 1
    stem = f"ben-landcover-{args.mode}"
    files = {"test_logits": f"{stem}-test-logits.npy", "test_idx": f"{stem}-test-idx.npy", "test_labels": f"{stem}-test-labels.npy"}
    np.save(out_dir / files["test_logits"], test_logits); np.save(out_dir / files["test_idx"], te); np.save(out_dir / files["test_labels"], y_te.astype(np.uint8))
    # Metrics below are computed from the saved files, not the in-memory arrays.
    test_logits, te_saved, y_te = (np.load(out_dir / files[k]) for k in ("test_logits", "test_idx", "test_labels"))
    y_te = y_te.astype(np.float32)
    report = metrics(y_te, test_logits, y_va, val_logits, labels[tr], classes)
    report["bootstrap"] = {"reps": BOOTSTRAP, "seed": SEED, "unit": "test patch", "macro_map_ci95": ci95(bootstrap_maps(y_te, test_logits, len(y_te)))}
    if args.control == "shuffle":
        perm = np.random.default_rng(SEED).permutation(len(y_te))
        shuffled = macro_map(y_te, test_logits[perm])[0]
        report["control"] = {"kind": "shuffle test images vs labels (same trained model)", "seed": SEED,
                             "macro_map": shuffled, "prior_baseline_macro_map": report["prior_baseline"]["macro_map"],
                             "model_macro_map": report["macro_map"]}
    else:
        report["control"] = None
    result = {"mode": args.mode, "seed": SEED,
              "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                         "optimizer": "AdamW", "schedule": "cosine", "loss": "BCEWithLogits", "model": "resnet18 weights=None",
                         "in_channels": in_ch, "augmentation": "per-sample flips + 90-degree rotations", "amp": device.type == "cuda",
                         "channels": {"s1": ["VV", "VH"], "s2": ["B02", "B03", "B04", "B08"]}, "inputs": list(arrays)},
              "preprocess_stats": stats, "epochs_run": len(history), "best_epoch": best["epoch"], "best_val_macro_map": best["map"],
              "history": history, "n": {"train": len(tr), "validation": len(va), "test": len(te)}, "test_evaluations": test_evaluations,
              "classes": classes, **report, "manifest_sha256": manifest_sha, "pack": str(pack),
              "code": {"git_sha": git_sha(), "landcover_py_sha256": sha256(Path(__file__))}, "files": files,
              "runtime_s": time.time() - t0, "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
              "device": str(device) + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
              "python": platform.python_version()}
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    log(args.progress, f"[{args.mode}] DONE test macro mAP {report['macro_map']:.4f} CI95 {report['bootstrap']['macro_map_ci95']} "
                       f"prior {report['prior_baseline']['macro_map']:.4f} control {report['control'] and report['control']['macro_map']}")
    return result


def metrics(y_te, test_logits, y_va, val_logits, y_tr, classes):
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    m, skipped, per_class = macro_map(y_te, test_logits)
    th = val_thresholds(y_va, sig(val_logits))
    pred = (sig(test_logits) >= th).astype(int)
    prior = np.broadcast_to(y_tr.mean(0), y_te.shape)
    pm, pskipped, pper = macro_map(y_te, prior)
    nan_none = lambda v: None if np.isnan(v) else float(v)
    return {"macro_map": m, "classes_skipped_no_test_positive": skipped,
            "micro_f1": float(f1_score(y_te, pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "thresholds_from": "validation, per class, max F1",
            "per_class": [{"class": c, "ap": nan_none(a), "prior_ap": nan_none(p), "threshold": float(t),
                           "test_positives": int(y_te[:, i].sum()), "train_prevalence": float(y_tr[:, i].mean())}
                          for i, (c, a, p, t) in enumerate(zip(classes, per_class, pper, th))],
            "prior_baseline": {"score": "train prevalence per class", "macro_map": pm, "classes_skipped": pskipped}}


def load_run(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    arr = {k: np.load(path.parent / v) for k, v in meta["files"].items()}
    return meta, arr


def compare(a_path, b_path):
    """Paired bootstrap of macro mAP(a) - macro mAP(b) on identical test resamples, from saved logits."""
    (ma, a), (mb, b) = load_run(a_path), load_run(b_path)
    if not np.array_equal(a["test_idx"], b["test_idx"]) or not np.array_equal(a["test_labels"], b["test_labels"]):
        raise ValueError("STOP: runs were evaluated on different test patches/labels")
    if ma["manifest_sha256"] != mb["manifest_sha256"]:
        raise ValueError("STOP: runs used different packs")
    y = a["test_labels"].astype(np.float32)
    da, db = bootstrap_maps(y, a["test_logits"], len(y)), bootstrap_maps(y, b["test_logits"], len(y))
    diff = da - db
    return {"a": ma["mode"], "b": mb["mode"], "metric": "test macro mAP", "a_macro_map": macro_map(y, a["test_logits"])[0],
            "b_macro_map": macro_map(y, b["test_logits"])[0],
            "diff": macro_map(y, a["test_logits"])[0] - macro_map(y, b["test_logits"])[0],
            "diff_ci95": ci95(diff), "share_resamples_diff_le_0": float((diff <= 0).mean()),
            "reps": BOOTSTRAP, "seed": SEED, "unit": "test patch (paired)", "manifest_sha256": ma["manifest_sha256"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack"); p.add_argument("--mode", choices=sorted(MODES))
    p.add_argument("--control", choices=("shuffle", "none"), default="none")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--out-dir", default="."); p.add_argument("--progress")
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON")); p.add_argument("--output")
    args = p.parse_args(argv)
    if args.compare:
        result = compare(*args.compare)
        text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(text + "\n")
        print(text)
        return result
    if not args.pack or not args.mode:
        p.error("--pack and --mode are required unless --compare is given")
    return run(args)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        if str(exc).startswith("STOP"):
            sys.exit(str(exc))
        raise
