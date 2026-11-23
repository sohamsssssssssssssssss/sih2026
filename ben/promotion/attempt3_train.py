"""Train the fixed attempt-3 folds and final checkpoints. The CLI requires CUDA."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ben import adapt, landcover as lc, transfer as tr

FLOORS = {"s1": 0.689087241566672, "s2": 0.6588597201349471}
AMENDMENT = "09f91e7e80eafa9339560737b2976a978ce18d08"
CHECKPOINTS = {
    "s1": "763c5450da2edc08a2fe2374de7cfcf810f4d5befdff330d0ca0106f3f630bb5",
    "s2": "703824bb60e4d61477897f94969bf1abdbbf7621a6149a7f7b56e2c4eb6c85fa",
}


def indices(meta, held_out):
    rng = np.random.default_rng(lc.SEED)
    train, validation, test = [], [], []
    for region in sorted(meta.region.unique()):
        shuffled = rng.permutation(np.flatnonzero(meta.region.to_numpy() == region))
        if len(shuffled) < 5:
            raise ValueError(f"STOP: too few patches for {region}")
        cut = int(.8 * len(shuffled))
        if region in held_out:
            test.extend(shuffled)
        else:
            train.extend(shuffled[:cut])
            validation.extend(shuffled[cut:])
    if not train or not validation or (held_out and not test):
        raise ValueError("STOP: empty required split")
    return tuple(np.sort(np.asarray(part, dtype=int)) for part in (train, validation, test))


def train_one(checkpoint, arrays, labels, train, validation, test, device, *, epochs=15, batch_size=128):
    members, _ = tr.coarse_members(checkpoint["classes"])
    stats = lc.train_stats(arrays, train)
    if any(not np.isfinite(s["std"]).all() or min(s["std"]) <= 0 for s in stats.values()):
        raise ValueError("STOP: invalid train-only statistics")
    adapt.seed_everything(1)
    model = lc.build_model(checkpoint["in_channels"], len(checkpoint["classes"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    selected = adapt.fit(model, arrays, stats, labels, train, validation, members, "full",
                         epochs, batch_size, device, seed=1)
    output = {key: checkpoint[key] for key in ("mode", "classes", "inputs", "in_channels")}
    output.update(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                  preprocess_stats=stats, best_epoch=selected["best_epoch"], train_seed=1,
                  val_map=selected["best_val_map"])
    if not len(test):
        return output, None
    scores = tr.to_coarse(lc.predict(model, arrays, stats, test, device, batch_size), members)
    return output, scores[:, [list(members).index(c) for c in adapt.COMMON]]


def macro(y, scores):
    keep = y.sum(axis=0) >= tr.MIN_POSITIVES
    if not keep.any():
        return None
    return float(lc.macro_map(y[:, keep], scores[:, keep])[0])


def evaluate_g1(labels, scores, folds, floor):
    pooled = macro(labels, scores)
    return {"pass": pooled is not None and pooled >= floor, "macro_map": pooled, "floor": floor,
            "groups": {f: macro(labels[folds == f], scores[folds == f]) for f in "ABCDE"}}


def load_sources(old, new, declared):
    sources = []
    for name, path in (("old", old), ("new", new)):
        if tr.verify_wc_manifest(path) != declared[name]:
            raise ValueError(f"STOP: {name} WorldCover manifest mismatch")
        raw, yy, meta, info = tr.load_wc(path)
        if list(meta.idx) != list(range(len(meta))):
            raise ValueError(f"STOP: {name} invalid row IDs")
        keep = (meta.group == "india").to_numpy()
        part = meta[keep].copy()
        part["patch_id"] = [f"{name}:{i}" for i in part.idx]
        sources.append((raw, yy, part, info, keep))
    meta = pd.concat([row[2] for row in sources], ignore_index=True)
    coords = ["utm_epsg", "utm_x", "utm_y"]
    if len(meta) != 4800 or meta.duplicated(coords).any() or meta.patch_id.duplicated().any():
        raise ValueError("STOP: wrong patch count or geographic overlap")
    if set(meta.groupby("region").size()) != {300}:
        raise ValueError("STOP: each region needs 300 patches")
    return sources, meta


def run(old, new, runs, groups_path, dest):
    declared = json.loads(groups_path.read_text())
    if declared["status"] != "assigned" or list(declared["groups"]) != list("ABCDE"):
        raise ValueError("STOP: group assignment is missing or incomplete")
    sources, meta = load_sources(old, new, declared["manifest_sha256"])
    groups = {name: set(row["regions"]) for name, row in declared["groups"].items()}
    flat = [r for group in groups.values() for r in group]
    if len(flat) != 16 or len(set(flat)) != 16 or set(flat) != set(meta.region):
        raise ValueError("STOP: group membership differs from pack rows")
    for mode, expected in CHECKPOINTS.items():
        if lc.sha256(runs / f"ben-landcover-{mode}-checkpoint.pt") != expected:
            raise ValueError(f"STOP: unexpected {mode} Europe checkpoint")
    started = time.monotonic()
    dest.mkdir(exist_ok=False, parents=True)
    device = torch.device("cuda")
    report = {}
    for mode, floor in FLOORS.items():
        checkpoint = torch.load(runs / f"ben-landcover-{mode}-checkpoint.pt", map_location="cpu", weights_only=True)
        if checkpoint["mode"] != mode or tuple(checkpoint["inputs"]) != (mode,):
            raise ValueError("STOP: checkpoint mode mismatch")
        members, _ = tr.coarse_members(checkpoint["classes"])
        arrays = {mode: np.concatenate([raw[mode][keep] for raw, _, _, _, keep in sources])}
        labels = np.concatenate([yy[keep][:, [info["coarse_classes"].index(c) for c in members]]
                                 for _, yy, _, info, keep in sources]).astype(np.float32)
        common = [list(members).index(c) for c in adapt.COMMON]
        scores = np.full((len(meta), len(common)), np.nan, dtype=np.float32)
        folds = np.full(len(meta), "", dtype="U1")
        for letter, held_out in [*groups.items(), ("final", set())]:
            train, val, test = indices(meta, held_out)
            print(f"{mode}/{letter}: train={len(train)} val={len(val)} held_out={len(test)}", flush=True)
            weights, predictions = train_one(checkpoint, arrays, labels, train, val, test, device)
            weights.update(group=letter, amendment=AMENDMENT, group_sha256=lc.sha256(groups_path),
                           manifest_sha256=declared["manifest_sha256"], epochs=15,
                           source_checkpoint_sha256=CHECKPOINTS[mode])
            with (dest / f"ben-landcover-india-{mode}-{letter}.pt").open("xb") as stream:
                torch.save(weights, stream)
            if predictions is not None:
                if np.any(folds[test] != ""):
                    raise ValueError("STOP: held-out predictions already exist")
                scores[test], folds[test] = predictions, letter
        if not np.isfinite(scores).all() or np.any(folds == ""):
            raise ValueError("STOP: held-out coverage is incomplete")
        target = labels[:, common].astype(np.uint8)
        with (dest / f"attempt3-{mode}-held-out.npz").open("xb") as stream:
            np.savez_compressed(stream, patch_id=meta.patch_id.to_numpy(dtype=str),
                                region=meta.region.to_numpy(dtype=str), fold=folds,
                                classes=np.array(adapt.COMMON), scores=scores, labels=target)
        report[mode] = evaluate_g1(target, scores, folds, floor)
        print(f"{mode} G1 {report[mode]}", flush=True)
    result = {"amendment": AMENDMENT, "group_sha256": lc.sha256(groups_path),
              "runtime_s": time.monotonic() - started, "G1": report,
              "pass": all(row["pass"] for row in report.values())}
    with (dest / "G1.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "new", "runs", "groups", "dest"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("HARD STOP: this registered job requires CUDA")
    result = run(args.old, args.new, args.runs, args.groups, args.dest)
    if not result["pass"]:
        raise SystemExit("HARD STOP: G1 failed. Outputs retained for review.")


if __name__ == "__main__":
    main()
