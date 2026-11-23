"""CPU-only G2 and G4 scoring for saved attempt-3 checkpoints and predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import torch

from ben import adapt, landcover as lc, transfer as tr
from ben.promotion.attempt3_train import AMENDMENT, CHECKPOINTS, macro


def scores_to_logits(scores):
    return logit(np.clip(scores, 1e-6, 1 - 1e-6))


def offsets(scores, labels):
    logits = scores_to_logits(scores)
    result = []
    for x, y in zip(logits.T, labels.T):
        def bce(b):
            return np.mean(np.logaddexp(0, x + b) - y * (x + b))
        fit = minimize_scalar(bce, bounds=(-10, 10), method="bounded")
        if not fit.success:
            raise ValueError("STOP: calibration fit failed")
        result.append(float(fit.x))
    return np.array(result)


def ece(scores, labels):
    # Search the exact registered bin edges; 1 belongs to the final bin.
    bins = np.searchsorted(np.linspace(0, 1, 11), scores, side="right") - 1
    bins = np.minimum(bins, 9)
    values = [np.abs(np.bincount(b, weights=p - y, minlength=10)).sum() / len(b)
              for b, p, y in zip(bins.T, scores.T, labels.T)]
    return float(np.mean(values))


def g2(scores, labels, folds):
    output = {}
    for fold in "ABCDE":
        keep = folds == fold
        if not keep.any():
            raise ValueError(f"STOP: missing held-out observations for {fold}")
        b = offsets(scores[~keep], labels[~keep])
        raw, calibrated = ece(scores[keep], labels[keep]), ece(expit(scores_to_logits(scores[keep]) + b), labels[keep])
        output[fold] = {"raw_macro_ece": raw, "calibrated_macro_ece": calibrated,
                        "offsets": b.tolist(), "pass": calibrated < raw}
    return {"groups": output, "final_offsets": offsets(scores, labels).tolist(),
            "pass": all(row["pass"] for row in output.values())}


def checkpoint_scores(checkpoint, mode, pixels):
    classes = checkpoint["classes"]
    members, _ = tr.coarse_members(classes)
    model = lc.build_model(checkpoint["in_channels"], len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    logits = lc.predict(model, {mode: pixels}, checkpoint["preprocess_stats"],
                        np.arange(len(pixels)), torch.device("cpu"), 64)
    return tr.to_coarse(logits, members)[:, [list(members).index(name) for name in adapt.COMMON]]


def comparison(y, india, europe, folds):
    keep = y.sum(axis=0) >= tr.MIN_POSITIVES
    if not keep.any():
        raise ValueError("STOP: G4 has no eligible pooled classes")
    ys, ii, ee = y[:, keep], india[:, keep], europe[:, keep]
    expected = [float(lc.macro_map(ys, values)[0]) for values in (ii, ee)]
    deltas = []
    rng = np.random.default_rng(17)
    for take in rng.integers(0, len(ys), size=(2000, len(ys))):
        deltas.append(lc.macro_map(ys[take], ii[take])[0] - lc.macro_map(ys[take], ee[take])[0])
    if not np.isfinite(deltas).all():
        raise ValueError("STOP: a G4 bootstrap draw has undefined mAP")
    ci = np.quantile(deltas, [.025, .975]).tolist()
    delta = expected[0] - expected[1]
    groups = {}
    for group in "ABCDE":
        active = folds == group
        a, b = [macro(y[active], values[active]) for values in (india, europe)]
        groups[group] = {"n": int(active.sum()), "india": a, "europe": b,
                         "delta": a - b if a is not None and b is not None else None}
    return {"india": expected[0], "europe": expected[1], "delta": delta, "ci95": ci,
            "valid_draws": len(deltas), "pass": delta > 0 and ci[0] > 0, "groups": groups}


def run(trained, europe, groups_path, pack):
    manifest = json.loads((pack / "MANIFEST.json").read_text())
    group_hash = lc.sha256(groups_path)
    if manifest["group_sha256"] != group_hash or len(manifest["regions"]) != 16:
        raise ValueError("STOP: G4 pack does not match the fold assignment")
    g1 = json.loads((trained / "G1.json").read_text())
    if g1["amendment"] != AMENDMENT or g1["group_sha256"] != group_hash:
        raise ValueError("STOP: training output does not match the G4 pack")
    columns = [manifest["label_classes"].index(c) for c in adapt.COMMON]
    output = {}
    for mode in CHECKPOINTS:
        with np.load(trained / f"attempt3-{mode}-held-out.npz", allow_pickle=False) as result:
            if (len(result["patch_id"]) != 4800 or len(set(result["patch_id"])) != 4800
                    or result["classes"].tolist() != adapt.COMMON):
                raise ValueError("STOP: registered held-out predictions incomplete")
            measurement = g2(result["scores"], result["labels"], result["fold"])
        europa = europe / f"ben-landcover-{mode}-checkpoint.pt"
        if lc.sha256(europa) != CHECKPOINTS[mode]:
            raise ValueError(f"STOP: Europe {mode} checkpoint hash differs")
        baseline = torch.load(europa, map_location="cpu", weights_only=True)
        iy, ey, yy, fold_ids = [], [], [], []
        for fold in "ABCDE":
            adapted = torch.load(trained / f"ben-landcover-india-{mode}-{fold}.pt",
                                 map_location="cpu", weights_only=True)
            if (adapted["group"] != fold or adapted["group_sha256"] != group_hash
                    or adapted["amendment"] != AMENDMENT or adapted["mode"] != mode):
                raise ValueError("STOP: fold checkpoint identity differs")
            for region in manifest["regions"]:
                if region["group"] != fold:
                    continue
                path = pack / region["file"]
                if not region["fetched"] or lc.sha256(path) != region["sha256"]:
                    raise ValueError(f"STOP: G4 inputs missing or changed for {region['region']}")
                with np.load(path, allow_pickle=False) as sample:
                    pixels = sample[mode]
                    iy.append(checkpoint_scores(adapted, mode, pixels))
                    ey.append(checkpoint_scores(baseline, mode, pixels))
                    yy.append(sample["labels"][:, columns])
                    fold_ids.extend([fold] * len(pixels))
        output[mode] = {"G2": measurement,
                        "G4": comparison(np.concatenate(yy), np.concatenate(iy), np.concatenate(ey), np.array(fold_ids))}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("trained", "europe", "groups", "g4_pack", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    scores = run(args.trained, args.europe, args.groups, args.g4_pack)
    with args.output.open("x") as stream:
        stream.write(json.dumps(scores, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
