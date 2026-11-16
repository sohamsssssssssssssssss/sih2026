"""Recompute the BigEarthNet land-cover baseline results from saved test logits + labels (numpy + sklearn only).

Cross-checks every recomputed number against the run JSONs (tolerance 1e-3) and exits non-zero on any mismatch.
  python3.11 analyze_ben_landcover.py [--json out.json]
"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
MODES = ("s1", "s2", "s1s2")
SEED, REPS, TOL = 17, 1000, 1e-3
UNRELIABLE_POSITIVES = 30
RISING_WINDOW = 5


def per_class_ap(y, s):
    ap = np.full(y.shape[1], np.nan)
    for c in np.flatnonzero(y.sum(0) > 0):
        ap[c] = average_precision_score(y[:, c], s[:, c])
    return ap


def macro_map(y, s):
    return float(np.nanmean(per_class_ap(y, s)))


def resamples(n):
    rng = np.random.default_rng(SEED)  # same generator sequence as ben/landcover.py bootstrap_maps
    return [rng.integers(0, n, n) for _ in range(REPS)]


def ci95(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    runs = {m: json.loads((HERE / f"ben-landcover-{m}.json").read_text()) for m in MODES}
    logits = {m: np.load(HERE / runs[m]["files"]["test_logits"]) for m in MODES}
    labels = {m: np.load(HERE / runs[m]["files"]["test_labels"]).astype(np.float32) for m in MODES}
    idx = {m: np.load(HERE / runs[m]["files"]["test_idx"]) for m in MODES}
    y = labels["s1"]
    checks, failures = [], []

    def check(name, recomputed, recorded):
        ok = abs(recomputed - recorded) <= TOL
        checks.append({"check": name, "recomputed": recomputed, "recorded": recorded, "ok": ok})
        if not ok:
            failures.append(name)

    if not all(np.array_equal(labels[m], y) and np.array_equal(idx[m], idx["s1"]) for m in MODES):
        sys.exit("STOP: modes were evaluated on different test labels/idx")
    classes = runs["s1"]["classes"]
    train_prev = np.array([row["train_prevalence"] for row in runs["s1"]["per_class"]])

    ap = {m: per_class_ap(y, logits[m]) for m in MODES}
    maps = {m: float(np.nanmean(ap[m])) for m in MODES}
    prior = macro_map(y, np.broadcast_to(train_prev, y.shape))
    perm = np.random.default_rng(SEED).permutation(len(y))
    shuffled = {m: macro_map(y, logits[m][perm]) for m in MODES}
    samples = resamples(len(y))
    boot = {m: np.array([macro_map(y[s], logits[m][s]) for s in samples]) for m in MODES}

    for m in MODES:
        check(f"{m} macro mAP", maps[m], runs[m]["macro_map"])
        check(f"{m} prior macro mAP", prior, runs[m]["prior_baseline"]["macro_map"])
        check(f"{m} shuffled macro mAP", shuffled[m], runs[m]["control"]["macro_map"])
        for k, v in zip(("lo", "hi"), ci95(boot[m])):
            check(f"{m} CI95 {k}", v, runs[m]["bootstrap"]["macro_map_ci95"][0 if k == "lo" else 1])
        for c, row in enumerate(runs[m]["per_class"]):
            if row["ap"] is not None:
                check(f"{m} AP {classes[c]}", float(ap[m][c]), row["ap"])
    compares = {}
    for a, b in (("s2", "s1"), ("s1s2", "s2")):
        rec = json.loads((HERE / f"ben-landcover-compare-{a}-vs-{b}.json").read_text())
        diff = maps[a] - maps[b]
        lo, hi = ci95(boot[a] - boot[b])
        check(f"{a}-{b} diff", diff, rec["diff"]); check(f"{a}-{b} CI lo", lo, rec["diff_ci95"][0]); check(f"{a}-{b} CI hi", hi, rec["diff_ci95"][1])
        compares[f"{a}-{b}"] = {"diff": diff, "ci95": [lo, hi], "ci95_recorded": rec["diff_ci95"], "ci_spans_zero": lo <= 0 <= hi,
                                "share_resamples_diff_le_0": float(((boot[a] - boot[b]) <= 0).mean())}

    positives = y.sum(0).astype(int)
    rows = sorted(({"class": classes[c], "test_positives": int(positives[c]), **{m: float(ap[m][c]) for m in MODES},
                    "s2_minus_s1": float(ap["s2"][c] - ap["s1"][c]), "s1s2_minus_s2": float(ap["s1s2"][c] - ap["s2"][c]),
                    "unreliable_lt30_pos": bool(positives[c] < UNRELIABLE_POSITIVES)}
                   for c in range(len(classes)) if positives[c] > 0), key=lambda r: -r["s2_minus_s1"])
    training = {}
    for m in MODES:
        h = runs[m]["history"]
        val = np.array([e["val_macro_map"] for e in h]); ep = np.array([e["epoch"] for e in h])
        last = slice(len(val) - RISING_WINDOW, len(val))
        slope = float(np.polyfit(ep[last], val[last], 1)[0])
        training[m] = {"epochs_run": len(h), "best_epoch": runs[m]["best_epoch"], "best_val_macro_map": runs[m]["best_val_macro_map"],
                       "val_last5": [float(v) for v in val[last]], "val_last5_slope_per_epoch": slope,
                       "val_rising_last5": slope > 0, "runtime_s": runs[m]["runtime_s"]}

    result = {"macro_map": maps, "prior_macro_map": prior, "shuffled_macro_map": shuffled,
              "macro_map_ci95": {m: ci95(boot[m]) for m in MODES}, "compare": compares,
              "classes_skipped_no_test_positive": int((positives == 0).sum()), "n_test": len(y),
              "per_class": rows, "s1_ge_s2": [r["class"] for r in rows if r["s1"] >= r["s2"]],
              "unreliable_classes": [r["class"] for r in rows if r["unreliable_lt30_pos"]], "training": training,
              "cross_check": {"n_checks": len(checks), "failures": failures, "tolerance": TOL}}

    print(f"Test patches {len(y)}; classes with >=1 test positive {int((positives > 0).sum())}/{len(classes)}")
    print(f"{'mode':<5} {'mAP':>7} {'CI95':>17} {'shuffled':>9}   prior {prior:.4f}")
    for m in MODES:
        print(f"{m:<5} {maps[m]:7.4f} [{result['macro_map_ci95'][m][0]:.4f}, {result['macro_map_ci95'][m][1]:.4f}] {shuffled[m]:9.4f}")
    for k, c in compares.items():
        print(f"{k}: {c['diff']:+.4f} CI95 [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] (recorded [{c['ci95_recorded'][0]:+.4f}, {c['ci95_recorded'][1]:+.4f}])")
    print(f"\n{'class':<58} {'pos':>5} {'s1':>6} {'s2':>6} {'s1s2':>6} {'s2-s1':>7}")
    for r in rows:
        print(f"{r['class'][:58]:<58} {r['test_positives']:>5} {r['s1']:6.3f} {r['s2']:6.3f} {r['s1s2']:6.3f} {r['s2_minus_s1']:+7.3f}"
              + ("  <30 pos" if r["unreliable_lt30_pos"] else ""))
    print("\ns1 >= s2:", result["s1_ge_s2"]); print("unreliable (<30 test positives):", result["unreliable_classes"])
    for m, t in training.items():
        print(f"{m}: best epoch {t['best_epoch']}/{t['epochs_run']}, val last5 {[round(v, 4) for v in t['val_last5']]}, "
              f"slope {t['val_last5_slope_per_epoch']:+.5f}/epoch, rising {t['val_rising_last5']}")
    print(f"\ncross-check: {len(checks)} checks, failures {failures}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        sys.exit("STOP: recomputed values differ from recorded JSONs")


if __name__ == "__main__":
    main()
