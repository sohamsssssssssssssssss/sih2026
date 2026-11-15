"""Analyse the DIOR val detector-count evaluation (stdlib only).

Usage: python3.11 analyze_count_dior.py [evaluation.json] [--json out.json]
"""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
CONFIGURED = "0.35"
SEED, REPS = 17, 2000


def stats(rows):
    errors = [abs(r["prediction"] - r["expected"]) for r in rows]
    n = len(errors)
    return {"n": n, "exact": sum(e == 0 for e in errors) / n, "mae": sum(errors) / n,
            "within1": sum(e <= 1 for e in errors) / n}


def gt_split(rows):
    absent = [r for r in rows if r["expected"] == 0]
    present = [r for r in rows if r["expected"] > 0]
    return {"gt0": {"n": len(absent), "false_positive_n": sum(r["prediction"] != 0 for r in absent),
                    "false_positive_rate": sum(r["prediction"] != 0 for r in absent) / len(absent)},
            "gt_pos": {**stats(present), "over_n": sum(r["prediction"] > r["expected"] for r in present),
                       "under_n": sum(r["prediction"] < r["expected"] for r in present),
                       "pred0_n": sum(r["prediction"] == 0 for r in present)}}


def per_class(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    return {c: {**stats(rs), "gt0_fp_rate": gt_split(rs)["gt0"]["false_positive_rate"]} for c, rs in sorted(by.items())}


def bootstrap(rows):
    """Paired bootstrap over image_ids: detector@0.35 minus always-0 on exact and MAE."""
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image_id"]].append(r)
    images = sorted(by_image)
    rng = random.Random(SEED)
    exact_d, mae_d = [], []
    for _ in range(REPS):
        sample = [r for image in rng.choices(images, k=len(images)) for r in by_image[image]]
        det, zero = stats(sample), stats([{**r, "prediction": 0} for r in sample])
        exact_d.append(det["exact"] - zero["exact"])
        mae_d.append(det["mae"] - zero["mae"])

    def ci(values):
        values = sorted(values)
        return [values[int(0.025 * REPS)], values[int(0.975 * REPS) - 1]]
    point_det, point_zero = stats(rows), stats([{**r, "prediction": 0} for r in rows])
    return {"reps": REPS, "seed": SEED, "unit": "image_id", "n_images": len(images),
            "exact_diff": point_det["exact"] - point_zero["exact"], "exact_diff_ci95": ci(exact_d),
            "mae_diff": point_det["mae"] - point_zero["mae"], "mae_diff_ci95": ci(mae_d)}


def analyse(evaluation):
    sweep = evaluation["threshold_sweep"]
    rows = sweep[CONFIGURED]["records"]
    zero_rows = [{**r, "prediction": 0} for r in rows]
    # Train modes are not in the report, so the majority baseline is read from the evaluator's own summary.
    baselines = {"always_zero": {**stats(zero_rows), **gt_split(zero_rows)},
                 "train_majority_count": {k: evaluation["baselines"]["train_majority_count"]["overall"][k]
                                          for k in ("n", "accuracy", "mean_absolute_error", "within_one")}}
    dropped = {split: {"n_requested": d["n_requested"], "n_kept": d["n_kept"],
                       "n_dropped_by_reason": d["n_dropped_by_reason"],
                       "ids": [x["id"] for x in d["invalid_annotation"]] + d["missing_xml"] + d["missing_image"],
                       "sizes": sorted({f'{x["detail"]["width"]}x{x["detail"]["height"]}'
                                        for x in d["invalid_annotation"] if x["reason"] == "nonstandard_size"})}
               for split, d in evaluation["dropped_before_sampling"].items()}
    return {"thresholds": {t: {**stats(sweep[t]["records"]), **gt_split(sweep[t]["records"])} for t in sorted(sweep)},
            "baselines": baselines, "per_class_at_0.35": per_class(rows), "bootstrap_0.35_vs_always_zero": bootstrap(rows),
            "dropped_before_sampling": dropped}


def report(a):
    out = ["Threshold  exact  MAE    w1     | GT0 FP        | GT>0 exact  over  under  pred0"]
    for t, s in a["thresholds"].items():
        g0, gp = s["gt0"], s["gt_pos"]
        out.append(f"{t:<10} {s['exact']:.3f}  {s['mae']:.3f}  {s['within1']:.3f}  | {g0['false_positive_n']:>3}/{g0['n']} "
                   f"({g0['false_positive_rate']:.0%}) | {gp['exact']:.3f}       {gp['over_n']:>4}  {gp['under_n']:>5}  {gp['pred0_n']:>5}")
    z, m = a["baselines"]["always_zero"], a["baselines"]["train_majority_count"]
    out.append(f"always_zero          exact {z['exact']:.3f}  MAE {z['mae']:.3f}  w1 {z['within1']:.3f}")
    out.append(f"train_majority_count exact {m['accuracy']:.3f}  MAE {m['mean_absolute_error']:.3f}  w1 {m['within_one']:.3f}")
    out.append("\nPer class @0.35       n   exact  MAE    w1     GT0-FP")
    for c, s in a["per_class_at_0.35"].items():
        out.append(f"{c:<20} {s['n']:>3}  {s['exact']:.3f}  {s['mae']:.3f}  {s['within1']:.3f}  {s['gt0_fp_rate']:.2f}")
    b = a["bootstrap_0.35_vs_always_zero"]
    out.append(f"\nPaired bootstrap ({b['reps']} reps, seed {b['seed']}, {b['n_images']} image_ids), detector@0.35 - always_zero:")
    out.append(f"  exact {b['exact_diff']:+.3f}  CI95 [{b['exact_diff_ci95'][0]:+.3f}, {b['exact_diff_ci95'][1]:+.3f}]")
    out.append(f"  MAE   {b['mae_diff']:+.3f}  CI95 [{b['mae_diff_ci95'][0]:+.3f}, {b['mae_diff_ci95'][1]:+.3f}]")
    out.append("\nDropped before sampling:")
    for split, d in a["dropped_before_sampling"].items():
        out.append(f"  {split}: {d['n_requested']} requested, {d['n_kept']} kept, {d['n_dropped_by_reason']}, "
                   f"sizes {d['sizes']}, ids {d['ids']}")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        out = Path(args.pop(args.index("--json") + 1))
        args.remove("--json")
    evaluation = json.loads(Path(args[0] if args else HERE / "dior-count-val-evaluation.json").read_text())
    analysis = analyse(evaluation)
    print(report(analysis))
    if out:
        out.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
