"""CDVQA G3 analysis from the stored Kaggle result.json files (stdlib only, read-only on inputs).

Run: python3 results/cdvqa-g3/analyze_g3.py   ->  writes results/cdvqa-g3/ANALYSIS.md
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("test1", "test2")
CONDITIONS = ("base_real", "adapter_real", "adapter_identical", "adapter_swapped")
TYPES = ("change_or_not", "increase_or_not", "decrease_or_not", "change_to_what",
         "largest_change", "smallest_change", "change_ratio", "change_ratio_types")


def norm(text):
    return text.strip().casefold()  # same rule as eval/cdvqa_pairs.normalise


def pct(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(item):
    return "n/a" if item.get("ci95") is None else f"[{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]"


def aligned(result, a, b):
    ra, rb = result["conditions"][a]["records"], result["conditions"][b]["records"]
    assert [r["sample_id"] for r in ra] == [r["sample_id"] for r in rb], "condition records not aligned"
    return list(zip(ra, rb))


def split_section(name, result):
    out = [f"## {name} ({result['eval_manifest']}, manifest sha256 `{result['eval_manifest_sha256']}`)", ""]
    conds = result["conditions"]
    records = conds["adapter_real"]["records"]
    if not records or "prediction" not in records[0]:
        return out + ["Predictions are not stored in result.json; items c and d cannot be computed.", ""]

    # a. accuracy
    majority = result["majority_baseline"]
    out += ["### a. Accuracy (macro = mean of the 8 per-type accuracies)", "",
            "| Type | n | Majority | " + " | ".join(CONDITIONS) + " |", "|---|---|---|" + "---|" * len(CONDITIONS)]
    for t in TYPES:
        n = conds["adapter_real"]["summary"]["per_type"][t]["n"]
        cells = [pct(conds[c]["summary"]["per_type"][t]["accuracy"]) for c in CONDITIONS]
        out.append(f"| {t} | {n} | {pct(majority['per_type'][t]['accuracy'])} | " + " | ".join(cells) + " |")
    out.append("| **Macro** | 400 | **" + pct(majority["macro_accuracy"]) + "** | "
               + " | ".join(f"**{pct(conds[c]['summary']['macro_accuracy'])}**" for c in CONDITIONS) + " |")
    out += ["", "Malformed answers: " + ", ".join(f"{c} {conds[c]['summary']['malformed_n']}" for c in CONDITIONS), ""]

    # b. paired differences
    pdiff = result["paired_differences"]
    out += [f"### b. Paired differences ({pdiff['method']}; seed {pdiff['seed']}, {pdiff['repetitions']} reps, "
            f"{pdiff['pair_clusters']} image pairs)", ""]
    for comparison, value in pdiff["comparisons"].items():
        out += [f"**{comparison}**: macro {value['macro']['difference']:+.4f}, CI95 {ci(value['macro'])}", "",
                "| Type | Difference | CI95 |", "|---|---|---|"]
        for t in TYPES:
            item = value["per_type"][t]
            out.append(f"| {t} | {item['difference']:+.3f} | {ci(item)} |")
        out.append("")

    # c. identical pair, change_or_not
    ident = [r for r in conds["adapter_identical"]["records"] if r["question_type"] == "change_or_not"]
    answers = Counter(norm(r["prediction"]) for r in ident)
    truth = Counter(norm(r["expected"]) for r in ident)
    n = len(ident)
    out += ["### c. adapter_identical (before image shown twice), change_or_not", "",
            f"- Questions: {n}",
            f"- Adapter answers: \"no\" {answers['no']}/{n} = {answers['no'] / n:.3f}; "
            f"\"yes\" {answers['yes']}/{n} = {answers['yes'] / n:.3f}"
            + (f"; other {n - answers['no'] - answers['yes']}" if n - answers['no'] - answers['yes'] else ""),
            f"- Ground-truth share of \"no\" for the real pairs: {truth['no']}/{n} = {truth['no'] / n:.3f}",
            "- For an identical pair the correct answer is always \"no\"; the stored `correct` field scores against "
            "the real-pair ground truth instead.", ""]

    # d. swapped vs real
    pairs = aligned(result, "adapter_real", "adapter_swapped")
    out += ["### d. adapter_swapped (after, before) vs adapter_real (before, after), same questions", "",
            "| Type | n | Flipped yes<->no | Unchanged answer |", "|---|---|---|---|"]
    unchanged_total = 0
    for t in TYPES:
        rows = [(norm(a["prediction"]), norm(b["prediction"])) for a, b in pairs if a["question_type"] == t]
        unchanged = sum(x == y for x, y in rows)
        unchanged_total += unchanged
        flip = (f"{sum({x, y} == {'yes', 'no'} for x, y in rows)}/{len(rows)} = "
                f"{sum({x, y} == {'yes', 'no'} for x, y in rows) / len(rows):.3f}"
                if t in ("increase_or_not", "decrease_or_not") else "-")
        out.append(f"| {t} | {len(rows)} | {flip} | {unchanged}/{len(rows)} = {unchanged / len(rows):.3f} |")
    out += [f"| **All 8 types** | {len(pairs)} | - | **{unchanged_total}/{len(pairs)} = "
            f"{unchanged_total / len(pairs):.3f}** |", ""]
    return out


def main():
    lines = ["# CDVQA G3 analysis", "",
             "Generated by `analyze_g3.py` from `test1-result.json` and `test2-result.json` (Kaggle cdvqa-g3-eval V1). "
             "Facts only; interpretation and claim limits are in SOURCE.md.", ""]
    for name in SPLITS:
        result = json.loads((HERE / f"{name}-result.json").read_text(encoding="utf-8"))
        lines += split_section(name, result)
    (HERE / "ANALYSIS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("wrote", HERE / "ANALYSIS.md")


if __name__ == "__main__":
    main()
