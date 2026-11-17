"""Generate the T18 report from the aggregate evaluation JSON."""
import argparse
import json
from pathlib import Path

PRESENCE = "change_or_not"
DIRECTIONS = ("increase_or_not", "decrease_or_not")


def fmt(metric, percent=False):
    if metric["value"] is None:
        return f"Insufficient support (n={metric['n']})"
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    lo, hi = (scale * x for x in metric["ci95"])
    return f"{scale * metric['value']:.3f}{suffix} [{lo:.3f}, {hi:.3f}]"


def report(data):
    lines = [
        "# T18 CDVQA image-differencing baseline", "",
        "This report uses the three binary question types in the [preregistration](../../docs/hypotheses/T18.md). Change destination, largest and smallest changed class, "
        "and both semantic change-ratio types are out of scope. Binary CDVQA questions also name classes. The pixel-only arm "
        "uses one image-pair feature vector per type, so it cannot tell which named class a question refers to.", "",
        "Intervals are percentile CI95 from 2,000 image-pair bootstrap replicates with seed 17. Accuracy is question-weighted "
        "within each type. Test1 and test2 share image pairs and should not be treated as independent replications. "
        "The oracle-mask arm uses supplied semantic masks alongside the RGB features.", "",
        "Identical and swapped agreement use the original labels, matching the G3 convention. The identical-no "
        "column instead measures the correct presence answer for the artificial identical pair. For directional "
        "questions, swap flip rate uses only originally correct answers and scores no counterfactual labels. "
        "A class with no net area change can legitimately receive no in both orders.", "",
    ]
    votes = {h: [] for h in ("H1", "H2", "H3 presence", "H3 increase", "H3 decrease")}
    for split, info in data["test"].items():
        lines += [f"## {split}", "",
                  "Full-split results and G3 scores use different rows. Adapter comparisons are confined to the matched "
                  "subtable with the exact frozen IDs, pair IDs, question text, and labels from commit `89fc5bf`.", ""]
        for label, group in (("Full official split", "full"), ("Exact G3 question subset", "g3_subset")):
            lines += [f"### {label}", "",
                      "| Type | Arm | Real accuracy | Identical agreement | Swapped agreement | Identical no | Swap change or correct-answer flip |",
                      "|---|---|---|---|---|---|---|"]
            for kind, cell in info[group].items():
                s = cell["support"]
                type_label = f"{kind} (n={cell['n']}, {cell['pairs']} pairs, {s.get('yes', 0)} yes)"
                for arm, metrics in cell["arms"].items():
                    scores = [fmt(metrics[f"{cond}_accuracy"]) for cond in ("real", "identical", "swapped")]
                    if kind == PRESENCE:
                        scores += [fmt(metrics["identical_no"], True), fmt(metrics["swap_change"], True)]
                    else:
                        flip = metrics["swap_flip_correct"]
                        flip_text = fmt(flip, True)
                        if flip["value"] is not None:
                            flip_text += f" (n={flip['n']})"
                        scores += ["—", flip_text + f"; correct yes={metrics['original_correct_yes']}"]
                    lines.append("| " + " | ".join([type_label, arm, *scores]) + " |")
            lines.append("")
        full = info["full"]
        presence = full[PRESENCE]
        gain = presence["gain_vs_majority"]
        selected = presence["arms"]["diff-baseline"]
        no, change = selected["identical_no"], selected["swap_change"]
        increase, decrease = (full[k]["arms"]["diff-baseline"]["swap_flip_correct"] for k in DIRECTIONS)
        checks = [
            ("H1", "Presence accuracy gain versus train majority", gain, lambda m: m["ci95"][0] > 0, False),
            ("H2", "Presence no on identical pairs (≥95%)", no, lambda m: m["value"] >= .95, True),
            ("H3 presence", "Presence swap changes (<5%)", change, lambda m: m["value"] < .05, True),
            ("H3 increase", "Increase swap flips (>80%)", increase, lambda m: m["value"] > .80, True),
            ("H3 decrease", "Decrease swap flips (>80%)", decrease, lambda m: m["value"] > .80, True),
        ]
        for h, label, m, predicate, percent in checks:
            verdict = "INCONCLUSIVE" if m["value"] is None else "PASS" if predicate(m) else "FAIL"
            votes[h].append(verdict)
            lines.append(f"- **{h} {verdict}.** {label}: {fmt(m, percent)}.")
        lines.append("")

    def combine(items):
        return "FAIL" if "FAIL" in items else "INCONCLUSIVE" if "INCONCLUSIVE" in items else "PASS"

    h1, h2 = (combine(votes[h]) for h in ("H1", "H2"))
    h3 = combine([v for h in votes if h.startswith("H3") for v in votes[h]])
    presence_gate = combine(votes["H1"] + votes["H2"] + votes["H3 presence"])
    lines += ["## Preregistered verdicts", "",
              f"**H1 {h1}. H2 {h2}. H3 {h3}.** H3 requires both directional controls and presence swap stability on both splits.", "",
              f"The presence criteria yield **{presence_gate}** (H1 {h1}, H2 {h2}, presence swap stability "
              f"{combine(votes['H3 presence'])}). "]
    last = lines.pop()
    accuracies = []
    for split, info in data["test"].items():
        scores = info["full"][PRESENCE]["arms"]
        text = ", ".join(f"{arm} {scores[arm]['real_accuracy']['value']:.3f}"
                         for arm in ("diff-baseline", "majority", "question-only"))
        accuracies.append(f"{split}: {text}")
    lines += [last + "On real CDVQA presence questions the accuracies were " + " and ".join(accuracies)
              + ". These criteria measure CDVQA presence performance and image controls. Independent validation on Indian imagery is outside this evaluation.", "",
              "The selected regularization, exact counts, unrounded metrics, G3 hashes and QA hashes are in `summary.json`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    args.report.write_text(report(json.loads(args.summary.read_text())), encoding="utf-8")


if __name__ == "__main__":
    main()
