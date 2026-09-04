"""Plot one or more resolution-ladder result files.

Plots open-ended and binary accuracy as separate curves, because the aggregate
of the two is misleading on this suite. A model that collapses to always-"yes"
at coarse GSD scores the binary base rate, which pulls the aggregate curve UP
at low resolution and reads as the model seeing better through blur. Rungs the
degeneracy guard flags are overlaid with hollow markers and a shaded band, so a
non-measurement never looks like a measurement.

Colour distinguishes models; marker and line style distinguish the three
series, so several result files can be overlaid and still be told apart. Rungs
with n=0 are omitted rather than drawn as 0.0, because "no data" and "scored
zero" must not render identically. Falls back to a single aggregate curve for
result files written before the per-rung guard existed.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sih26167-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sih26167-cache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ladder_curve.png")
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(8, 5))
    # Marker/linestyle encode the series; colour encodes the model, so multiple
    # result files overlay legibly instead of collapsing to one palette.
    series = (
        ("open_accuracy", "open-ended", "o", "-"),
        ("binary_accuracy", "binary (yes/no)", "s", "--"),
        ("accuracy", "aggregate", "^", ":"),
    )
    model_colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    flagged_any = False
    for index, result_path in enumerate(args.results):
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report.get("suite") != "ladder":
            raise ValueError(f"Not a ladder report: {result_path}")
        per_rung = report["per_rung"]
        colour = model_colours[index % len(model_colours)]
        model = report.get("model", result_path.stem)

        # Omit empty rungs entirely; a missing measurement is not a zero.
        rungs = sorted(float(g) for g, v in per_rung.items() if v.get("n", 0) > 0)
        if not rungs:
            raise ValueError(f"No populated rungs in {result_path}")
        skipped = len(per_rung) - len(rungs)
        if skipped:
            print(f"  note: omitted {skipped} empty rung(s) from {result_path.name}")

        # Judge stratification only on rungs that carry data — an empty rung
        # writes just {accuracy, n} and must not veto the stratified view.
        stratified = all("open_accuracy" in per_rung[f"{g:.1f}"] for g in rungs)

        for key, label, marker, style in series:
            if not stratified and key != "accuracy":
                continue
            ys = [per_rung[f"{g:.1f}"][key] for g in rungs]
            axis.plot(rungs, ys, marker=marker, linestyle=style, color=colour,
                      label=f"{model} — {label}" if stratified else model,
                      zorder=3, markersize=7)

        flagged = [g for g in rungs if per_rung[f"{g:.1f}"].get("warning")]
        if flagged:
            flagged_any = True
            for key, _label, marker, _style in series:
                if not stratified and key != "accuracy":
                    continue
                axis.plot(flagged, [per_rung[f"{g:.1f}"][key] for g in flagged],
                          marker=marker, linestyle="none", markerfacecolor="white",
                          markeredgecolor=colour, markeredgewidth=2, markersize=11,
                          zorder=4)
            for g in flagged:
                axis.axvspan(g * 0.82, g * 1.22, color="#D55E00", alpha=0.07, zorder=0)

    axis.set_xscale("log")
    axis.set_xticks([0.3, 1.0, 2.0, 5.0, 10.0], labels=["0.3", "1", "2", "5", "10"])
    axis.set_xlabel("Ground sample distance (m/pixel)  —  coarser resolution to the right")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.grid(True, which="both", alpha=0.3)
    title = "Resolution ladder: accuracy vs ground sample distance"
    if flagged_any:
        title += "\nhollow markers = degenerate rung (always-yes); not a measurement"
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
