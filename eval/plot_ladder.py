"""Plot one or more resolution-ladder result files."""

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
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for result_path in args.results:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report.get("suite") != "ladder":
            raise ValueError(f"Not a ladder report: {result_path}")
        points = sorted((float(gsd), values["accuracy"]) for gsd, values in report["per_rung"].items())
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=report["model"],
        )
    axis.set_xscale("log")
    axis.set_xticks([0.3, 1.0, 2.0, 5.0, 10.0], labels=["0.3", "1", "2", "5", "10"])
    axis.set_xlabel("Ground sample distance (m/pixel)")
    axis.set_ylabel("Exact-match accuracy")
    axis.set_ylim(0, 1)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
