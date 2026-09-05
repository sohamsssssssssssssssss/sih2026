"""Check robustness presentation against the committed, rescored artifact."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class RobustnessTabTests(unittest.TestCase):
    def test_chart_and_table_show_exact_metrics_and_artifact_flags(self) -> None:
        report = json.loads(
            (ROOT / "results/qwen2.5vl-3b__ladder__rescored__20260904.json").read_text()
        )
        rungs = sorted(report["per_rung"], key=float)
        degenerate = set(report["degenerate_rungs"])
        with patch("streamlit.pyplot", wraps=st.pyplot) as pyplot:
            app = AppTest.from_file(str(ROOT / "demo_gui/app.py")).run(timeout=30)
        self.assertFalse(list(app.exception))
        tab = next(t for t in app.tabs if t.label == "Resolution robustness")
        table = tab.dataframe[0].value
        self.assertEqual(table["GSD (m)"].tolist(), [float(r) for r in rungs])
        for column, field in (
            ("Open-question accuracy", "open_accuracy"),
            ("Accuracy", "accuracy"),
            ("Binary predicted-yes rate", "pred_yes_rate_on_binary"),
        ):
            self.assertEqual(table[column].tolist(), [report["per_rung"][r][field] for r in rungs])
        self.assertEqual(
            table["Status"].tolist(),
            ["⚠ Degenerate" if r in degenerate else "Not flagged" for r in rungs],
        )
        self.assertEqual(
            [w.value for w in tab.warning],
            [report["per_rung"][r]["warning"] for r in report["degenerate_rungs"]],
        )

        pyplot.assert_called_once()
        axis = pyplot.call_args.args[0].axes[0]
        primary, aggregate = axis.lines
        self.assertEqual(primary.get_label(), "Open-question accuracy")
        self.assertGreater(primary.get_linewidth(), aggregate.get_linewidth())
        self.assertEqual(aggregate.get_linestyle(), "--")
        for line, field in ((primary, "open_accuracy"), (aggregate, "accuracy")):
            self.assertEqual(list(line.get_xdata()), [float(r) for r in rungs])
            self.assertEqual(list(line.get_ydata()), [report["per_rung"][r][field] for r in rungs])
        self.assertEqual(
            [tick.get_text() for tick in axis.get_xticklabels()],
            [f"{float(r):g}{'*' if r in degenerate else ''}" for r in rungs],
        )
        self.assertEqual(len(axis.collections), 2)
        for markers, field in zip(axis.collections, ("open_accuracy", "accuracy")):
            self.assertEqual(
                markers.get_offsets().tolist(),
                [[float(r), report["per_rung"][r][field]] for r in rungs if r in degenerate],
            )


if __name__ == "__main__":
    unittest.main()
