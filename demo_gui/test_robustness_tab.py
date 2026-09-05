"""Check robustness presentation against the committed, rescored artifact."""

import json
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "demo_gui/app.py"
RESULTS_PATH = ROOT / "results/qwen2.5vl-3b__ladder__rescored__20260904.json"


class RobustnessTabTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()

    def tearDown(self) -> None:
        st.cache_data.clear()

    def test_chart_and_table_show_exact_metrics_and_artifact_flags(self) -> None:
        report = json.loads(RESULTS_PATH.read_text())
        rungs = sorted(report["per_rung"], key=float)
        degenerate = set(report["degenerate_rungs"])
        with patch("streamlit.pyplot", wraps=st.pyplot) as pyplot:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
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

    def test_zero_degenerate_rungs_have_no_flag_markers_or_legend_entry(self) -> None:
        report = copy.deepcopy(json.loads(RESULTS_PATH.read_text()))
        report["degenerate_rungs"] = []
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == RESULTS_PATH:
                return json.dumps(report)
            return original(path, *args, **kwargs)

        with (
            patch.object(Path, "read_text", read_text),
            patch("streamlit.pyplot", wraps=st.pyplot) as pyplot,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

        self.assertFalse(list(app.exception))
        tab = next(t for t in app.tabs if t.label == "Resolution robustness")
        table = tab.dataframe[0].value
        self.assertEqual(table["Status"].tolist(), ["Not flagged"] * len(report["per_rung"]))
        self.assertEqual(list(tab.warning), [])
        self.assertNotIn(
            "* and orange X markers identify degenerate rungs flagged in the committed artifact.",
            [item.value for item in tab.caption],
        )

        axis = pyplot.call_args.args[0].axes[0]
        self.assertEqual(len(axis.collections), 0)
        self.assertNotIn(
            "Flagged degenerate rung",
            [item.get_text() for item in axis.get_legend().get_texts()],
        )
        self.assertTrue(
            all("*" not in tick.get_text() for tick in axis.get_xticklabels())
        )


if __name__ == "__main__":
    unittest.main()
