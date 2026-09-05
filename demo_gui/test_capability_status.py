"""Verify the compact capability maturity indicator stays honest."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "demo_gui/app.py"

AVAILABLE = (
    "Single-image VQA",
    "Resolution robustness evaluation",
    "Execution audit trace",
    "SAR preprocessing / analyst validation",
)
IN_DEVELOPMENT = (
    "Grounding",
    "Bi-temporal Change-VQA",
    "Optical-SAR fusion",
    "RS fine-tuning",
)


class CapabilityStatusTests(unittest.TestCase):
    def test_capabilities_render_in_distinct_maturity_groups(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertIn("CAPABILITY STATUS", [item.value for item in app.caption])

        # Header columns precede the two capability-status columns.
        available_column, development_column = app.get("column")[2:4]
        self.assertIn(
            "**AVAILABLE NOW**", [item.value for item in available_column.markdown]
        )
        self.assertIn(
            "**IN DEVELOPMENT**",
            [item.value for item in development_column.markdown],
        )

        available_text = " ".join(item.value for item in available_column.caption)
        development_text = " ".join(item.value for item in development_column.caption)
        self.assertEqual(available_text, " · ".join(AVAILABLE))
        self.assertEqual(development_text, " · ".join(IN_DEVELOPMENT))
        for item in IN_DEVELOPMENT:
            self.assertNotIn(item, available_text)
        for readiness_label in ("AVAILABLE", "OPERATIONAL", "READY"):
            self.assertNotIn(readiness_label, development_text.upper())


if __name__ == "__main__":
    unittest.main()
