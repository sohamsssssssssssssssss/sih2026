"""Verify manual SAR excerpts and display fallbacks without changing annotations."""

import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
from PIL import Image

import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_gui.test_support import image_elements

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = ROOT / "data/sar_gate/annotation_template.md"
IMAGE_PATH = ROOT / "data/sar_gate/rendered/mumbai_coastal.png"


def full_mumbai_section(document: str) -> str:
    section = document.split("## Mumbai coastal", 1)[1].split("## Maharashtra farmland", 1)[0]
    return "\n".join(line for line in section.strip().splitlines() if not line.startswith("![")).strip()


class SARTabTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()
        self.document = ANNOTATION_PATH.read_text()

    def tearDown(self) -> None:
        st.cache_data.clear()

    def sar_tab(self):
        app = AppTest.from_file(str(ROOT / "demo_gui/app.py")).run(timeout=30)
        self.assertFalse(list(app.exception))
        return next(tab for tab in app.tabs if tab.label == "SAR interpretation")

    def test_real_annotation_has_verbatim_summaries_and_full_text(self) -> None:
        tab = self.sar_tab()
        self.assertIn("HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT", [i.value for i in tab.info])
        self.assertEqual(
            [h.value for h in tab.subheader],
            ["Analyst interpretation", "Water", "Built-up", "Vegetation", "Terrain"],
        )
        full = full_mumbai_section(self.document)
        headings = (
            "Water areas:", "Urban/built-up:", "Vegetation:",
            "Terrain artifacts (layover/foreshortening/shadow):", "Why it looks this way:",
        )
        excerpts = [m.value for m in tab.markdown[:4]]
        for index, excerpt in enumerate(excerpts):
            bucket = full.split(headings[index], 1)[1].split(headings[index + 1], 1)[0]
            self.assertTrue(excerpt.strip())
            for line in excerpt.splitlines():
                self.assertIn(line, bucket)
            self.assertNotIn("Reasoning:", excerpt)
            self.assertLess(len(excerpt), len(bucket.strip()))
        self.assertEqual(len(excerpts), 4)
        expander = next(e for e in tab.expander if e.label == "View full analyst annotation")
        self.assertEqual(expander.markdown[0].value, full)
        self.assertNotIn("HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT", [i.value for i in expander.info])

    def test_missing_image_keeps_annotation_available(self) -> None:
        synthetic = Path(tempfile.mktemp(suffix=".png"))
        Image.new("RGB", (100, 100)).save(synthetic)

        original_bytes = IMAGE_PATH.read_bytes() if IMAGE_PATH.is_file() else None
        try:
            synthetic.replace(IMAGE_PATH)
            self.assertEqual(len(image_elements(self.sar_tab())), 1)

            original = Path.is_file
            with patch.object(Path, "is_file", lambda path: False if path == IMAGE_PATH else original(path)):
                tab = self.sar_tab()
        finally:
            if original_bytes is not None:
                IMAGE_PATH.write_bytes(original_bytes)
            else:
                IMAGE_PATH.unlink(missing=True)

        self.assertIn(
            "The local processed Mumbai SAR render is unavailable on this machine.",
            [i.value for i in tab.info],
        )
        self.assertEqual(len(image_elements(tab)), 0)
        self.assertEqual(tab.expander[0].markdown[0].value, full_mumbai_section(self.document))

    def test_missing_annotation_shows_error(self) -> None:
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == ANNOTATION_PATH:
                raise FileNotFoundError("annotation file unavailable")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            tab = self.sar_tab()
        self.assertIn(
            "Analyst interpretation could not be loaded: annotation file unavailable",
            [e.value for e in tab.error],
        )
        self.assertEqual(len(tab.expander), 0)

    def test_missing_mumbai_heading_has_sanitized_error(self) -> None:
        changed = self.document.replace("## Mumbai coastal", "## Coastal scene", 1)
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            return changed if path == ANNOTATION_PATH else original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            tab = self.sar_tab()
        errors = [item.value for item in tab.error]
        self.assertIn(
            "Analyst interpretation could not be loaded: "
            "The Mumbai coastal annotation section is missing or incomplete.",
            errors,
        )
        self.assertTrue(all("list index out of range" not in message for message in errors))
        self.assertEqual(len(tab.expander), 0)

    def test_missing_category_fails_safely_and_preserves_full_annotation(self) -> None:
        changed = self.document.replace("Vegetation:", "Plant cover:", 1)
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            return changed if path == ANNOTATION_PATH else original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            tab = self.sar_tab()
        self.assertIn(
            "Category summaries unavailable: Expected one 'Vegetation:' heading. "
            "View the full annotation below.",
            [e.value for e in tab.error],
        )
        self.assertEqual(tab.expander[0].markdown[0].value, full_mumbai_section(changed))


if __name__ == "__main__":
    unittest.main()
