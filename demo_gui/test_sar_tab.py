"""Verify manual SAR excerpts and display fallbacks without changing annotations."""

import contextlib
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from demo_gui.test_support import image_elements

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = ROOT / "data/sar_gate/annotation_template.md"
IMAGE_PATH = ROOT / "data/sar_gate/rendered/mumbai_coastal.png"


@contextlib.contextmanager
def synthetic_render_present():
    """Temporarily ensure a real, readable image file exists at IMAGE_PATH.

    data/sar_gate/rendered/mumbai_coastal.png is not guaranteed to be
    committed (known behaviour in some clones — see the project handover),
    so this cannot assume it's already there. AppTest.from_file executes
    app.py as a fresh script each run rather than reusing the already-
    imported demo_gui.app module object, so patching that module's
    SAR_IMAGE_PATH attribute has no effect; and Streamlit genuinely opens
    the file to read its bytes, so Path.is_file() alone isn't enough
    either. This writes a real synthetic PNG to the real path (only if one
    isn't already there) and removes exactly what it created afterward,
    regardless of test outcome.
    """
    directory_existed = IMAGE_PATH.parent.is_dir()
    file_existed = IMAGE_PATH.is_file()
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not file_existed:
        Image.new("RGB", (2, 2)).save(IMAGE_PATH)
    try:
        yield
    finally:
        if not file_existed and IMAGE_PATH.is_file():
            IMAGE_PATH.unlink()
        if not directory_existed and IMAGE_PATH.parent.is_dir():
            IMAGE_PATH.parent.rmdir()


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
        return next(tab for tab in app.tabs if tab.label == "SAR Validation")

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
        # Positive control, using a synthetic file at the real path rather
        # than depending on data/sar_gate/rendered/mumbai_coastal.png being
        # committed (it may not be — see synthetic_render_present() above).
        with synthetic_render_present():
            self.assertEqual(len(image_elements(self.sar_tab())), 1)

        original = Path.is_file
        with patch.object(Path, "is_file", lambda path: False if path == IMAGE_PATH else original(path)):
            tab = self.sar_tab()

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
