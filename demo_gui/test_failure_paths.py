"""Judge-facing failure-path regressions for the Streamlit demo."""

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from demo_gui import golden_assets
from models.qwen_vl.model import QwenVLModel

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "demo_gui/app.py"
RESULTS_PATH = ROOT / "results/qwen2.5vl-3b__ladder__rescored__20260904.json"
SAR_IMAGE_PATH = ROOT / "data/sar_gate/rendered/mumbai_coastal.png"
NO_GPU_ERROR = (
    "Qwen2.5-VL inference requires a CUDA GPU for this baseline; "
    "use the Kaggle T4 runner"
)


def rendered_text(app: AppTest) -> str:
    collections = (
        app.caption,
        app.code,
        app.error,
        app.header,
        app.info,
        app.markdown,
        app.subheader,
        app.success,
        app.text,
        app.title,
        app.warning,
    )
    return "\n".join(str(item.value) for collection in collections for item in collection)


class FailurePathTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.golden_path = Path(self.temp_dir.name) / "golden.png"
        Image.new("RGB", (2, 2)).save(self.golden_path)

    def tearDown(self) -> None:
        st.cache_data.clear()
        self.temp_dir.cleanup()

    def assert_safe(self, app: AppTest) -> str:
        self.assertFalse(list(app.exception))
        text = rendered_text(app)
        self.assertNotIn("Traceback (most recent call last)", text)
        return text

    @staticmethod
    def ask(app: AppTest) -> AppTest:
        next(button for button in app.button if button.label == "Ask").click()
        return app.run(timeout=30)

    def test_no_gpu_uses_cached_fallback_without_traceback(self) -> None:
        with (
            patch.object(golden_assets, "local_golden_image", return_value=self.golden_path),
            patch.object(QwenVLModel, "_load", side_effect=RuntimeError(NO_GPU_ERROR)),
        ):
            app = self.ask(AppTest.from_file(str(APP_PATH)).run(timeout=30))
        text = self.assert_safe(app)
        self.assertIn("Live Qwen inference unavailable (no GPU)", text)
        self.assertIn("VERIFIED CACHED RESULT", text)
        self.assertIn("Yes", text)

    def test_missing_model_cache_uses_cached_fallback_without_traceback(self) -> None:
        message = "Local model cache/weights are unavailable."
        with (
            patch.object(golden_assets, "local_golden_image", return_value=self.golden_path),
            patch.object(QwenVLModel, "_load", side_effect=OSError(message)),
        ):
            app = self.ask(AppTest.from_file(str(APP_PATH)).run(timeout=30))
        text = self.assert_safe(app)
        self.assertIn(message, text)
        self.assertIn("VERIFIED CACHED RESULT", text)
        self.assertIn("Yes", text)

    def test_no_cached_match_shows_error_and_no_fabricated_answer(self) -> None:
        uploaded = BytesIO(self.golden_path.read_bytes())
        uploaded.name = "not-in-artifact.png"
        with (
            patch("streamlit.file_uploader", return_value=uploaded),
            patch.object(QwenVLModel, "_load", side_effect=RuntimeError(NO_GPU_ERROR)),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.radio[0].set_value("Upload a scene").run(timeout=30)
            app.text_input[0].input("A question with no cached result").run(timeout=30)
            app = self.ask(app)
        text = self.assert_safe(app)
        self.assertIn("No verified cached result matches this scene and question", text)
        self.assertNotIn("Answer", [item.value for item in app.subheader])
        self.assertNotIn("VERIFIED CACHED RESULT", text)
        self.assertNotIn("LIVE INFERENCE", text)

    def test_missing_golden_image_keeps_cached_result_usable(self) -> None:
        with patch.object(golden_assets, "local_golden_image", return_value=None):
            app = self.ask(AppTest.from_file(str(APP_PATH)).run(timeout=30))
        text = self.assert_safe(app)
        self.assertIn("Local golden pixels are absent", text)
        self.assertIn("VERIFIED CACHED RESULT", text)
        self.assertIn("Yes", text)

    def test_missing_results_artifact_stops_safely(self) -> None:
        original = Path.read_text

        def read_text(path: Path, *args, **kwargs):
            if path == RESULTS_PATH:
                raise FileNotFoundError("results artifact unavailable")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read_text):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        text = self.assert_safe(app)
        self.assertIn(
            "Required ladder artifact could not be loaded: results artifact unavailable",
            text,
        )
        self.assertEqual(list(app.tabs), [])

    def test_missing_sar_render_is_informational_without_traceback(self) -> None:
        original = Path.is_file

        def is_file(path: Path) -> bool:
            return False if path == SAR_IMAGE_PATH else original(path)

        with patch.object(Path, "is_file", is_file):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        text = self.assert_safe(app)
        self.assertIn(
            "The local processed Mumbai SAR render is unavailable on this machine.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
