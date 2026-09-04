"""Reproducible no-GPU golden-path verification for the Streamlit demo."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demo_gui import golden_assets
from models.qwen_vl.model import QwenVLModel
from orchestrator import trace
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).with_name("app.py")
NO_GPU_ERROR = (
    "Qwen2.5-VL inference requires a CUDA GPU for this baseline; "
    "use the Kaggle T4 runner"
)


class GoldenPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_records = trace.records()
        trace._TRACE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temp_dir.name) / "trace.jsonl"
        self.golden_path = Path(self.temp_dir.name) / "golden.png"
        Image.new("RGB", (1, 1)).save(self.golden_path)
        self.path_patch = patch.object(trace, "TRACE_PATH", self.trace_path)
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        trace._TRACE.clear()
        trace._TRACE.extend(self.original_records)
        self.temp_dir.cleanup()

    def test_no_gpu_golden_path_uses_verified_cached_result(self) -> None:
        with (
            patch.object(
                golden_assets, "local_golden_image", return_value=self.golden_path
            ),
            patch.object(QwenVLModel, "_load", side_effect=RuntimeError(NO_GPU_ERROR)),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertFalse(list(app.exception))

            next(button for button in app.button if button.label == "Ask").click()
            app.run(timeout=30)
            self.assertFalse(list(app.exception))

            fallback_messages = [
                item.value
                for item in app.warning
                if item.value.startswith("Live Qwen inference unavailable")
            ]
            self.assertEqual(
                fallback_messages,
                [
                    "Live Qwen inference unavailable (no GPU) — showing cached result "
                    "for this scene."
                ],
            )
            self.assertIn("Yes", [item.value for item in app.markdown])
            self.assertIn(
                "Confidence calibration pending", [item.value for item in app.caption]
            )

            next(button for button in app.button if button.label == "Verify trace").click()
            app.run(timeout=30)
            self.assertFalse(list(app.exception))
            self.assertIn("✓ Chain verified", [item.value for item in app.success])

    def test_missing_local_pixels_uses_verified_cached_result(self) -> None:
        with patch.object(golden_assets, "local_golden_image", return_value=None):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertFalse(list(app.exception))

            next(button for button in app.button if button.label == "Ask").click()
            app.run(timeout=30)
            self.assertFalse(list(app.exception))

            fallback_messages = [
                item.value
                for item in app.warning
                if item.value.startswith("Live Qwen inference unavailable")
            ]
            self.assertEqual(
                fallback_messages,
                [
                    "Live Qwen inference unavailable (local scene pixels unavailable) — "
                    "showing cached result for this scene."
                ],
            )
            self.assertIn("Yes", [item.value for item in app.markdown])
            self.assertIn(
                "Confidence calibration pending", [item.value for item in app.caption]
            )


if __name__ == "__main__":
    unittest.main()
