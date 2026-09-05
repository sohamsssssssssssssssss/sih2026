"""Reproducible no-GPU golden-path verification for the Streamlit demo."""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from demo_gui import golden_assets
from models.qwen_vl.model import QwenVLModel
from orchestrator import trace
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).with_name("app.py")
RESULTS_ARTIFACT = "results/qwen2.5vl-3b__ladder__rescored__20260904.json"
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
            self.assertIn("VERIFIED CACHED RESULT", [item.value for item in app.info])
            self.assertNotIn("LIVE INFERENCE", [item.value for item in app.success])
            self.assertIn(RESULTS_ARTIFACT, [item.value for item in app.code])
            evidence = json.loads(app.json[0].value)
            self.assertEqual(evidence["execution_mode"], "cached_result")
            self.assertEqual(evidence["results_artifact"], RESULTS_ARTIFACT)
            self.assertIn(
                "Confidence calibration pending", [item.value for item in app.caption]
            )

            next(button for button in app.button if button.label == "Verify trace").click()
            app.run(timeout=30)
            self.assertFalse(list(app.exception))
            self.assertIn("✓ Chain verified", [item.value for item in app.success])

    def test_live_success_renders_live_badge(self) -> None:
        with (
            patch.object(golden_assets, "local_golden_image", return_value=self.golden_path),
            patch.object(QwenVLModel, "infer", return_value={"answer": "Yes", "evidence": []}) as infer,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            next(button for button in app.button if button.label == "Ask").click()
            app.run(timeout=30)
            self.assertFalse(list(app.exception))
            infer.assert_called_once()
            self.assertIn("LIVE INFERENCE", [item.value for item in app.success])
            self.assertIn(
                "Live Qwen2.5-VL-3B inference completed.",
                [item.value for item in app.success],
            )
            self.assertNotIn("VERIFIED CACHED RESULT", [item.value for item in app.info])
            self.assertNotIn(RESULTS_ARTIFACT, [item.value for item in app.code])
            self.assertIn("Yes", [item.value for item in app.markdown])
            evidence = json.loads(app.json[0].value)
            self.assertEqual(evidence["execution_mode"], "live")
            self.assertNotIn("results_artifact", evidence)

    def test_missing_or_unknown_execution_mode_renders_error(self) -> None:
        for params in ({}, {"execution_mode": "unexpected"}):
            with self.subTest(params=params):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                app.session_state["last_response"] = {
                    "answer": "Yes", "trace": {"params": params}
                }
                app.run(timeout=30)
                self.assertFalse(list(app.exception))
                self.assertIn(
                    "Missing or unrecognized execution_mode in response trace: "
                    f"{params.get('execution_mode')!r}",
                    [item.value for item in app.error],
                )
                self.assertNotIn("LIVE INFERENCE", [item.value for item in app.success])
                self.assertNotIn("VERIFIED CACHED RESULT", [item.value for item in app.info])

    def test_no_response_renders_error_notice_without_badge(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        app.session_state["last_response"] = None
        app.session_state["last_notice"] = "No local image or verified cached result matches this query."
        app.session_state["last_notice_error"] = True
        app.run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertIn(
            "No local image or verified cached result matches this query.",
            [item.value for item in app.error],
        )
        self.assertNotIn("Answer", [item.value for item in app.subheader])
        self.assertNotIn("LIVE INFERENCE", [item.value for item in app.success])
        self.assertNotIn("VERIFIED CACHED RESULT", [item.value for item in app.info])

    def test_scene_metadata_and_uploaded_preview_use_input_column(self) -> None:
        with patch.object(golden_assets, "local_golden_image", return_value=self.golden_path):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertFalse(list(app.exception))
            left, right = app.get("column")
            self.assertIn("GSD: 0.3 m", [item.value for item in left.caption])
            self.assertIn("Sensor: LoveDA", [item.value for item in left.caption])
            self.assertEqual(len(left.get("imgs")), 1)
            self.assertEqual(right.text_input[0].label, "Question")

            app.radio[0].set_value("Upload a scene").run(timeout=30)
            self.assertFalse(list(app.exception))
            left, right = app.get("column")
            self.assertIn("GSD: unknown", [item.value for item in left.caption])
            self.assertEqual(len(left.get("imgs")), 0)

            uploaded = BytesIO(self.golden_path.read_bytes())
            uploaded.name = "uploaded_scene.png"
            with patch("streamlit.file_uploader", return_value=uploaded):
                app.run(timeout=30)
            self.assertFalse(list(app.exception))
            left, right = app.get("column")
            self.assertIn("GSD: unknown", [item.value for item in left.caption])
            self.assertIn("uploaded_scene.png", [item.value for item in left.code])
            self.assertEqual(len(left.get("imgs")), 1)
            self.assertEqual(right.text_input[0].label, "Question")

    def test_evidence_cards_preserve_full_hashes_and_raw_fields(self) -> None:
        cases = (
            ("0123456789abcdef" * 4, "", "01234567...89abcdef", "First record"),
            ("short", "fedcba9876543210" * 4, "short", "fedcba98...76543210"),
            (None, None, "Not recorded", "Not recorded"),
        )
        for record_hash, prev_hash, record_display, prev_display in cases:
            with self.subTest(record_hash=record_hash, prev_hash=prev_hash):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
                params = {
                    "scene_id": golden_assets.GOLDEN_SCENE_ID,
                    "sensor": "LoveDA",
                    "execution_mode": "cached_result",
                    "results_artifact": RESULTS_ARTIFACT,
                }
                expected = {
                    **params,
                    "question": golden_assets.GOLDEN_QUESTION,
                    "model_name": "qwen2.5vl-3b",
                    "model_version": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "timestamp": "2026-09-05T00:00:00+00:00",
                }
                if record_hash is not None:
                    expected["record_hash"] = record_hash
                if prev_hash is not None:
                    expected["prev_hash"] = prev_hash
                app.session_state["last_response"] = {
                    "answer": "Yes",
                    "trace": {
                        "params": params,
                        "input_summary": {"question": expected["question"]},
                        "model_name": expected["model_name"],
                        "model_version": expected["model_version"],
                        "timestamp_iso": expected["timestamp"],
                        "record_hash": record_hash,
                        "prev_hash": prev_hash,
                    },
                }
                app.run(timeout=30)
                self.assertFalse(list(app.exception))
                section = next(e for e in app.expander if e.label == "Evidence and execution trace")
                metrics = {m.label: m for m in section.metric}
                self.assertEqual(metrics["Model"].value, expected["model_name"])
                self.assertEqual(metrics["Sensor"].value, expected["sensor"])
                self.assertEqual(metrics["Execution mode"].value, "cached_result")
                self.assertEqual(metrics["Record hash"].value, record_display)
                self.assertEqual(metrics["Previous hash"].value, prev_display)
                for label, value in (("Record hash", record_hash), ("Previous hash", prev_hash)):
                    if value:
                        self.assertEqual(metrics[label].proto.help, f"Full value: {value}")
                    elif value == "":
                        self.assertIn('Full value: ""', metrics[label].proto.help)
                for key in ("model_version", "scene_id", "timestamp", "results_artifact"):
                    self.assertIn(expected[key], [c.value for c in section.code])
                self.assertIn(expected["question"], [t.value for t in section.text])
                raw = next(e for e in section.expander if e.label == "Raw evidence JSON")
                self.assertFalse(raw.proto.expanded)
                self.assertEqual(json.loads(raw.json[0].value), expected)
                explainer = (
                    "Every model/tool invocation is chained to the previous execution record. "
                    "Altering an earlier record invalidates verification."
                )
                self.assertIn(explainer, [c.value for c in app.caption])
                self.assertNotIn(explainer, [c.value for c in section.caption])

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
