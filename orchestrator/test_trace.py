"""Tests for hash-chain verification."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator import trace


class TraceVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_records = trace.records()
        trace._TRACE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temp_dir.name) / "trace.jsonl"
        self.path_patch = patch.object(trace, "TRACE_PATH", self.trace_path)
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        trace._TRACE.clear()
        trace._TRACE.extend(self.original_records)
        self.temp_dir.cleanup()

    def test_untampered_chain_verifies(self) -> None:
        trace.append_record({"model_name": "qwen2.5vl-3b", "question": "first"})
        trace.append_record({"model_name": "qwen2.5vl-3b", "question": "second"})

        verified, message = trace.verify_chain()

        self.assertTrue(verified)
        self.assertEqual(message, "Chain verified (2 records)")

    def test_tampered_record_is_detected(self) -> None:
        trace.append_record({"model_name": "qwen2.5vl-3b", "question": "first"})
        trace.append_record({"model_name": "qwen2.5vl-3b", "question": "second"})
        tampered = trace.records()
        tampered[0]["question"] = "altered after recording"

        verified, message = trace.verify_chain(tampered)

        self.assertFalse(verified)
        self.assertEqual(message, "Record 1 hash mismatch")


if __name__ == "__main__":
    unittest.main()
