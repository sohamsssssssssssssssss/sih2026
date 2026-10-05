"""Tests for hash-chain verification."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from orchestrator import trace


class TraceVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_records = [record.copy() for record in trace._TRACE]
        self.original_loaded_path = trace._LOADED_PATH
        trace._TRACE.clear()
        trace._LOADED_PATH = None
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.temp_dir.name) / "trace.jsonl"
        self.path_patch = patch.object(trace, "TRACE_PATH", self.trace_path)
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        trace._TRACE.clear()
        trace._TRACE.extend(self.original_records)
        trace._LOADED_PATH = self.original_loaded_path
        self.temp_dir.cleanup()

    def simulate_restart(self) -> None:
        trace._TRACE.clear()
        trace._LOADED_PATH = None

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

    def test_missing_trace_file_is_valid_empty_history(self) -> None:
        self.assertEqual(trace.records(), [])
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (0 records)"))

    def test_zero_byte_trace_file_is_valid_empty_history(self) -> None:
        self.trace_path.touch()
        self.assertEqual(trace.records(), [])
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (0 records)"))

    def test_restart_restores_history_and_continues_chain(self) -> None:
        first = trace.append_record({"question": "A"})
        second = trace.append_record({"question": "B"})
        self.simulate_restart()

        self.assertEqual(trace.records(), [first, second])
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (2 records)"))
        third = trace.append_record({"question": "C"})
        self.assertEqual(third["prev_hash"], second["record_hash"])

    def test_path_change_loads_the_new_trace_file(self) -> None:
        trace.append_record({"question": "first path"})
        other_path = Path(self.temp_dir.name) / "other.jsonl"
        with patch.object(trace, "TRACE_PATH", other_path):
            self.assertEqual(trace.records(), [])

    def test_malformed_json_fails_closed(self) -> None:
        self.trace_path.write_text("{broken\n", encoding="utf-8")
        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()

    def test_non_object_json_fails_closed(self) -> None:
        self.trace_path.write_text('["not", "an", "object"]\n', encoding="utf-8")
        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()

    def test_blank_record_fails_closed(self) -> None:
        self.trace_path.write_text("\n", encoding="utf-8")
        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()

    def test_invalid_hash_and_link_types_fail_closed(self) -> None:
        for record in (
            {"prev_hash": "", "record_hash": None},
            {"prev_hash": "", "record_hash": "é"},
            {"prev_hash": "", "record_hash": "abc"},
            {"prev_hash": "", "record_hash": "z" * 64},
            {"prev_hash": 123, "record_hash": "0" * 64},
        ):
            with self.subTest(record=record):
                self.trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                self.simulate_restart()
                with self.assertRaises(trace.TraceIntegrityError):
                    trace.records()

    def test_non_ascii_previous_hash_fails_closed(self) -> None:
        trace.append_record({"question": "A"})
        second = trace.append_record({"question": "B"})
        second["prev_hash"] = "é"
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        lines[1] = json.dumps(second, ensure_ascii=False)
        self.trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()

    def test_broken_link_fails_closed(self) -> None:
        trace.append_record({"question": "A"})
        second = trace.append_record({"question": "B"})
        second["prev_hash"] = "broken"
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        lines[1] = json.dumps(second)
        self.trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()

    def test_tampered_persisted_record_fails_closed(self) -> None:
        trace.append_record({"question": "A"})
        record = json.loads(self.trace_path.read_text(encoding="utf-8"))
        record["question"] = "tampered"
        self.trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.verify_chain()

    def test_explicit_chain_does_not_load_corrupt_disk_history(self) -> None:
        self.trace_path.write_text("corrupt\n", encoding="utf-8")
        self.assertEqual(
            trace.verify_chain([]), (True, "Chain verified (0 records)")
        )

    def test_same_process_concurrent_appends_remain_one_chain(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda index: trace.append_record({"index": index}), range(20)))

        self.assertEqual(len(trace.records()), 20)
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (20 records)"))

    def test_failed_disk_append_does_not_update_memory(self) -> None:
        trace.records()
        trace.TRACE_PATH = self.trace_path.parent / "missing" / "trace.jsonl"
        trace._LOADED_PATH = trace.TRACE_PATH

        with self.assertRaises(OSError):
            trace.append_record({"question": "not persisted"})
        self.assertEqual(trace._TRACE, [])

    def quarantine_files(self) -> list[Path]:
        return sorted(self.trace_path.parent.glob("trace.jsonl.torn-*"))

    def test_append_flushes_and_fsyncs_before_acknowledging(self) -> None:
        with patch.object(trace.os, "fsync", wraps=trace.os.fsync) as fsync:
            trace.append_record({"question": "durable"})

        self.assertGreaterEqual(fsync.call_count, 1)
        self.assertEqual(len(self.trace_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_failed_fsync_is_not_acknowledged_and_forces_reload(self) -> None:
        trace.append_record({"question": "A"})
        with patch.object(trace.os, "fsync", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                trace.append_record({"question": "B"})

        self.assertIsNone(trace._LOADED_PATH)
        self.assertEqual([r["question"] for r in trace.records()], ["A", "B"])
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (2 records)"))

    def test_torn_final_line_is_quarantined_and_chain_continues(self) -> None:
        first = trace.append_record({"question": "A"})
        second = trace.append_record({"question": "B"})
        good_bytes = self.trace_path.read_bytes()
        torn = b'{"prev_hash": "' + second["record_hash"].encode() + b'", "quest'
        with self.trace_path.open("ab") as handle:
            handle.write(torn)
        self.simulate_restart()

        self.assertEqual(trace.records(), [first, second])
        self.assertEqual(self.trace_path.read_bytes(), good_bytes)
        quarantined = self.quarantine_files()
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), torn)

        third = trace.append_record({"question": "C"})
        self.assertEqual(third["prev_hash"], second["record_hash"])
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (3 records)"))
        self.simulate_restart()
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (3 records)"))

    def test_torn_only_line_recovers_to_empty_history(self) -> None:
        self.trace_path.write_bytes(b'{"prev_hash": "", "qu')

        self.assertEqual(trace.records(), [])
        self.assertEqual(self.trace_path.read_bytes(), b"")
        self.assertEqual(len(self.quarantine_files()), 1)
        first = trace.append_record({"question": "A"})
        self.assertEqual(first["prev_hash"], "")
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (1 records)"))

    def test_complete_final_record_missing_newline_is_preserved(self) -> None:
        first = trace.append_record({"question": "A"})
        second = trace.append_record({"question": "B"})
        self.trace_path.write_bytes(self.trace_path.read_bytes().rstrip(b"\n"))
        self.simulate_restart()

        self.assertEqual(trace.records(), [first, second])
        self.assertTrue(self.trace_path.read_bytes().endswith(b"\n"))
        self.assertEqual(self.quarantine_files(), [])
        third = trace.append_record({"question": "C"})
        self.assertEqual(third["prev_hash"], second["record_hash"])
        self.simulate_restart()
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (3 records)"))

    def test_malformed_middle_line_still_fails_closed(self) -> None:
        trace.append_record({"question": "A"})
        trace.append_record({"question": "B"})
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        lines.insert(1, "{broken")
        original = "\n".join(lines) + "\n"
        self.trace_path.write_text(original, encoding="utf-8")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()
        self.assertEqual(self.trace_path.read_text(encoding="utf-8"), original)
        self.assertEqual(self.quarantine_files(), [])

    def test_malformed_terminated_final_line_still_fails_closed(self) -> None:
        trace.append_record({"question": "A"})
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write("{broken\n")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()
        self.assertEqual(self.quarantine_files(), [])

    def test_torn_tail_after_tampered_chain_is_not_repaired(self) -> None:
        trace.append_record({"question": "A"})
        record = json.loads(self.trace_path.read_text(encoding="utf-8"))
        record["question"] = "tampered"
        original = json.dumps(record) + "\n" + '{"prev_hash": "'
        self.trace_path.write_text(original, encoding="utf-8")
        self.simulate_restart()

        with self.assertRaises(trace.TraceIntegrityError):
            trace.records()
        self.assertEqual(self.trace_path.read_text(encoding="utf-8"), original)
        self.assertEqual(self.quarantine_files(), [])

    def test_on_disk_tampering_after_load_is_detected(self) -> None:
        trace.append_record({"question": "A"})
        trace.append_record({"question": "B"})
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (2 records)"))
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[0])
        tampered["question"] = "altered on disk"
        lines[0] = json.dumps(tampered, sort_keys=True)
        self.trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.assertEqual(trace.verify_chain(), (False, "Record 1 hash mismatch"))

    def test_on_disk_divergence_after_load_is_detected(self) -> None:
        trace.append_record({"question": "A"})
        trace.append_record({"question": "B"})
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        # Truncating the tail leaves a self-consistent chain that no longer
        # matches what this process recorded.
        self.trace_path.write_text(lines[0] + "\n", encoding="utf-8")

        self.assertEqual(
            trace.verify_chain(),
            (False, "Persisted trace diverges from loaded history"),
        )

    def test_verify_reports_torn_tail_without_repairing(self) -> None:
        trace.append_record({"question": "A"})
        with self.trace_path.open("ab") as handle:
            handle.write(b'{"prev_')
        original = self.trace_path.read_bytes()

        self.assertEqual(
            trace.verify_chain(), (False, "Persisted trace line 2 is incomplete")
        )
        self.assertEqual(self.trace_path.read_bytes(), original)
        self.assertEqual(self.quarantine_files(), [])

    def test_verify_handles_records_with_tuple_values(self) -> None:
        trace.append_record({"question": "A", "order": ("t1", "t2")})
        self.assertEqual(trace.verify_chain(), (True, "Chain verified (1 records)"))


if __name__ == "__main__":
    unittest.main()
