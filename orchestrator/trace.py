"""Hash-chained, append-only execution tracing."""

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

TRACE_PATH = Path(__file__).resolve().parents[1] / "trace.jsonl"
_TRACE: list[dict[str, Any]] = []


def _record_hash(record_without_hash: dict[str, Any], prev_hash: str) -> str:
    canonical = json.dumps(record_without_hash, sort_keys=True)
    return hashlib.sha256((canonical + prev_hash).encode("utf-8")).hexdigest()


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    """Append a record to memory and disk, returning it with its hash."""
    prev_hash = _TRACE[-1]["record_hash"] if _TRACE else ""
    chained_record = {**record, "prev_hash": prev_hash}
    record_hash = _record_hash(chained_record, prev_hash)
    stored = {**chained_record, "record_hash": record_hash}
    _TRACE.append(stored)
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stored, sort_keys=True) + "\n")
    return stored.copy()


def records() -> list[dict[str, Any]]:
    return [record.copy() for record in _TRACE]


def verify_chain(chain: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """Verify record hashes and links for an in-memory trace chain."""
    candidate = records() if chain is None else [record.copy() for record in chain]
    expected_prev_hash = ""
    for index, stored in enumerate(candidate, start=1):
        record_hash = stored.get("record_hash")
        if not isinstance(record_hash, str) or not record_hash:
            return False, f"Record {index} has no record_hash"
        record_without_hash = {
            key: value for key, value in stored.items() if key != "record_hash"
        }
        prev_hash = record_without_hash.get("prev_hash")
        if prev_hash != expected_prev_hash:
            return False, f"Record {index} has an invalid prev_hash"
        expected_hash = _record_hash(record_without_hash, expected_prev_hash)
        if not hmac.compare_digest(record_hash, expected_hash):
            return False, f"Record {index} hash mismatch"
        expected_prev_hash = record_hash
    return True, f"Chain verified ({len(candidate)} records)"
