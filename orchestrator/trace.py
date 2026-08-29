"""Hash-chained, append-only execution tracing."""

import hashlib
import json
from pathlib import Path
from typing import Any

TRACE_PATH = Path(__file__).resolve().parents[1] / "trace.jsonl"
_TRACE: list[dict[str, Any]] = []


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    """Append a record to memory and disk, returning it with its hash."""
    prev_hash = _TRACE[-1]["record_hash"] if _TRACE else ""
    chained_record = {**record, "prev_hash": prev_hash}
    canonical = json.dumps(chained_record, sort_keys=True)
    record_hash = hashlib.sha256((canonical + prev_hash).encode("utf-8")).hexdigest()
    stored = {**chained_record, "record_hash": record_hash}
    _TRACE.append(stored)
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stored, sort_keys=True) + "\n")
    return stored.copy()


def records() -> list[dict[str, Any]]:
    return [record.copy() for record in _TRACE]
