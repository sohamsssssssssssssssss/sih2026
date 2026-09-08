"""Hash-chained, append-only execution tracing."""

import hashlib
import hmac
import json
from pathlib import Path
from threading import RLock
from typing import Any

TRACE_PATH = Path(__file__).resolve().parents[1] / "trace.jsonl"
_TRACE: list[dict[str, Any]] = []
_LOADED_PATH: Path | None = None
_LOCK = RLock()


class TraceIntegrityError(RuntimeError):
    """Persisted trace history could not be trusted."""


def _record_hash(record_without_hash: dict[str, Any], prev_hash: str) -> str:
    canonical = json.dumps(record_without_hash, sort_keys=True)
    return hashlib.sha256((canonical + prev_hash).encode("utf-8")).hexdigest()


def _is_sha256_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _verify(candidate: list[dict[str, Any]]) -> tuple[bool, str]:
    expected_prev_hash = ""
    for index, stored in enumerate(candidate, start=1):
        record_hash = stored.get("record_hash")
        if not _is_sha256_hash(record_hash):
            return False, f"Record {index} has no record_hash"
        record_without_hash = {
            key: value for key, value in stored.items() if key != "record_hash"
        }
        prev_hash = record_without_hash.get("prev_hash")
        valid_prev_hash = prev_hash == "" if index == 1 else _is_sha256_hash(prev_hash)
        if not valid_prev_hash or not hmac.compare_digest(prev_hash, expected_prev_hash):
            return False, f"Record {index} has an invalid prev_hash"
        expected_hash = _record_hash(record_without_hash, expected_prev_hash)
        if not hmac.compare_digest(record_hash, expected_hash):
            return False, f"Record {index} hash mismatch"
        expected_prev_hash = record_hash
    return True, f"Chain verified ({len(candidate)} records)"


def _ensure_loaded() -> None:
    global _LOADED_PATH
    with _LOCK:
        path = Path(TRACE_PATH)
        if _LOADED_PATH == path:
            return
        try:
            if not path.exists() or path.stat().st_size == 0:
                candidate: list[dict[str, Any]] = []
            else:
                candidate = []
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            raise TraceIntegrityError(
                                f"Persisted trace line {line_number} is empty"
                            )
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise TraceIntegrityError(
                                f"Persisted trace line {line_number} is malformed"
                            ) from exc
                        if not isinstance(record, dict):
                            raise TraceIntegrityError(
                                f"Persisted trace line {line_number} is not an object"
                            )
                        candidate.append(record)
        except TraceIntegrityError:
            raise
        except (OSError, UnicodeError) as exc:
            raise TraceIntegrityError("Persisted trace history could not be read") from exc

        verified, message = _verify(candidate)
        if not verified:
            raise TraceIntegrityError(message)
        _TRACE[:] = candidate
        _LOADED_PATH = path


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    """Append a record to memory and disk, returning it with its hash."""
    with _LOCK:
        _ensure_loaded()
        prev_hash = _TRACE[-1]["record_hash"] if _TRACE else ""
        chained_record = {**record, "prev_hash": prev_hash}
        record_hash = _record_hash(chained_record, prev_hash)
        stored = {**chained_record, "record_hash": record_hash}
        serialized = json.dumps(stored, sort_keys=True) + "\n"
        with TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
        _TRACE.append(stored)
        return stored.copy()


def records() -> list[dict[str, Any]]:
    with _LOCK:
        _ensure_loaded()
        return [record.copy() for record in _TRACE]


def verify_chain(chain: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """Verify record hashes and links for an in-memory trace chain."""
    candidate = records() if chain is None else [record.copy() for record in chain]
    return _verify(candidate)
