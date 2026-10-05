"""Hash-chained, append-only execution tracing."""

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

TRACE_PATH = Path(__file__).resolve().parents[1] / "trace.jsonl"
_TRACE: list[dict[str, Any]] = []
_LOADED_PATH: Path | None = None
_LOCK = RLock()

logger = logging.getLogger(__name__)


class TraceIntegrityError(RuntimeError):
    """Persisted trace history could not be trusted."""


@dataclass(frozen=True)
class _TailRepair:
    """A crash-recovery action for the unterminated final line of the trace.

    ``torn_tail`` holds the bytes of an incomplete final record that must be
    quarantined; ``good_length`` is the byte length of the file once they are
    removed. ``needs_newline`` means the final record is complete but its
    trailing newline never reached disk. ``file_length`` is the file size the
    repair was planned against, so a file that changed underneath is refused.
    """

    file_length: int
    good_length: int
    torn_tail: bytes | None = None
    needs_newline: bool = False


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


def _record_hash(record_without_hash: dict[str, Any], prev_hash: str) -> str:
    canonical = _canonical(record_without_hash)
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


def _decode_record(raw: bytes) -> object:
    """Decode one persisted line; raises ValueError on bad UTF-8 or JSON."""
    return json.loads(raw.decode("utf-8"))


def _parse_trace(
    data: bytes, *, allow_tail_repair: bool
) -> tuple[list[dict[str, Any]], _TailRepair | None]:
    """Parse persisted trace bytes into records.

    Every newline-terminated line must be a JSON object; anything else raises
    ``TraceIntegrityError``. Only the final, unterminated line gets special
    treatment, and only when ``allow_tail_repair`` is set: a crash mid-append
    leaves exactly that shape, so it is planned for quarantine (if it is not a
    complete JSON object) or for newline completion (if it is). Without repair,
    an unterminated final line is reported as an integrity failure.
    """
    candidate: list[dict[str, Any]] = []
    *complete_lines, tail = data.split(b"\n")
    for line_number, raw in enumerate(complete_lines, start=1):
        if not raw.strip():
            raise TraceIntegrityError(f"Persisted trace line {line_number} is empty")
        try:
            record = _decode_record(raw)
        except ValueError as exc:
            raise TraceIntegrityError(
                f"Persisted trace line {line_number} is malformed"
            ) from exc
        if not isinstance(record, dict):
            raise TraceIntegrityError(
                f"Persisted trace line {line_number} is not an object"
            )
        candidate.append(record)

    if not tail:
        return candidate, None

    tail_line_number = len(complete_lines) + 1
    try:
        tail_record: object = _decode_record(tail)
    except ValueError:
        tail_record = None
    if not allow_tail_repair:
        if isinstance(tail_record, dict):
            raise TraceIntegrityError(
                f"Persisted trace line {tail_line_number} has no trailing newline"
            )
        raise TraceIntegrityError(
            f"Persisted trace line {tail_line_number} is incomplete"
        )
    if isinstance(tail_record, dict):
        candidate.append(tail_record)
        return candidate, _TailRepair(
            file_length=len(data), good_length=len(data), needs_newline=True
        )
    return candidate, _TailRepair(
        file_length=len(data),
        good_length=len(data) - len(tail),
        torn_tail=tail,
    )


def _read_trace_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    return path.read_bytes()


def _fsync_directory(directory: Path) -> None:
    """Persist directory entries (new or renamed files) where supported."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _quarantine_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = path.with_name(f"{path.name}.torn-{stamp}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _apply_tail_repair(path: Path, repair: _TailRepair) -> None:
    """Durably apply a planned crash-recovery repair to the trace file."""
    with path.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() != repair.file_length:
            raise TraceIntegrityError("Persisted trace changed during recovery")
        if repair.needs_newline:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            logger.warning("Completed trailing newline on final persisted trace record")
            return

        assert repair.torn_tail is not None
        # Preserve the torn bytes durably before removing them from the log.
        quarantine = _quarantine_path(path)
        with quarantine.open("xb") as torn_handle:
            torn_handle.write(repair.torn_tail)
            torn_handle.flush()
            os.fsync(torn_handle.fileno())
        _fsync_directory(path.parent)

        handle.truncate(repair.good_length)
        handle.flush()
        os.fsync(handle.fileno())
        logger.warning(
            "Quarantined incomplete final trace line (%d bytes) to %s",
            len(repair.torn_tail),
            quarantine.name,
        )


def _ensure_loaded() -> None:
    global _LOADED_PATH
    with _LOCK:
        path = Path(TRACE_PATH)
        if _LOADED_PATH == path:
            return
        try:
            data = _read_trace_bytes(path)
        except OSError as exc:
            raise TraceIntegrityError("Persisted trace history could not be read") from exc
        candidate, repair = _parse_trace(data, allow_tail_repair=True)

        verified, message = _verify(candidate)
        if not verified:
            raise TraceIntegrityError(message)
        if repair is not None:
            try:
                _apply_tail_repair(path, repair)
            except TraceIntegrityError:
                raise
            except OSError as exc:
                raise TraceIntegrityError(
                    "Persisted trace history could not be recovered"
                ) from exc
        _TRACE[:] = candidate
        _LOADED_PATH = path


def append_record(record: dict[str, Any]) -> dict[str, Any]:
    """Append a record to memory and disk, returning it with its hash.

    The record is flushed and fsynced before it is acknowledged.
    """
    global _LOADED_PATH
    with _LOCK:
        _ensure_loaded()
        prev_hash = _TRACE[-1]["record_hash"] if _TRACE else ""
        chained_record = {**record, "prev_hash": prev_hash}
        record_hash = _record_hash(chained_record, prev_hash)
        stored = {**chained_record, "record_hash": record_hash}
        serialized = json.dumps(stored, sort_keys=True) + "\n"
        path = Path(TRACE_PATH)
        created = not path.exists()
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            if created:
                _fsync_directory(path.parent)
        except BaseException:
            # Disk may now hold a partial or unsynced line that memory lacks;
            # force the next access to reload (and recover) from disk.
            _LOADED_PATH = None
            raise
        _TRACE.append(stored)
        return stored.copy()


def records() -> list[dict[str, Any]]:
    with _LOCK:
        _ensure_loaded()
        return [record.copy() for record in _TRACE]


def verify_chain(chain: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """Verify record hashes and links.

    With an explicit ``chain``, only that in-memory chain is verified. With no
    argument, the persisted trace file is re-read and verified (without any
    crash recovery), and must match the history loaded into memory.
    """
    if chain is not None:
        return _verify([record.copy() for record in chain])

    with _LOCK:
        _ensure_loaded()
        try:
            data = _read_trace_bytes(Path(TRACE_PATH))
        except OSError:
            return False, "Persisted trace history could not be read"
        try:
            persisted, _ = _parse_trace(data, allow_tail_repair=False)
        except TraceIntegrityError as exc:
            return False, str(exc)
        verified, message = _verify(persisted)
        if not verified:
            return False, message
        # Compare canonical serializations: in-memory records may hold values
        # (e.g. tuples) that legitimately round-trip through JSON as lists.
        if [_canonical(record) for record in persisted] != [
            _canonical(record) for record in _TRACE
        ]:
            return False, "Persisted trace diverges from loaded history"
        return True, message
