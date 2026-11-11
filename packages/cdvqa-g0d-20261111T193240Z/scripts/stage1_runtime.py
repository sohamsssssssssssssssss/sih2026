"""Shared checkpoint provenance and bounded checkpoint storage. No model imports."""
import hashlib
import json
import os
import time
from pathlib import Path

WORKING_CAP_BYTES = 15 * 1024**3


def verify_checkpoint(checkpoint, revision, weight_sha256):
    """Require independent commit proof AND every expected weight digest."""
    checkpoint = Path(checkpoint)
    began = time.perf_counter()
    result = {"weights_match": bool(weight_sha256), "revision_source": None,
              "revision_matches": False, "passed": False, "weight_sha256": {},
              "measured_revision": None, "verification_bytes": 0}
    try:
        config = json.loads((checkpoint / "config.json").read_text())
        if "_commit_hash" in config:
            result["revision_source"] = "config.json"
            result["measured_revision"] = config["_commit_hash"]
        else:
            metadata = checkpoint / ".cache/huggingface/download/config.json.metadata"
            if metadata.is_file():
                result["revision_source"] = "hf_download_metadata"
                lines = metadata.read_text().splitlines()
                result["measured_revision"] = lines[0] if lines else None
        result["revision_matches"] = (isinstance(revision, str) and len(revision) == 40
                                      and result["measured_revision"] == revision)
    except (OSError, ValueError) as error:
        result["revision_error"] = repr(error)
    for filename, expected in weight_sha256.items():
        try:
            digest = hashlib.sha256()
            with (checkpoint / filename).open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    result["verification_bytes"] += len(chunk)
            found = digest.hexdigest()
            entry = {"expected": expected, "measured": found, "matches": found == expected}
        except OSError as error:
            entry = {"expected": expected, "measured": None, "matches": False, "error": repr(error)}
        result["weight_sha256"][filename] = entry
        result["weights_match"] &= entry["matches"]
    result["passed"] = result["weights_match"] and result["revision_matches"]
    result["verification_seconds"] = time.perf_counter() - began
    return result


def directory_bytes(root):
    """Count regular files once per inode, including datasets/retry leftovers."""
    seen = set()
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def storage_root(path):
    path = Path(path).resolve()
    working = Path("/kaggle/working")
    return working if path.is_relative_to(working) else path.parent


def bounded_write_json(path, payload, *, cap_bytes=WORKING_CAP_BYTES):
    """Apply the same working-tree cap to evidence/log growth between saves."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2) + "\n"
    old_size = path.stat().st_size if path.exists() else 0
    if directory_bytes(storage_root(path)) - old_size + len(content.encode()) >= cap_bytes:
        raise OSError("Evidence write would reach working storage cap")
    path.write_text(content)


def bounded_checkpoint_save(state, path, *, cap_bytes=WORKING_CAP_BYTES):
    """Atomic save; keep last two managed states across stages and retries.

    Count the whole working tree, including weights and failed temporary files.
    Refuse a write before crossing the cap; never delete results or unregistered
    files. A failed temporary file stays for diagnosis and counts on retry.
    """
    import torch
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = storage_root(path)
    registry = root / ".stage1-checkpoints.json"
    entries = json.loads(registry.read_text()) if registry.exists() else []
    for name in entries:
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or candidate.suffix != ".pt":
            raise ValueError("Unsafe managed checkpoint registry")
    temp = path.with_suffix(".pt.tmp")
    if temp.exists():
        raise FileExistsError(f"Preserved failed checkpoint exists: {temp}; use a new output")
    reserve = 64 * 1024  # registry and atomic-save bookkeeping
    baseline = directory_bytes(root) + reserve
    if baseline >= cap_bytes:
        raise OSError("Checkpoint storage cap already reached")

    class LimitedWriter:
        def __init__(self, handle):
            self.handle = handle
            self.written = 0
        def write(self, data):
            if baseline + self.written + len(data) >= cap_bytes:
                raise OSError("Checkpoint write would reach the working storage cap")
            count = self.handle.write(data)
            self.written += count
            return count
        def flush(self):
            self.handle.flush()
        def tell(self):
            return self.handle.tell()

    with temp.open("xb") as handle:
        torch.save(state, LimitedWriter(handle))
        handle.flush()
        os.fsync(handle.fileno())
    if directory_bytes(root) >= cap_bytes:
        raise OSError("Working storage changed during save; preserved temporary checkpoint")
    relative = str(path.relative_to(root))
    # Preserve the immediately preceding version when the same output resumes.
    if path.exists():
        previous = path.with_name(path.stem + "-previous.pt")
        if previous.exists() and str(previous.relative_to(root)) not in entries:
            raise FileExistsError("Unmanaged previous checkpoint; refusing replacement")
        os.replace(path, previous)
        prev_name = str(previous.relative_to(root))
        entries = [name for name in entries if name not in (relative, prev_name)] + [prev_name]
    os.replace(temp, path)
    entries = [name for name in entries if name != relative] + [relative]
    expired, kept = entries[:-2], entries[-2:]
    registry.write_text(json.dumps(kept, indent=2) + "\n")
    for name in expired:
        (root / name).unlink(missing_ok=True)
    return {"cap_bytes": cap_bytes, "keep_last": 2, "managed_checkpoints": kept,
            "working_bytes": directory_bytes(root)}
