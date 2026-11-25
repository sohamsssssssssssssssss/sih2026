"""Verify the complete grounding Qwen dataset before opening model files."""
from hashlib import file_digest


def find_qwen_directory(root):
    mounts = [root / "satquery-qwen-pinned",
              root / "datasets/sohamssssss/satquery-qwen-pinned"]
    shards = [path for mount in mounts
              for path in mount.rglob("model-00001-of-00002.safetensors") if path.is_file()]
    if len(shards) != 1:
        raise ValueError(f"HARD STOP: expected one Qwen shard under the supported mounts, found {len(shards)}")
    return shards[0].parent


def verify_qwen_directory(directory, pins):
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    missing, extra = sorted(pins.keys() - actual), sorted(actual - pins.keys())
    if missing or extra:
        raise ValueError(f"HARD STOP: Qwen dataset file list mismatch (missing={missing}, extra={extra})")
    for name, expected in pins.items():
        size = (directory / name).stat().st_size
        if size != expected["bytes"]:
            raise ValueError(f"HARD STOP: Qwen size mismatch for {name} (expected {expected['bytes']}, got {size})")
    for name, expected in pins.items():
        with (directory / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"HARD STOP: Qwen SHA-256 mismatch for {name}")
