from hashlib import sha256

import pytest

from kaggle.grounding_qwen_inputs import find_qwen_directory, verify_qwen_directory
from kaggle.run_grounding_baselines import discover_pack


@pytest.mark.parametrize("changes, error", [
    ({}, None),
    ({"b": b"wrong size"}, "size mismatch for b"),
    ({"a": b"X"}, "SHA-256 mismatch for a"),
    ({"extra": b"X"}, r"extra=\['extra'\]"),
    ({"a": None}, r"missing=\['a'\]"),
])
def test_qwen_inventory(tmp_path, changes, error):
    original = {"a": b"A", "b": b"B"}
    pins = {name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
            for name, data in original.items()}
    for name, data in (original | changes).items():
        if data is not None:
            (tmp_path / name).write_bytes(data)
    if error:
        with pytest.raises(ValueError, match=error):
            verify_qwen_directory(tmp_path, pins)
    else:
        verify_qwen_directory(tmp_path, pins)


@pytest.mark.parametrize("prefix", ["", "datasets/sohamssssss"])
def test_kaggle_mounts(tmp_path, prefix):
    root = tmp_path / prefix
    qwen = root / "satquery-qwen-pinned"
    pack = root / "satquery-grounding-baseline-pack-zip"
    qwen.mkdir(parents=True)
    pack.mkdir()
    (qwen / "model-00001-of-00002.safetensors").touch()
    (pack / "MANIFEST.json").touch()
    assert find_qwen_directory(tmp_path) == qwen
    assert discover_pack(tmp_path) == pack
