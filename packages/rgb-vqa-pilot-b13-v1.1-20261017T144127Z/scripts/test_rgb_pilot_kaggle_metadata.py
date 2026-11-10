"""CPU regression: upload metadata may be absent from Kaggle runtime mounts."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import rgb_pilot_package as package


def test_kaggle_mount_without_dataset_metadata_keeps_runtime_hashes_required(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.txt"
    runtime.write_bytes(b"synthetic runtime payload")
    metadata_hash = hashlib.sha256(b"synthetic upload metadata").hexdigest()
    (tmp_path / "package-manifest.json").write_text(json.dumps({
        "payload_sha256": {"runtime.txt": hashlib.sha256(runtime.read_bytes()).hexdigest()},
        "upload_metadata_sha256": {"dataset-metadata.json": metadata_hash},
    }))
    # The Kaggle mount intentionally omits dataset-metadata.json.
    monkeypatch.setattr(package, "validate_pilot", lambda *args: ([], [], {}))
    b13 = tmp_path / "data/manifests/rsvqa/b13-6k.v1"
    b13.mkdir(parents=True)
    (b13 / "train.jsonl").write_text("")
    (b13 / "provenance.json").write_text(json.dumps({"selected_image_sha256": {}}))
    (tmp_path / "pilot-config.json").write_text("{}")
    monkeypatch.setattr(package, "check_b13_spec", lambda *args: None)
    assert not (tmp_path / "dataset-metadata.json").exists()
    assert package.check_payload(tmp_path)["upload_metadata_sha256"]["dataset-metadata.json"] == metadata_hash

    runtime.unlink()
    with pytest.raises(FileNotFoundError):
        package.check_payload(tmp_path)
