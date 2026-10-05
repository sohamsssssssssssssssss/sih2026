"""Revision proof regression tests: synthetic files only, no model loading."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_stage1_smoke as smoke, run_stage1_t4 as t4

PIN = "6" * 40


@pytest.fixture
def proof(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    package = tmp_path / "package"
    checkpoint.mkdir()
    package.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic weights")
    (checkpoint / "config.json").write_text("{}")
    config = yaml.safe_load(Path("packages/stage1-t4-smoke/config.yaml").read_text())
    config.update(revision=PIN, weight_sha256={"model.safetensors": hashlib.sha256(b"synthetic weights").hexdigest()})
    (package / "config.yaml").write_text(yaml.safe_dump(config))
    (package / "captions.json").write_text(json.dumps({"split_name": "test/train", "captions": [{"output": "test"}]}))
    (package / "revision.json").write_text(json.dumps(config))
    return checkpoint, package, tmp_path / "results"


def metadata(checkpoint, revision=PIN):
    path = checkpoint / ".cache/huggingface/download/config.json.metadata"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(revision + "\nblob-etag\n0\n")


def verify(proof):
    checkpoint, package, out = proof
    return t4.stage_verify(checkpoint, package / "revision.json", package, out)


def test_local_dir_metadata_proves_revision(proof):
    metadata(proof[0])
    result = verify(proof)
    assert result["passed"]
    assert result["revision_source"] == "hf_download_metadata"
    assert result["weights_match"] and result["revision_matches"]


def test_missing_source_fails(proof):
    result = verify(proof)
    assert not result["passed"]
    assert result["revision_source"] is None
    assert result["weights_match"] and not result["revision_matches"]


def test_metadata_mismatch_fails(proof):
    metadata(proof[0], "7" * 40)
    result = verify(proof)
    assert not result["passed"]
    assert result["revision_source"] == "hf_download_metadata"
    assert not result["revision_matches"]


def test_weight_mismatch_fails(proof):
    metadata(proof[0])
    (proof[0] / "model.safetensors").write_bytes(b"wrong")
    result = verify(proof)
    assert not result["passed"]
    assert result["revision_matches"] and not result["weights_match"]


def test_smoke_missing_source_never_reaches_model(proof, monkeypatch):
    checkpoint, package, out = proof
    def forbidden(*args, **kwargs):
        raise RuntimeError("MODEL CONSTRUCTION REACHED")
    monkeypatch.setattr(smoke, "build_model", forbidden)
    with pytest.raises(ValueError, match="revision proof"):
        smoke.run(package / "config.yaml", checkpoint, package, out / "smoke.json")
    result = json.loads((out / "smoke.json").read_text())
    assert not result["passed"] and result["revision_source"] is None


@pytest.mark.parametrize("config_hash", [PIN, "7" * 40, None])
def test_config_source_has_priority(proof, config_hash):
    metadata(proof[0])
    (proof[0] / "config.json").write_text(json.dumps({"_commit_hash": config_hash}))
    result = verify(proof)
    assert result["revision_source"] == "config.json"
    assert result["passed"] is (config_hash == PIN)


def test_missing_weight_file_is_recorded_failure(proof):
    from scripts.stage1_runtime import verify_checkpoint
    metadata(proof[0])
    result = verify_checkpoint(proof[0], PIN, {"absent.safetensors": "0" * 64})
    assert result["revision_matches"] and not result["weights_match"]
    assert not result["passed"]


def test_empty_expected_weights_never_passes(proof):
    from scripts.stage1_runtime import verify_checkpoint
    metadata(proof[0])
    assert not verify_checkpoint(proof[0], PIN, {})["passed"]


def test_checkpoint_rotation_and_cap(tmp_path, monkeypatch):
    import torch
    from scripts import stage1_runtime as runtime
    state = {"model": torch.arange(10)}
    # Assert rotation actions without executing any deletion even of fixtures.
    deleted = []
    monkeypatch.setattr(Path, "unlink", lambda self, **kwargs: deleted.append(self.name))
    for name in ("one", "two", "retry"):
        result = runtime.bounded_checkpoint_save(state, tmp_path / f"{name}.pt")
    assert result["managed_checkpoints"] == ["two.pt", "retry.pt"]
    assert deleted == ["one.pt"]
    assert (tmp_path / "one.pt").exists()  # mock; no deletion performed
    evidence = tmp_path / "failed-attempt.json"
    evidence.write_text("{}")
    before = runtime.directory_bytes(tmp_path)
    with pytest.raises(OSError, match="cap"):
        runtime.bounded_checkpoint_save(state, tmp_path / "too-large.pt", cap_bytes=before + 64)
    assert evidence.exists() and not (tmp_path / "too-large.pt").exists()


def test_cap_counts_non_checkpoint_retry_files(tmp_path):
    from scripts.stage1_runtime import bounded_checkpoint_save
    (tmp_path / "retry.log").write_bytes(b"x" * 2000)
    with pytest.raises(OSError, match="cap"):
        bounded_checkpoint_save({}, tmp_path / "new.pt", cap_bytes=2000)


def test_large_checkpoint_cannot_cross_cap_and_failure_is_retained(tmp_path):
    import torch
    from scripts.stage1_runtime import bounded_checkpoint_save, directory_bytes
    with pytest.raises((OSError, RuntimeError)):
        bounded_checkpoint_save({"model": torch.zeros(100000)}, tmp_path / "large.pt", cap_bytes=100000)
    assert directory_bytes(tmp_path) < 100000
    assert (tmp_path / "large.pt.tmp").exists()
    assert not (tmp_path / "large.pt").exists()
    with pytest.raises(FileExistsError, match="Preserved failed"):
        bounded_checkpoint_save({}, tmp_path / "large.pt")


def test_result_growth_obeys_same_cap(tmp_path):
    from scripts.stage1_runtime import bounded_write_json
    path = tmp_path / "evidence.json"
    bounded_write_json(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(OSError, match="storage cap"):
        bounded_write_json(path, {"log": "x" * 2000}, cap_bytes=1000)
    assert path.read_bytes() == before


def test_failed_proof_is_not_overwritten(proof):
    first = verify(proof)
    path = proof[2] / "stage-verify.json"
    before = path.read_bytes()
    assert not first["passed"]
    metadata(proof[0])
    with pytest.raises(FileExistsError, match="prior verification"):
        verify(proof)
    assert path.read_bytes() == before
