"""CPU-only synthetic tests for pinned upstream revision provenance."""
import hashlib
import json

import pytest

from scripts import stage1_runtime as runtime


def fixture_checkpoint(tmp_path, monkeypatch):
    repository = "synthetic/Qwen-fixture"
    revision = "a" * 40
    files = {"one.safetensors": b"synthetic shard one", "two.safetensors": b"synthetic shard two"}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "synthetic"}))
    monkeypatch.setattr(runtime, "UPSTREAM_LFS_PROVENANCE", {
        "repository_id": repository, "revision": revision, "weight_sha256": hashes,
    })
    return repository, revision, hashes


def test_exact_upstream_repository_revision_and_all_shards_pass(tmp_path, monkeypatch):
    repository, revision, hashes = fixture_checkpoint(tmp_path, monkeypatch)
    result = runtime.verify_checkpoint(tmp_path, revision, hashes, repository_id=repository)
    assert result["passed"]
    assert result["revision_source"] == "official_hf_lfs_sha256_allowlist"
    assert result["repository_id"] == repository
    assert result["measured_revision"] == revision
    assert set(result["weight_sha256"]) == set(hashes)
    assert all(item["matches"] for item in result["weight_sha256"].values())


@pytest.mark.parametrize("repository,revision,change", [
    ("synthetic/other", "pinned", None),
    ("pinned", "b" * 40, None),
    ("pinned", "short", None),
    ("pinned", "pinned", "missing-shard"),
])
def test_upstream_fallback_rejects_unpinned_claims(tmp_path, monkeypatch, repository, revision, change):
    pinned_repository, pinned_revision, hashes = fixture_checkpoint(tmp_path, monkeypatch)
    requested_repository = pinned_repository if repository == "pinned" else repository
    requested_revision = pinned_revision if revision == "pinned" else revision
    requested_hashes = dict(hashes)
    if change == "missing-shard":
        requested_hashes.pop("two.safetensors")
    result = runtime.verify_checkpoint(tmp_path, requested_revision, requested_hashes,
                                       repository_id=requested_repository)
    assert not result["passed"]
    assert result["revision_source"] is None


def test_upstream_fallback_still_hashes_every_file(tmp_path, monkeypatch):
    repository, revision, hashes = fixture_checkpoint(tmp_path, monkeypatch)
    (tmp_path / "two.safetensors").write_bytes(b"tampered synthetic shard")
    result = runtime.verify_checkpoint(tmp_path, revision, hashes, repository_id=repository)
    assert result["revision_source"] == "official_hf_lfs_sha256_allowlist"
    assert result["revision_matches"] is False
    assert result["weights_match"] is False
    assert not result["passed"]


def test_existing_config_and_download_metadata_paths_keep_priority(tmp_path, monkeypatch):
    repository, revision, hashes = fixture_checkpoint(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"_commit_hash": revision}))
    config_result = runtime.verify_checkpoint(tmp_path, revision, hashes)
    assert config_result["passed"] and config_result["revision_source"] == "config.json"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "synthetic"}))
    metadata = tmp_path / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(revision + "\n")
    cache_result = runtime.verify_checkpoint(tmp_path, revision, hashes)
    assert cache_result["passed"] and cache_result["revision_source"] == "hf_download_metadata"
    (tmp_path / "config.json").write_text(json.dumps({"_commit_hash": "b" * 40}))
    conflict = runtime.verify_checkpoint(tmp_path, revision, hashes, repository_id=repository)
    assert not conflict["passed"] and conflict["revision_source"] == "config.json"


def test_production_allowlist_matches_independently_verified_pin():
    assert runtime.UPSTREAM_LFS_PROVENANCE == {
        "repository_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "weight_sha256": {
            "model-00001-of-00002.safetensors": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
            "model-00002-of-00002.safetensors": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
        },
    }
