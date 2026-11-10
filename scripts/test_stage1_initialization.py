"""Synthetic CPU fixtures: initialization identity, not pretrained evidence."""
import hashlib
import json

import pytest
import yaml

from models.qwen_vl.stage1 import patch_initialization, validate_initialization_metadata
from scripts import run_stage1_full as full
from scripts import run_stage1_smoke as smoke

RGB = "rgb_preserving_zero9_v1"


def test_legacy_full_identity_is_unchanged_and_new_identity_is_distinct():
    config = {"seed": 7, "steps": 200, "name": "synthetic"}
    historical = hashlib.sha256(json.dumps({"seed": 7}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert full.config_identity(config) == historical
    assert patch_initialization(config) == "mean_repeat_v1"
    assert full.config_identity(dict(config, patch_embed_initialization=RGB)) != historical
    full.validate_resume_identity({}, config)  # Legacy missing hash is preserved.


@pytest.mark.parametrize("recorded", [None, "mean_repeat_v1", "typo", RGB])
def test_new_full_resume_requires_both_identity_and_initialization(recorded):
    config = {"patch_embed_initialization": RGB}
    saved = {"config_sha256": full.config_identity(config), "patch_embed_initialization": recorded}
    if recorded != RGB:
        with pytest.raises(ValueError, match="initialization"):
            full.validate_resume_identity(saved, config)
    else:
        full.validate_resume_identity(saved, config)
        for identity in (None, "other"):
            with pytest.raises(ValueError, match="different config"):
                full.validate_resume_identity(dict(saved, config_sha256=identity), config)


def test_smoke_identity_legacy_digest_and_metadata_guards(tmp_path):
    config = {"captions": "captions.json", "contract": "contract.json"}
    (tmp_path / "captions.json").write_text("synthetic captions")
    (tmp_path / "contract.json").write_text("synthetic contract")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    historical = smoke.canonical_digest({
        "config": config, "total_steps": 200, "shuffle_images": False,
        "captions_sha256": hashlib.sha256(b"synthetic captions").hexdigest(),
        "contract_sha256": hashlib.sha256(b"synthetic contract").hexdigest()})
    assert smoke.checkpoint_identity(path, total_steps=200) == historical
    smoke.validate_checkpoint_identity({"config_identity": historical, "total_steps": 200}, path)
    config["patch_embed_initialization"] = RGB
    path.write_text(yaml.safe_dump(config))
    identity = smoke.checkpoint_identity(path, total_steps=200)
    assert identity != historical
    for saved in ({"config_identity": historical, "total_steps": 200},
                  {"config_identity": identity, "total_steps": 200}):
        with pytest.raises(ValueError):
            smoke.validate_checkpoint_identity(saved, path)
    smoke.validate_checkpoint_identity({"config_identity": identity, "total_steps": 200,
                                       "patch_embed_initialization": RGB}, path)
    with pytest.raises(ValueError, match="initialization"):
        validate_initialization_metadata({"patch_embed_initialization": RGB}, {})


def test_full_checkpoint_records_initialization(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    captured = {}
    def capture(state, path):
        captured.update(state)
        return {}
    monkeypatch.setattr(full, "bounded_checkpoint_save", capture)
    full.save_checkpoint(tmp_path / "synthetic.pt", model=model, optimizer=optimizer,
                         scheduler=scheduler, dataset=SimpleNamespace(state=lambda: {}), step=0,
                         config_sha256="synthetic", wall_clock_seconds=0, sample_sequence_digest="",
                         patch_embed_initialization=RGB)
    assert captured["patch_embed_initialization"] == RGB
