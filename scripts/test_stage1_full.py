"""CPU tests for the Stage-1 full training stage.

No weights are loaded and no shard is read from the real archive: shards are
built from the synthetic fixture in data/test_bigearthnet_shards.py and the
model, tokenizer, and prompt builder are replaced with deterministic doubles.
The assertions are about the stage's own bookkeeping (sequence, checkpoints,
guard, provenance), never about learning.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from data.bigearthnet_shards import ShardDataset, build_shards

from scripts import run_stage1_full as full

CONTRACT = Path("data/manifests/bigearthnet/norm_contract.json")
THROUGHPUT_RATE = 4.0          # fake measured samples/s for the mocked stage
EFFECTIVE_BATCH = 2            # batch_size 1 x gradient_accumulation_steps 2 (fixture)
THROUGHPUT_BUDGET_FOR_STEPS = 4.0    # floor(4.0 x 4.0 / 2) = 8 planned steps
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")


class FakeTokenizer:
    """Maps a caption to a fixed three-token id list."""

    def __call__(self, text, add_special_tokens=False):
        class Output:
            input_ids = [7] * min(3, max(1, len(text) // 8))
        return Output()

    def convert_tokens_to_ids(self, token):
        return 5


class FakeLoss:
    def __init__(self, value):
        self.loss = value


class FakeModel(torch.nn.Module):
    """Deterministic loss that depends only on the sample and one parameter."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.training_steps = 0

    def forward(self, input_ids=None, pixel_values=None, labels=None, **kwargs):
        fingerprint = pixel_values[0, 0].abs().sum()
        target = labels[labels != -100].sum().float()
        return FakeLoss((fingerprint + target + self.scale) * 1e-6)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def eval(self):
        return super().train(False)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """A ready-to-run stage: synthetic shards, fake model, fake tokenizer."""
    from data.test_bigearthnet_shards import _fixture

    train_ids, _eval_ids, rows, parts, captions = _fixture(tmp_path, per_tile=3, eval_patches=3)
    train_dir = tmp_path / "train-shards"
    eval_dir = tmp_path / "eval-shards"
    build_shards(parts, train_ids, captions, rows, train_dir,
                 git_sha="test-sha", seed=3, patches_per_shard=4)
    eval_rows = {key: value for key, value in rows.items() if value["geo_split"] == "eval"}
    eval_ids = [key for key, value in rows.items() if value["geo_split"] == "eval"]
    build_shards(parts, eval_ids, captions, eval_rows, eval_dir,
                 git_sha="test-sha", seed=3, side="eval", patches_per_shard=4)
    config = {
        "name": "stage-1-full-test",
        "git_sha": "test-sha",
        "revision": "0" * 40,
        "weight_sha256": {"model.safetensors": "0" * 64},
        "seed": 7,
        "device": "cpu",
        "precision": "fp32",
        "attention": "sdpa",
        "optimizer": "adamw",
        "learning_rates": {"vision": 1e-5, "patch_embed": 1e-4, "lm_lora": 1e-4},
        "warmup_fraction": 0.03,
        "grad_clip_norm": 1.0,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        # Loop 9: the step count comes from measured throughput, not the config.
        "steps": None,
        "max_steps_per_session": None,
        "budget_seconds": THROUGHPUT_BUDGET_FOR_STEPS,
        "checkpoint_every_steps": 2,
        "eval_every_steps": 4,
        "eval_patches": 2,
        "wall_clock_limit_seconds": 39600,
        "wall_clock_reserve_seconds": 300,
        "contract": str(CONTRACT),
        "resume": False,
    }
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"_commit_hash": config["revision"]}))
    (weights / "model.safetensors").write_bytes(b"synthetic")
    config["weight_sha256"] = {"model.safetensors": hashlib.sha256(b"synthetic").hexdigest()}
    config_path = tmp_path / "config-full.yaml"
    config_path.write_text(yaml.safe_dump(config))

    # A fresh model per run, exactly like build_model: reusing one instance would
    # let a later session start from an already-updated parameter.
    def fake_build_model(_config, _checkpoint, _dtype, _device):
        model = FakeModel()
        groups = [{"name": "vision", "params": [model.scale], "lr": 1e-5}]
        return model, groups

    monkeypatch.setattr(full, "build_model", fake_build_model)
    monkeypatch.setattr(full.AutoTokenizer, "from_pretrained",
                        lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(full, "build_prompt_ids", lambda *_args, **_kwargs: [1, 5, 5, 2])
    monkeypatch.setattr(full, "load_norm_contract", lambda _path: json.loads(CONTRACT.read_text()))
    manifest_sha = ShardDataset(train_dir / "shards-manifest.json", seed=7).manifest_sha256
    throughput = _throughput(tmp_path, rate=THROUGHPUT_RATE, manifest_sha=manifest_sha,
                             config_sha256=full.config_identity(config), device=config["device"], dtype=config["precision"])
    return {"tmp_path": tmp_path, "config": config_path, "train_shards": train_dir,
            "eval_shards": eval_dir, "checkpoint_dir": tmp_path / "weights",
            "throughput": throughput, "manifest_sha": manifest_sha}


def _run(stage, name="run", **kwargs):
    output = stage["tmp_path"] / f"{name}.json"
    kwargs.setdefault("throughput_json", stage["throughput"])
    result = full.run_full(stage["config"], stage["checkpoint_dir"], stage["train_shards"],
                           output, eval_shards=stage["eval_shards"], **kwargs)
    return result, output


def _refresh_throughput(stage) -> None:
    """Re-stamp the throughput record for the current config identity."""
    config = yaml.safe_load(stage["config"].read_text())
    stage["throughput"].write_text(json.dumps({
        "kind": "stage-1 smoke throughput",
        "measured_samples_per_second": THROUGHPUT_RATE,
        "shard_manifest_sha256": stage["manifest_sha"],
        "config_sha256": full.config_identity(config),
        "device": config["device"], "dtype": config["precision"], "source": "stage-smoke.json",
    }))


def _set_planned_steps(stage, steps: int) -> None:
    """Size the total plan through the throughput formula."""
    config = yaml.safe_load(stage["config"].read_text())
    config["budget_seconds"] = steps * EFFECTIVE_BATCH / THROUGHPUT_RATE
    stage["config"].write_text(yaml.safe_dump(config))
    _refresh_throughput(stage)


def _set_session_cap(stage, steps: int) -> None:
    """Cap one session's steps through the wall-clock guard, leaving the plan alone.

    A resumed session must keep the same total plan (the cosine schedule is built
    on it), so only the per-session cap moves between the interrupted and
    uninterrupted runs.
    """
    config = yaml.safe_load(stage["config"].read_text())
    reserve = float(config["wall_clock_reserve_seconds"])
    config["wall_clock_limit_seconds"] = reserve + (steps * EFFECTIVE_BATCH / THROUGHPUT_RATE)
    stage["config"].write_text(yaml.safe_dump(config))
    _refresh_throughput(stage)


def test_full_run_records_provenance_and_losses(stage):
    result, output = _run(stage)
    assert result["status"] == "completed"
    assert result["steps_completed"] == 8
    assert result["n"] == 6
    assert result["device"] == "cpu" and result["dtype"] == "fp32"
    assert result["revision"] == "0" * 40
    assert result["shard_manifest_sha256"] == ShardDataset(
        stage["train_shards"] / "shards-manifest.json", seed=7).manifest_sha256
    assert result["eval_shard_manifest_sha256"] == ShardDataset(
        stage["eval_shards"] / "shards-manifest.json", seed=7).manifest_sha256
    assert [entry["step"] for entry in result["losses"]] == list(range(1, 9))
    assert all(entry["samples_per_second"] > 0 for entry in result["losses"])
    assert all(entry["lr"] is not None for entry in result["losses"])
    assert json.loads(output.read_text())["steps_completed"] == 8


def test_checkpoint_holds_position_optimizer_and_rng(stage):
    result, _output = _run(stage)
    saved = torch.load(stage["tmp_path"] / "run.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 8
    assert saved["config_sha256"] == full.config_identity(
        yaml.safe_load(stage["config"].read_text()))
    assert set(saved["data_position"]) >= {"seed", "epoch", "cursor", "shard_manifest_sha256"}
    assert saved["optimizer"]["state"], "optimizer state must not be empty"
    assert saved["scheduler"]["last_epoch"] == 8
    assert saved["torch_rng_state"] is not None
    assert result["last_checkpoint"].endswith("run.pt")


def test_resume_reproduces_the_uninterrupted_sample_sequence(stage):
    full_result, _ = _run(stage, name="whole")
    assert full_result["steps_planned"] == 8
    _set_session_cap(stage, 3)
    part, _ = _run(stage, name="part")
    assert part["steps_completed"] == 3
    _set_session_cap(stage, 8)
    resumed = full.run_full(stage["config"], stage["checkpoint_dir"], stage["train_shards"],
                            stage["tmp_path"] / "part.json",
                            eval_shards=stage["eval_shards"],
                            resume_from=stage["tmp_path"] / "part.pt",
                            throughput_json=stage["throughput"])
    assert resumed["resumed_from_step"] == 3
    assert resumed["steps_completed"] == 8
    assert resumed["sample_sequence_digest"] == full_result["sample_sequence_digest"]
    assert [entry["step"] for entry in resumed["losses"]] == [4, 5, 6, 7, 8]
    assert [entry["loss"] for entry in resumed["losses"]] == \
           [entry["loss"] for entry in full_result["losses"][3:]]


def test_cli_entry_points_run_from_the_repository_root():
    """The documented interfaces must work when invoked as scripts, not imports."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    for script in ("scripts/run_stage1_full.py", "scripts/build_stage1_shards.py"):
        completed = subprocess.run([sys.executable, str(root / script), "--help"],
                                   cwd=root, capture_output=True, text=True)
        assert completed.returncode == 0, f"{script}: {completed.stderr[-400:]}"
        assert "--help" in completed.stdout or "usage:" in completed.stdout


def test_config_identity_ignores_session_keys_only(stage):
    config = yaml.safe_load(stage["config"].read_text())
    baseline = full.config_identity(config)
    for key in full.SESSION_ONLY_KEYS:
        changed = dict(config)
        changed[key] = "changed"
        assert full.config_identity(changed) == baseline
    changed = dict(config)
    changed["grad_clip_norm"] = 0.25
    assert full.config_identity(changed) != baseline


def test_resume_rejects_a_foreign_config(stage):
    _set_session_cap(stage, 2)
    _run(stage, name="part")
    config = yaml.safe_load(stage["config"].read_text())
    config["grad_clip_norm"] = 0.5
    stage["config"].write_text(yaml.safe_dump(config))
    _refresh_throughput(stage)
    with pytest.raises(ValueError, match="different config"):
        full.run_full(stage["config"], stage["checkpoint_dir"], stage["train_shards"],
                      stage["tmp_path"] / "again.json",
                      resume_from=stage["tmp_path"] / "part.pt",
                      throughput_json=stage["throughput"])


def test_wall_clock_guard_exits_cleanly_with_a_final_checkpoint(stage):
    clock = {"value": 1000.0}

    def ticking_now():
        clock["value"] += 100.0
        return clock["value"]

    result, output = _run(stage, name="guarded", wall_clock_limit_seconds=700,
                          now=ticking_now)
    assert result["status"] == "wall_clock_guard_exit"
    assert result["finished"] is True
    assert result["steps_completed"] < result["steps_requested"]
    assert result["wall_clock_stopped_at_step"] == result["steps_completed"]
    assert result["steps_requested"] == 8
    saved = torch.load(stage["tmp_path"] / "guarded.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == result["steps_completed"]
    assert json.loads(output.read_text())["status"] == "wall_clock_guard_exit"


def test_periodic_eval_uses_eval_shards_only(stage):
    result, _ = _run(stage)
    assert [entry["step"] for entry in result["eval_losses"]] == [4, 8]
    assert all(entry["n"] == 2 for entry in result["eval_losses"])
    assert all(entry["eval_split_name"].endswith("/eval") for entry in result["eval_losses"])
    eval_tiles = ShardDataset(stage["eval_shards"] / "shards-manifest.json", seed=7).tiles
    train_tiles = ShardDataset(stage["train_shards"] / "shards-manifest.json", seed=7).tiles
    assert eval_tiles and not (eval_tiles & train_tiles)


def test_nonfinite_loss_aborts_without_writing_a_checkpoint(stage, monkeypatch):
    def exploding(self, **kwargs):
        return FakeLoss(torch.tensor(float("nan")))

    monkeypatch.setattr(FakeModel, "forward", exploding)
    with pytest.raises(FloatingPointError):
        _run(stage, name="nan")
    assert not (stage["tmp_path"] / "nan.pt").exists()


def test_missing_shard_manifest_is_rejected(stage):
    with pytest.raises(FileNotFoundError):
        full.run_full(stage["config"], stage["checkpoint_dir"],
                      stage["tmp_path"] / "absent", stage["tmp_path"] / "x.json")

# --------------------------------------------------------------------------- #
# plan from measured throughput (loop 9)
# --------------------------------------------------------------------------- #
def _throughput(tmp_path, *, rate=1.5, manifest_sha=None, config_sha=None, **extra):
    payload = {
        "kind": "stage-1 smoke throughput",
        "measured_samples_per_second": rate,
        "shard_manifest_sha256": manifest_sha or "0" * 64,
        "config_sha256": config_sha or "0" * 64,
        "device": "cuda", "dtype": "fp16",
        "source": "stage-smoke.json",
        "timestamp_utc": "2026-10-05T00:00:00+00:00",
    }
    payload.update(extra)
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(payload))
    return path


def _plan_config(**overrides):
    config = {"gradient_accumulation_steps": 8, "batch_size": 1,
              "budget_seconds": 3600.0, "wall_clock_limit_seconds": 39600,
              "wall_clock_reserve_seconds": 300}
    config.update(overrides)
    return config


def test_plan_arithmetic_is_floored_and_explicit():
    config = _plan_config()
    plan = full.plan_from_throughput(config, {"measured_samples_per_second": 1.5,
                                              "shard_manifest_sha256": "a" * 64,
                                              "config_sha256": "b" * 64}, n_available=150000)
    # 3600 x 1.5 / 8 = 675 exactly; 39600-300 = 39300 x 1.5 / 8 = 7368.75 -> 7368
    assert plan["effective_batch"] == 8
    assert plan["max_optimizer_steps"] == 675
    assert plan["max_steps_per_session"] == 7368
    assert plan["n_used"] == 150000
    assert plan["samples_covered"] == 675 * 8
    assert plan["epochs"] == pytest.approx((675 * 8) / 150000)


def test_plan_budget_override_and_zero_step_refusal():
    with pytest.raises(ValueError, match="zero optimizer steps"):
        full.plan_from_throughput(_plan_config(), {"measured_samples_per_second": 0.0001,
                                                  "shard_manifest_sha256": "a" * 64,
                                                  "config_sha256": "b" * 64}, n_available=150000)
    plan = full.plan_from_throughput(_plan_config(), {"measured_samples_per_second": 1.5,
                                                      "shard_manifest_sha256": "a" * 64,
                                                      "config_sha256": "b" * 64},
                                     n_available=100, budget_seconds=800.0)
    assert plan["max_optimizer_steps"] == int(800 * 1.5 // 8) == 150
    assert plan["budget_seconds"] == 800.0


def test_throughput_loader_rejects_missing_and_mismatched_records(tmp_path):
    with pytest.raises(FileNotFoundError):
        full.load_throughput(tmp_path / "absent.json", shard_manifest_sha256="a" * 64,
                             config_sha256="b" * 64)
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"measured_samples_per_second": 1.0}))
    with pytest.raises(ValueError, match="missing required fields"):
        full.load_throughput(path, shard_manifest_sha256="a" * 64, config_sha256="b" * 64)
    good = _throughput(tmp_path, manifest_sha="a" * 64, config_sha="b" * 64)
    assert full.load_throughput(good, shard_manifest_sha256="a" * 64,
                                config_sha256="b" * 64)["measured_samples_per_second"] == 1.5
    with pytest.raises(ValueError, match="different shard manifest"):
        full.load_throughput(good, shard_manifest_sha256="c" * 64, config_sha256="b" * 64)
    with pytest.raises(ValueError, match="different training config"):
        full.load_throughput(good, shard_manifest_sha256="a" * 64, config_sha256="d" * 64)
    zero = _throughput(tmp_path, rate=0.0, manifest_sha="a" * 64, config_sha256="b" * 64)
    with pytest.raises(ValueError, match="unusable samples_per_second"):
        full.load_throughput(zero, shard_manifest_sha256="a" * 64, config_sha256="b" * 64)


def test_run_full_refuses_without_throughput_and_writes_the_plan_first(stage):
    manifest_sha = ShardDataset(stage["train_shards"] / "shards-manifest.json", seed=7).manifest_sha256
    config = yaml.safe_load(stage["config"].read_text())
    config_sha = full.config_identity(config)
    config["steps"] = None
    config["max_steps_per_session"] = None
    config["budget_seconds"] = 60.0
    stage["config"].write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="throughput JSON is required"):
        full.run_full(stage["config"], stage["checkpoint_dir"], stage["train_shards"],
                      stage["tmp_path"] / "notrain.json")
    assert not (stage["tmp_path"] / "notrain.json").exists()

    path = _throughput(stage["tmp_path"], rate=4.0, manifest_sha=manifest_sha,
                       config_sha256=config_sha, device=config["device"], dtype=config["precision"])
    result, output = _run(stage, name="planned", throughput_json=path)
    assert result["plan"]["measured_samples_per_second"] == 4.0
    assert result["steps_planned"] == int(60.0 * 4.0 // 2)
    assert result["n_used"] == result["n"]
    assert result["plan"]["epochs"] == pytest.approx(result["steps_planned"] * 2 / result["n"])
    assert result["plan"]["throughput_shard_manifest_sha256"] == manifest_sha
    assert result["plan"]["formula"].startswith("max_optimizer_steps = floor(")
    written = json.loads(output.read_text())
    assert written["plan"] == result["plan"], "the plan must be in the JSON before training"
    assert written["steps_completed"] >= 1


def test_config_identity_ignores_the_null_step_keys(stage):
    config = yaml.safe_load(stage["config"].read_text())
    baseline = full.config_identity(config)
    for key, value in (("steps", None), ("max_steps_per_session", None), ("budget_seconds", 10)):
        changed = dict(config)
        changed[key] = value
        assert full.config_identity(changed) == baseline


def test_config_full_yaml_has_no_placeholder_steps():
    import yaml as _yaml

    config = _yaml.safe_load(
        Path("packages/stage1-t4-smoke/config-full.yaml").read_text())
    assert config["steps"] is None
    assert config["max_steps_per_session"] is None
    assert config["budget_seconds"] > 0


def test_caption_only_mask_includes_eos(stage):
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text.endswith("<|im_end|>")
            return type("Tokens", (), {"input_ids": [7, 8, 9]})()

    dataset = ShardDataset(stage["train_shards"] / "shards-manifest.json", seed=7)
    sample = full.make_sample(dataset, 0, [1, 5, 2], Tokenizer(),
                              json.loads(CONTRACT.read_text()), "cpu", torch.float32, False)
    assert sample["labels"].tolist() == [[-100, -100, -100, 7, 8, 9]]
    assert sample["input_ids"].tolist() == [[1, 5, 2, 7, 8, 9]]
    assert sample["attention_mask"].all()  # batch=1 has no padding


def test_checkpoint_complete_and_resume_lr_weights_exact(stage):
    whole, _ = _run(stage, name="whole-exact")
    whole_state = torch.load(stage["tmp_path"] / "whole-exact.pt", weights_only=False)
    _set_session_cap(stage, 3)
    _run(stage, name="part-exact")
    part_state = torch.load(stage["tmp_path"] / "part-exact.pt", weights_only=False)
    assert set(part_state) >= {"scaler", "numpy_rng_state", "python_rng_state", "cuda_rng_state",
                               "model", "optimizer", "scheduler", "data_position", "step", "total_steps"}
    _set_session_cap(stage, 8)
    resumed, _ = _run(stage, name="part-exact", resume_from=stage["tmp_path"] / "part-exact.pt")
    assert [(x["loss"], x["lr"]) for x in resumed["losses"]] == [
        (x["loss"], x["lr"]) for x in whole["losses"][3:]]
    final = torch.load(stage["tmp_path"] / "part-exact.pt", weights_only=False)
    assert final["sample_sequence_digest"] == whole_state["sample_sequence_digest"]
    for key in whole_state["model"]:
        torch.testing.assert_close(final["model"][key], whole_state["model"][key], rtol=0, atol=0)


def test_scheduler_preserves_distinct_group_ratios():
    groups = [{"name": name, "params": [torch.nn.Parameter(torch.ones(1))], "lr": lr}
              for name, lr in (("vision", 1e-5), ("patch_embed", 1e-4), ("lm_lora", 1e-4))]
    optimizer, scheduler = full.build_optimizer_scheduler(
        {"optimizer": "adamw", "warmup_fraction": 0.03}, groups, 100)
    for _ in range(99):
        lrs = scheduler.get_last_lr()
        assert lrs[1] == pytest.approx(lrs[0] * 10)
        assert lrs[2] == lrs[1]
        optimizer.step()
        scheduler.step()


def test_sharded_smoke_produces_full_accepted_throughput(stage):
    path = stage["tmp_path"] / "measured.json"
    record = full.measure_shard_throughput(stage["config"], stage["checkpoint_dir"],
                                         stage["train_shards"], path, steps=2)
    assert record["steps_measured"] == 2 and record["samples_measured"] == 4
    assert len(record["sequence_lengths"]) == 4
    assert record["peak_memory_bytes"] > 0 and record["gpu"] is None
    assert full.load_throughput(path, shard_manifest_sha256=stage["manifest_sha"],
                                config_sha256=record["config_sha256"]) == record
    config = yaml.safe_load(stage["config"].read_text())
    config["budget_seconds"] = 2 * EFFECTIVE_BATCH / record["measured_samples_per_second"]
    stage["config"].write_text(yaml.safe_dump(config))
    result, _ = _run(stage, name="from-producer", throughput_json=path)
    assert result["steps_completed"] == 2


def test_full_and_resume_rehash_before_model(stage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model must not be loaded")
    monkeypatch.setattr(full, "build_model", forbidden)
    (stage["checkpoint_dir"] / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="revision proof"):
        _run(stage, name="tampered", resume_from=stage["tmp_path"] / "nonexistent.pt")
    result = json.loads((stage["tmp_path"] / "tampered.json").read_text())
    assert not result["passed"] and not result["weights_match"]
    assert result["checkpoint_verification"]["verification_bytes"] == len(b"tampered")


def test_full_rejects_throughput_device_mismatch(stage):
    payload = json.loads(stage["throughput"].read_text())
    payload["device"] = "cuda"
    stage["throughput"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="device/dtype"):
        _run(stage)


def test_eval_no_grad_and_wall_budget_interrupt(stage):
    dataset = ShardDataset(stage["eval_shards"] / "shards-manifest.json", seed=7)
    class GuardedModel(FakeModel):
        def forward(self, **kwargs):
            assert not torch.is_grad_enabled()
            return super().forward(**kwargs)
    calls = []
    def should_stop():
        calls.append(True)
        return len(calls) > 1
    report = full.evaluate_caption_loss(GuardedModel(), dataset, [0, 1, 2], [1, 5, 2],
                                       FakeTokenizer(), json.loads(CONTRACT.read_text()),
                                       "cpu", torch.float32, False, should_stop=should_stop)
    assert report["n"] == 1
