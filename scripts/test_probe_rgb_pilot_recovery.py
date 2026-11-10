"""Synthetic CPU recovery checks; no Qwen model, GPU, training, or checkpoint pickle load."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import probe_rgb_pilot_recovery as probe
from training.remote_sensing import TrainingConfig, sha256_file


def synthetic_checkpoint(path, *, scaler=False, step=16):
    path.mkdir(parents=True, exist_ok=True)
    names = set(probe.REQUIRED)
    if scaler:
        names.add("scaler.pt")
    for name in names:
        (path / name).write_bytes(f"synthetic {name}".encode())
    (path / "trainer_state.json").write_text(json.dumps({"max_steps": 128, "global_step": step}))
    marker = {
        "run_identity": "synthetic-identity", "step": step, "scaler_expected": scaler,
        "files_sha256": {name: sha256_file(path / name) for name in sorted(names)},
    }
    (path / "pilot-checkpoint.json").write_text(json.dumps(marker))
    return marker


@pytest.mark.parametrize("scaler", [False, True])
def test_checkpoint_requires_scaler_only_when_runtime_used_it(tmp_path, scaler):
    checkpoint = tmp_path / "checkpoint-16"
    synthetic_checkpoint(checkpoint, scaler=scaler)
    assert probe.verify_checkpoint(checkpoint, "synthetic-identity")["scaler_expected"] is scaler
    with pytest.raises(ValueError, match="identity"):
        probe.verify_checkpoint(checkpoint, "changed")
    (checkpoint / "optimizer.pt").write_bytes(b"tampered synthetic optimizer")
    with pytest.raises(ValueError, match="hash|inventory"):
        probe.verify_checkpoint(checkpoint, "synthetic-identity")


def test_checkpoint_rejects_missing_scaler_when_expected_and_unrecorded_extra(tmp_path):
    checkpoint = tmp_path / "checkpoint-16"
    synthetic_checkpoint(checkpoint, scaler=True)
    (checkpoint / "scaler.pt").unlink()
    with pytest.raises(ValueError, match="inventory"):
        probe.verify_checkpoint(checkpoint, "synthetic-identity")
    (checkpoint / "scaler.pt").write_bytes(b"synthetic scaler.pt")
    (checkpoint / "unrecorded.bin").write_bytes(b"synthetic extra")
    with pytest.raises(ValueError, match="inventory"):
        probe.verify_checkpoint(checkpoint, "synthetic-identity")


def test_checkpoint_rejects_missing_required_state_and_wrong_step(tmp_path):
    checkpoint = tmp_path / "checkpoint-16"
    marker = synthetic_checkpoint(checkpoint)
    marker["files_sha256"].pop("scheduler.pt")
    (checkpoint / "scheduler.pt").unlink()
    (checkpoint / "pilot-checkpoint.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="state missing"):
        probe.verify_checkpoint(checkpoint, "synthetic-identity")
    synthetic_checkpoint(checkpoint, step=17)
    with pytest.raises(ValueError, match="step mismatch"):
        probe.verify_checkpoint(checkpoint, "synthetic-identity")


def test_backup_is_separate_verified_and_read_only(tmp_path):
    checkpoint = tmp_path / "active/checkpoint-16"
    synthetic_checkpoint(checkpoint)
    backup = tmp_path / "backups/checkpoint-16"
    probe.make_backup(checkpoint, backup, "synthetic-identity")
    assert checkpoint != backup
    assert probe.verify_checkpoint(backup, "synthetic-identity")["step"] == 16
    assert backup.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in backup.rglob("*"))
    with pytest.raises(FileExistsError):
        probe.make_backup(checkpoint, backup, "synthetic-identity")
    for path in backup.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    backup.chmod(0o755)


def test_boundary_detects_replay_skip_and_reordering():
    expected = [f"synthetic-question-{index}" for index in range(1024)]
    boundary = probe.check_continuity(expected, expected[:128], expected[128:136])
    assert boundary["last_before_checkpoint"] == expected[120:128]
    assert boundary["first_after_resume"] == expected[128:136]
    for incorrect in (expected[127:135], expected[129:137], list(reversed(expected[128:136]))):
        with pytest.raises(ValueError, match="boundary"):
            probe.check_continuity(expected, expected[:128], incorrect)
    with pytest.raises(ValueError, match="Step-16"):
        probe.check_continuity(expected, expected[1:129], None)


def test_stop_controls_enforce_exact_steps_and_save_first_checkpoint():
    probe.check_start_step(0, "start")
    probe.check_start_step(16, "resume")
    with pytest.raises(RuntimeError, match="expected 16"):
        probe.check_start_step(0, "resume")
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    probe.stop_control(SimpleNamespace(global_step=15), control, 16)
    assert not control.should_training_stop
    probe.stop_control(SimpleNamespace(global_step=16), control, 16)
    assert control.should_training_stop and control.should_save
    resumed = SimpleNamespace(should_training_stop=False, should_save=False)
    probe.stop_control(SimpleNamespace(global_step=17), resumed, 17)
    assert resumed.should_training_stop and not resumed.should_save
    with pytest.raises(RuntimeError, match="exceeded"):
        probe.stop_control(SimpleNamespace(global_step=18), resumed, 17)


def test_probe_retains_128_step_schedule_with_synthetic_arguments(tmp_path, monkeypatch):
    module = ModuleType("transformers")
    module.TrainingArguments = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "transformers", module)
    cfg = TrainingConfig(model_path=tmp_path, dataset_manifest=tmp_path / "synthetic",
                         image_root=tmp_path, output_dir=tmp_path, max_samples=1024, max_steps=128)
    args = probe.training_arguments(cfg, tmp_path / "phase16")
    assert (args.max_steps, args.save_steps, args.per_device_train_batch_size,
            args.gradient_accumulation_steps, args.save_only_model,
            args.ignore_data_skip) == (128, 16, 1, 8, False, False)
    module.TrainingArguments = lambda **kwargs: SimpleNamespace(**{**kwargs, "max_steps": 16})
    with pytest.raises(ValueError, match="frozen training schedule"):
        probe.training_arguments(cfg, tmp_path / "phase16")


def test_timeout_report_and_existing_root_block_step_zero_restart(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as error:
        probe.main(["start", "--weights", str(tmp_path/"synthetic-weights"),
                    "--recovery-root", str(tmp_path/"uncreated"), "--max-seconds", "180"])
    assert error.value.code == 2
    assert not (tmp_path/"uncreated").exists()
    monkeypatch.setattr(probe, "binding", lambda root: "synthetic-identity")
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 180
        raise subprocess.TimeoutExpired(command, 180)
    monkeypatch.setattr(probe.subprocess, "run", timeout)
    root = tmp_path / "recovery"
    args = SimpleNamespace(phase="start", recovery_root=root, weights=tmp_path / "synthetic-weights",
                           max_seconds=180)
    assert probe.parent(args) == 1
    assert json.loads((root / "phase16-report.json").read_text())["status"] == "TIMEOUT"
    with pytest.raises(FileExistsError, match="step-0 restart"):
        probe.parent(args)


def test_synthetic_two_phase_parent_preserves_backup_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "binding", lambda root: "synthetic-identity")
    root = tmp_path / "recovery"
    expected = [f"synthetic-question-{index}" for index in range(1024)]

    def fake_worker(command, **kwargs):
        phase = command[2]
        assert kwargs["timeout"] == 180
        if phase == "start":
            checkpoint = root / "phase16-output/checkpoints/checkpoint-16"
            marker = synthetic_checkpoint(checkpoint, scaler=False)
            report = {
                "status": "PASS", "global_step": 16, "run_identity": "synthetic-identity",
                "expected_sample_ids": expected, "expected_order_sha256": probe.digest_ids(expected),
                "trained_sample_ids": expected[:128], "checkpoint_files_sha256": marker["files_sha256"],
            }
            target = root / "phase16-output/worker-report.json"
        else:
            assert phase == "resume"
            report = {
                "status": "PASS", "global_step": 17, "run_identity": "synthetic-identity",
                "expected_order_sha256": probe.digest_ids(expected),
                "trained_sample_ids": expected[128:136],
            }
            target = root / "phase17-output/worker-report.json"
        target.write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(probe.subprocess, "run", fake_worker)
    args = SimpleNamespace(phase="start", recovery_root=root, weights=tmp_path / "synthetic-weights",
                           max_seconds=180)
    assert probe.parent(args) == 0
    backup = root / "immutable-backup/checkpoint-16"
    assert backup.exists() and backup.stat().st_mode & 0o222 == 0
    assert json.loads((root / "phase16-report.json").read_text())["status"] == "PASS"
    args.phase = "resume"
    assert probe.parent(args) == 0
    assert json.loads((root / "phase17-report.json").read_text())["status"] == "PASS"
    assert json.loads((root / "probe-state.json").read_text())["status"] == "resume_verified"
    with pytest.raises(ValueError, match="completed matching phase-16"):
        probe.parent(args)
    for path in backup.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    backup.chmod(0o755)
