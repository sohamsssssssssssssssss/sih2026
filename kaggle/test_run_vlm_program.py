import hashlib
import json
from pathlib import Path

import pytest

import kaggle.run_vlm_program as runner
from scripts.evaluate_remote_sensing_adapter import parse_args as parse_eval_args
from scripts.train_remote_sensing_adapter import parse_args as parse_train_args
from training.remote_sensing import sha256_file


def program(tmp_path: Path, stage: str = "ladder", *extra: str) -> runner.Program:
    args = runner.parse_args([
        stage, "--work-dir", str(tmp_path / "work"), "--model-dir", str(tmp_path / "model"),
        "--rsvqa-root", str(tmp_path / "rsvqa"), "--manifest", str(tmp_path / "rsvqa.jsonl"), *extra,
    ])
    instance = runner.Program(args)
    instance.work.mkdir(parents=True)
    (instance.work / "environment.json").write_text(json.dumps({"gpu": "Tesla T4", "preflight_passed": True}))
    return instance


def records(instance: runner.Program) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(instance.registry.glob("SQ-*.json"))]


def eval_part(prediction: str = "yes") -> dict:
    summary = {"n": 1, "strict_accuracy": 1.0, "degenerate_binary": False, "per_type": {}}
    row = {"sample_id": "validation-1", "strict": prediction == "yes"}
    return {"summary": summary, "latency_seconds": 1.0, "peak_cuda_memory_bytes": 5, "results": [row]}


def test_committed_artifacts_match_the_pinned_hashes():
    assert sha256_file(runner.DEV_SUBSET) == runner.DEV_SUBSET_SHA256
    pinned = json.loads(runner.MODEL_FILES.read_text())
    assert pinned["revision"] == runner.MODEL_REVISION
    assert {"model-00001-of-00002.safetensors", "config.json"} <= set(pinned["files"])


def test_model_file_verification_reports_every_problem(tmp_path: Path):
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "weights.safetensors").write_bytes(b"abc")
    manifest = tmp_path / "files.json"
    manifest.write_text(json.dumps({"revision": "r", "files": {
        "config.json": {"size": 2, "git_blob_sha1": runner.blob_sha1(tmp_path / "config.json")},
        "weights.safetensors": {"size": 3, "sha256": hashlib.sha256(b"abd").hexdigest()},
        "tokenizer.json": {"size": 1, "sha256": "x"},
    }}))

    result = runner.verify_model_files(tmp_path, manifest)

    assert result["verified"] is False
    assert result["problems"] == ["missing tokenizer.json", "sha256 mismatch weights.safetensors"]


def test_experiment_ids_stay_in_the_vlm_block_and_never_collide(tmp_path: Path):
    for name in ("SQ-20260926-100.json", "SQ-20260926-101.json", "SQ-20260926-201.json"):
        (tmp_path / name).write_text("{}")

    assert runner.next_experiment_id("20260926", [tmp_path]) == "SQ-20260926-102"
    assert runner.next_experiment_id("20260927", [tmp_path]) == "SQ-20260927-100"


def test_plan_is_bounded_and_estimates_are_labelled_assumptions():
    assert len(runner.MATRIX) + 1 <= 6  # plus the reused rank-16 ladder arm
    assert all(hp["lora_alpha"] == 2 * hp["lora_rank"] for hp in runner.MATRIX)
    sessions = runner.session_plan()
    assert [session["session"][:2] for session in sessions] == ["S1", "S2", "S3", "S4", "S5"]
    assert all(low < high for low, high in (s["estimated_hours_low_high"] for s in sessions))


def test_generated_commands_are_accepted_by_the_real_scripts(tmp_path: Path):
    instance = program(tmp_path, "train-eval", "--max-samples", "7", "--resume-from-checkpoint", "/ckpt")
    train = instance.train_command(tmp_path / "out", 7, runner.BASELINE_HP)
    config, dry_run = parse_train_args(train[2:])

    assert (config.max_samples, config.image_size, config.lora_rank, dry_run) == (7, 252, 16, False)
    assert config.model_revision == runner.MODEL_REVISION
    assert str(config.resume_from_checkpoint) == "/ckpt"
    ladder = program(tmp_path / "other", "ladder", "--resume-from-checkpoint", "/ckpt")
    assert "--resume-from-checkpoint" not in ladder.train_command(tmp_path / "o", 7, runner.BASELINE_HP)

    evaluation = instance.eval_command(runner.DEV_SUBSET, runner.DEV_SUBSET_SHA256, tmp_path / "e.json",
                                       adapter=tmp_path / "adapter", skip_base=True)
    parsed = parse_eval_args(evaluation[2:])
    assert parsed.skip_base and parsed.image_size == 252


def test_dry_run_fails_when_an_adapter_is_emitted(tmp_path: Path, monkeypatch):
    instance = program(tmp_path, "dry-run")

    def fake_run(command, log):
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "training-report.json").write_text(json.dumps({"status": "dry_run_passed"}))
        (out / "adapter").mkdir()
        return 0

    monkeypatch.setattr(instance, "run", fake_run)
    instance.stage_dry_run()

    [record] = records(instance)
    assert record["status"] == "FAILED" and record["metrics"]["adapter_emitted"] is True
    assert record["environment"]["gpu"] == "Tesla T4"


def test_training_run_records_loss_eval_and_paired_comparison(tmp_path: Path, monkeypatch):
    instance = program(tmp_path)
    (instance.work / runner.BASE_V0).write_text(json.dumps({"base": eval_part("no")}))

    def fake_run(command, log):
        if "--output-dir" in command:
            out = Path(command[command.index("--output-dir") + 1])
            (out / "adapter").mkdir(parents=True)
            (out / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
            (out / "training-report.json").write_text(json.dumps({
                "trainer_metrics": {"train_loss": 0.4, "train_runtime": 9.0},
                "trainer_log_history": [{"step": 1, "loss": 0.5}, {"step": 1, "train_runtime": 9.0}],
                "peak_cuda_memory_bytes": 7, "adapter_size_bytes": 7,
            }))
        else:
            Path(command[command.index("--out") + 1]).write_text(json.dumps({"adapted": eval_part("yes")}))
        return 0

    monkeypatch.setattr(instance, "run", fake_run)
    instance.train_and_evaluate("ladder N=3", 3, runner.BASELINE_HP)

    [record] = records(instance)
    assert record["status"] == "PASSED"
    assert record["adapter_revision"] == hashlib.sha256(b"weights").hexdigest()
    assert record["metrics"]["loss_history"] == [{"step": 1, "loss": 0.5}]
    assert record["metrics"]["paired_vs_base_v0"]["base_wrong_adapter_right"] == 1
    assert record["hyperparameters"]["max_samples"] == 3


def test_stages_refuse_to_run_after_a_failed_env_preflight(tmp_path: Path):
    instance = program(tmp_path)
    (instance.work / "environment.json").write_text(json.dumps({"preflight_passed": False}))

    with pytest.raises(RuntimeError, match="env stage failed"):
        instance.environment()


def test_final_test_refuses_an_adapter_changed_after_locking(tmp_path: Path):
    instance = program(tmp_path, "final-test")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    (instance.work / "locked-config.json").write_text(json.dumps({
        "adapter_path": str(adapter), "adapter_weights_sha256": hashlib.sha256(b"locked").hexdigest(),
    }))

    with pytest.raises(RuntimeError, match="changed after it was locked"):
        instance.stage_final_test()
