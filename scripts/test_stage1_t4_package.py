"""Loop-7 CPU tests for the T4 evidence package: staging, verification, resume.

The model is never loaded: scripts.run_stage1_smoke.run and the runner's
generation helper are monkeypatched with deterministic doubles, so every
number asserted here comes from the fake, and gate math is checked against
thresholds imported from the runner module (never re-typed).

Fake run() writes a per-step loss series shaped by monkeypatched constants:
FAKE_FINAL / FAKE_FIRST ratios below the pre-registered thresholds make the
overfit stage pass; out-of-range constants flip the gates, and those flips
are asserted.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG = ROOT / "packages" / "stage1-t4-smoke"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
if str(PKG / "src") not in sys.path:
    sys.path.insert(0, str(PKG / "src"))

# Import through the package namespace so run.py's own
# `from scripts.run_stage1_t4 import main` resolves to the SAME module
# instance these tests monkeypatch.
from scripts import run_stage1_smoke, run_stage1_t4  # noqa: E402


FAKE_FIRST = 2.0
# Ratios deliberately chosen to pass each pre-registered threshold.
FAKE_FINAL = FAKE_FIRST * (run_stage1_t4.FINAL_LOSS_RATIO_THRESHOLD / 2)
FAKE_SPIKE = FAKE_FIRST * (run_stage1_t4.EARLY_SPIKE_RATIO_THRESHOLD / 2)
FAKE_SHUFFLE_MEAN_TRUE = 1.0
FAKE_SHUFFLE_MEAN_SHUFFLED = FAKE_SHUFFLE_MEAN_TRUE * (run_stage1_t4.SHUFFLE_RATIO_THRESHOLD + 0.5)
FAKE_DIFF = run_stage1_t4.LOSS_CONTINUITY_THRESHOLD / 4


TOTAL_REFERENCE_STEPS = 50


def fake_losses(steps, first, final, spike=None, start_step=0, ramp_span=None, lr_span=None):
    """Loss values as a function of the GLOBAL step number.

    Mirrors the real run() records: per-step entries tagged with the global
    step number, so a resumed run's list starts at start_step + 1 and its
    values at each global step equal an uninterrupted run's. The optional
    spike lands at global step 2. ``ramp_span``/``lr_span`` default to the
    50-step reference schedule; run-local runs (max_steps) pass their own N.
    """
    ramp = (ramp_span or TOTAL_REFERENCE_STEPS) - 1
    lr_ref = lr_span or TOTAL_REFERENCE_STEPS
    entries = []
    for i in range(steps):
        global_step = start_step + i + 1
        value = first + (final - first) * (global_step - 1) / ramp
        if spike is not None and global_step == 2:
            value = spike
        entries.append({"step": global_step, "loss": value, "seconds": 1.0,
                        "samples_per_second": 8.0, "peak_memory_bytes": 1,
                        "lr": [round(1e-4 * (lr_ref - global_step) / lr_ref, 15)]})
    return entries


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the heavy run() with a deterministic recorder.

    Side effects written like the real run(): a losses JSON next to the
    requested output and a checkpoint file containing the full state dict the
    continuity math needs.
    """
    calls = []
    written_steps = {}

    def _run(config_path, checkpoint, patch_root, output, *, stop_after_step=None,
             max_steps=None, resume_from=None, shuffle_images=False):
        config = yaml.safe_load(Path(config_path).read_text())
        steps = int(max_steps if max_steps is not None else config["steps"])
        end = steps if stop_after_step is None else min(steps, stop_after_step)
        start_step = 0
        if resume_from is not None:
            start_step = written_steps.get(str(resume_from), end // 2)
        if shuffle_images:
            # Control arm: the fake makes shuffled pixels clearly worse.
            losses = fake_losses(end - start_step, FAKE_FIRST * 2, FAKE_FINAL * 2,
                                 start_step=start_step, ramp_span=end, lr_span=end)
        elif max_steps is not None:
            # Run-local schedule (overfit stage sizes its own step count).
            losses = fake_losses(end, FAKE_FIRST, FAKE_FINAL, ramp_span=end, lr_span=end)
        else:
            losses = fake_losses(end - start_step, FAKE_FIRST, FAKE_FINAL,
                                 spike=FAKE_SPIKE if steps > TOTAL_REFERENCE_STEPS else None,
                                 start_step=start_step)
        payload = {
            "kind": "fake", "losses": losses, "passed": True, "step": end,
            "tokens_per_sample": [16] * 16,
            "shuffle_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] if shuffle_images else None,
            "optimizer_state_digest_after_load": "digest-resumed",
            "optimizer_state_digest": "digest-final",
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        ckpt_path = str(output.with_suffix(".pt"))
        output.with_suffix(".pt").write_bytes(b"checkpoint-bytes")
        written_steps[ckpt_path] = end
        calls.append({"config": Path(config_path).name, "end": end,
                      "max_steps": max_steps, "resume_from": resume_from,
                      "shuffle_images": shuffle_images, "output": Path(output).name})
        return payload

    monkeypatch.setattr(run_stage1_t4, "run", _run)
    monkeypatch.setattr(run_stage1_t4, "evaluate_losses", _evaluate)
    monkeypatch.setattr(run_stage1_t4, "TRAIN_CAPTION_COUNT", 160)
    return calls


def _evaluate(config_path, checkpoint, patch_root, output, *, from_checkpoint, steps=1,
              shuffle_images=False):
    """Eval-only control arm: one loss per image, shuffled arm clearly worse."""
    base = 0.6 if not shuffle_images else 0.6 * (run_stage1_t4.SHUFFLE_RATIO_THRESHOLD + 0.5)
    losses = [{"step": i + 1, "loss": base, "seconds": 1.0, "samples_per_second": 1.0,
               "peak_memory_bytes": 1, "lr": None} for i in range(16)]
    payload = {"kind": "fake-eval", "losses": losses, "passed": True,
               "shuffle_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] if shuffle_images else None,
               "from_checkpoint": str(from_checkpoint), "steps": steps}
    Path(output).write_text(json.dumps(payload, indent=2))
    return payload


@pytest.fixture
def fake_generation(monkeypatch):
    monkeypatch.setattr(run_stage1_t4, "_greedy_decode", lambda *args, **kwargs: {
        "reproduced": run_stage1_t4.DECODE_REPRODUCTION_MIN + 2,
        "threshold_min": run_stage1_t4.DECODE_REPRODUCTION_MIN,
        "config_used": "config.yaml", "attempt": 1, "outputs_head": [],
    })


@pytest.fixture
def package_dir(tmp_path):
    """Copy the real package configs/captions/revision into tmp for staging."""
    root = tmp_path / "pkg"
    root.mkdir()
    for name in ("config.yaml", "config-vision-fp32.yaml", "config-last4.yaml",
                 "config-last4-vision-fp32.yaml", "captions.json", "revision.json", "norm_contract.json"):
        (root / name).write_bytes((PKG / name).read_bytes())
    return root


def new_out(tmp_path):
    """Out dir as main() would provide it: already created."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return out_dir


def test_thresholds_match_preregistration():
    assert run_stage1_t4.LOSS_CONTINUITY_THRESHOLD == 0.05
    assert run_stage1_t4.FINAL_LOSS_RATIO_THRESHOLD == 0.2
    assert run_stage1_t4.EARLY_SPIKE_RATIO_THRESHOLD == 2.0
    assert run_stage1_t4.SHUFFLE_RATIO_THRESHOLD == 1.10
    assert run_stage1_t4.DECODE_REPRODUCTION_MIN == 12
    assert run_stage1_t4.OVERFIT_BUDGET_SECONDS == 1200
    assert run_stage1_t4.TRAIN_CAPTION_COUNT == 463932


def test_stage_verify_passes_when_hashes_match(package_dir, tmp_path):
    out_dir = tmp_path / "out"
    result = run_stage1_t4.stage_verify(_checkpoint_with_matching_hashes(package_dir),
                                        package_dir / "revision.json", package_dir, out_dir)
    assert result["passed"] is True
    assert result["revision_matches"] is True
    assert all(entry["matches"] for entry in result["weight_sha256"].values())
    assert json.loads((out_dir / "stage-verify.json").read_text())["passed"]


def _checkpoint_with_matching_hashes(package_dir):
    """Build a checkpoint dir whose weight files hash to the pinned values.

    The real pinned hashes are of the multi-GB safetensors; here the pinned
    revision.json is what defines truth, so the test rewrites it with the
    hashes of the tiny files it creates. Verification logic is identical.
    """
    checkpoint = package_dir / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    first, second = b"weights-one", b"weights-two"
    (checkpoint / "config.json").write_text(json.dumps({"_commit_hash": revision_revision(package_dir)}))
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(first)
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(second)
    revision = json.loads((package_dir / "revision.json").read_text())
    revision["weight_sha256"] = {
        "model-00001-of-00002.safetensors": hashlib.sha256(first).hexdigest(),
        "model-00002-of-00002.safetensors": hashlib.sha256(second).hexdigest(),
    }
    (package_dir / "revision.json").write_text(json.dumps(revision, indent=2))
    return checkpoint


def revision_revision(package_dir):
    return json.loads((package_dir / "revision.json").read_text())["revision"]


def test_stage_verify_aborts_on_weight_mismatch(package_dir, tmp_path):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"tampered")
    result = run_stage1_t4.stage_verify(checkpoint, package_dir / "revision.json",
                                        package_dir, tmp_path / "out")
    assert result["passed"] is False
    assert not result["weight_sha256"]["model-00002-of-00002.safetensors"]["matches"]


def test_stage_verify_aborts_on_revision_mismatch(package_dir, tmp_path):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    (checkpoint / "config.json").write_text(json.dumps({"_commit_hash": "deadbeef"}))
    result = run_stage1_t4.stage_verify(checkpoint, package_dir / "revision.json",
                                        package_dir, tmp_path / "out")
    assert result["passed"] is False
    assert result["revision_matches"] is False
    assert result["passed"] is False
    assert result["revision_matches"] is False


def test_stage_smoke_passes_first_attempt_and_records_rate(package_dir, fake_run, tmp_path):
    out_dir = new_out(tmp_path)
    smoke = run_stage1_t4.stage_smoke(package_dir, "ckpt-unused", "patches", out_dir)
    assert smoke["passed"] is True
    assert smoke["config"] == "config.yaml"
    assert smoke["vision_dtype_used"] == "float16"
    assert smoke["samples_per_second"] == 8.0
    assert fake_run[0]["config"] == "config.yaml"


def test_stage_smoke_falls_back_to_vision_fp32(package_dir, monkeypatch, tmp_path):
    calls = []

    def flaky_run(config_path, checkpoint, patch_root, output, **kwargs):
        name = Path(config_path).name
        calls.append(name)
        if name == "config.yaml":
            raise FloatingPointError("Vision tower produced inf/NaN; use vision-fp32 variant")
        payload = {"kind": "fake", "losses": fake_losses(50, FAKE_FIRST, FAKE_FINAL),
                   "passed": True, "tokens_per_sample": [16] * 16,
                   "optimizer_state_digest": "digest-final"}
        Path(output).write_text(json.dumps(payload))
        Path(output).with_suffix(".pt").write_bytes(b"ckpt")
        return payload

    monkeypatch.setattr(run_stage1_t4, "run", flaky_run)
    out_dir = new_out(tmp_path)
    smoke = run_stage1_t4.stage_smoke(package_dir, "ckpt", "patches", out_dir)
    assert calls == ["config.yaml", "config-vision-fp32.yaml"]
    assert smoke["passed"] is True
    assert smoke["config"] == "config-vision-fp32.yaml"
    assert smoke["vision_dtype_used"] == "float32"


def test_stage_smoke_both_attempts_fail(package_dir, monkeypatch, tmp_path):
    def always_fails(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(run_stage1_t4, "run", always_fails)
    out_dir = new_out(tmp_path)
    smoke = run_stage1_t4.stage_smoke(package_dir, "ckpt", "patches", out_dir)
    assert smoke["passed"] is False
    assert len(smoke["attempts"]) == 2


def test_stage_resume_stops_and_resumes_with_continuity(package_dir, fake_run, tmp_path):
    calls = fake_run
    out_dir = new_out(tmp_path)
    result = run_stage1_t4.stage_resume(package_dir, "config.yaml", "ckpt", "patches", out_dir)
    assert result["passed"] is True
    assert result["loss_continuity"]["passed"] is True
    assert result["lr_schedule_continuity"]["passed"] is True
    names = [c["output"] for c in calls]
    assert names == ["stage-resume-interrupted.json", "stage-resume-resumed.json",
                     "stage-resume-uninterrupted.json"]
    assert calls[0]["end"] == 25
    assert calls[1]["resume_from"] is not None
    assert calls[2]["end"] == 50 and calls[2]["resume_from"] is None
    # The fake's losses are an exact function of the global step, so a correct
    # resume reproduces them exactly; anything above zero would be a bug.
    assert result["loss_continuity"]["max_abs_loss_diff"] <= run_stage1_t4.LOSS_CONTINUITY_THRESHOLD
    assert result["loss_continuity"]["max_abs_loss_diff"] == pytest.approx(0.0, abs=1e-12)


def test_stage_resume_fails_beyond_threshold(package_dir, monkeypatch, tmp_path):
    calls = []

    def diverging_run(config_path, checkpoint, patch_root, output, *, stop_after_step=None,
                      max_steps=None, resume_from=None, shuffle_images=False):
        config = yaml.safe_load(Path(config_path).read_text())
        steps = int(max_steps if max_steps is not None else config["steps"])
        end = steps if stop_after_step is None else min(steps, stop_after_step)
        if resume_from is not None:
            losses = fake_losses(end - end // 2, FAKE_FIRST, FAKE_FIRST * 4, start_step=end // 2)  # diverges hard
        else:
            losses = fake_losses(end, FAKE_FIRST, FAKE_FINAL)
        payload = {"kind": "fake", "losses": losses, "passed": True, "step": end,
                   "tokens_per_sample": [16] * 16,
                   "optimizer_state_digest_after_load": "digest-resumed",
                   "optimizer_state_digest": "digest-final"}
        Path(output).write_text(json.dumps(payload))
        Path(output).with_suffix(".pt").write_bytes(b"ckpt")
        calls.append(Path(output).name)
        return payload

    monkeypatch.setattr(run_stage1_t4, "run", diverging_run)
    out_dir = new_out(tmp_path)
    result = run_stage1_t4.stage_resume(package_dir, "config.yaml", "ckpt", "patches", out_dir)
    assert result["passed"] is False
    assert result["loss_continuity"]["max_abs_loss_diff"] > run_stage1_t4.LOSS_CONTINUITY_THRESHOLD
    assert len(calls) == 6  # two full attempts of three runs each


def test_stage_overfit_gates_pass_and_steps_sized_from_smoke(package_dir, fake_run, fake_generation, tmp_path):
    out_dir = new_out(tmp_path)
    smoke = run_stage1_t4.stage_smoke(package_dir, "ckpt", "patches", out_dir)
    result = run_stage1_t4.stage_overfit(package_dir, "config.yaml", "ckpt", "patches",
                                         out_dir, smoke)
    assert result["passed"] is True
    gates = result["gates"]
    assert gates["final_loss_ratio"]["measured"] == pytest.approx(FAKE_FINAL / FAKE_FIRST)
    assert gates["no_early_spike"]["passed"] is True
    assert gates["greedy_decode_reproduction"]["passed"] is True
    assert gates["image_shuffle_control"]["passed"] is True
    assert gates["image_shuffle_control"]["measured_ratio"] == pytest.approx(
        run_stage1_t4.SHUFFLE_RATIO_THRESHOLD + 0.5)
    # Sizing: smoke losses are 1s/step -> 1200 // 1 = 1200 steps requested.
    train_call = next(c for c in fake_run if c["output"] == "stage-overfit-train.json")
    assert train_call["max_steps"] == run_stage1_t4.OVERFIT_BUDGET_SECONDS // 1


def test_stage_overfit_shuffle_gate_fails_when_ratio_low(package_dir, fake_run, fake_generation, monkeypatch, tmp_path):
    out_dir = new_out(tmp_path)
    smoke = run_stage1_t4.stage_smoke(package_dir, "ckpt", "patches", out_dir)

    def flat_evaluate(config_path, checkpoint, patch_root, output, *, from_checkpoint,
                      steps=1, shuffle_images=False):
        base = 0.5 if not shuffle_images else 0.5 * (run_stage1_t4.SHUFFLE_RATIO_THRESHOLD - 0.01)
        losses = [{"step": i + 1, "loss": base, "seconds": 1.0, "samples_per_second": 1.0,
                   "peak_memory_bytes": 1, "lr": None} for i in range(16)]
        payload = {"kind": "fake-eval", "losses": losses, "passed": True}
        Path(output).write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(run_stage1_t4, "evaluate_losses", flat_evaluate)
    result = run_stage1_t4.stage_overfit(package_dir, "config.yaml", "ckpt", "patches", out_dir, smoke)
    assert result["passed"] is False
    assert result["gates"]["image_shuffle_control"]["passed"] is False


def test_stage_extrapolate_uses_measured_rate(package_dir, tmp_path):
    out_dir = new_out(tmp_path)
    smoke = {"samples_per_second": 0.8, "attempt": 1}
    result = run_stage1_t4.stage_extrapolate(package_dir, "config.yaml", out_dir, smoke)
    assert result["epoch_hours_1x_t4"] == pytest.approx(463932 / 0.8 / 3600)
    assert result["epoch_hours_2x_t4_ddp"] is None
    assert "extrapolation" in result["label"]
    assert json.loads((out_dir / "stage-extrapolate.json").read_text()) == result


def test_happy_path_runs_stages_in_registered_order(package_dir, fake_run, fake_generation, tmp_path):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    out_dir = new_out(tmp_path)
    calls = fake_run
    verify = run_stage1_t4.stage_verify(checkpoint, package_dir / "revision.json", package_dir, out_dir)
    smoke = run_stage1_t4.stage_smoke(package_dir, checkpoint, "patches", out_dir)
    resume = run_stage1_t4.stage_resume(package_dir, smoke["config"], checkpoint, "patches", out_dir)
    overfit = run_stage1_t4.stage_overfit(package_dir, smoke["config"], checkpoint, "patches", out_dir, smoke)
    extrapolate = run_stage1_t4.stage_extrapolate(package_dir, smoke["config"], out_dir, smoke)
    assert [verify, smoke, resume, overfit, extrapolate] == sorted(
        [verify, smoke, resume, overfit, extrapolate], key=lambda stage: not stage["passed"])
    assert all(stage["passed"] for stage in (verify, smoke, resume, overfit, extrapolate))
    kinds = [call["output"].replace("stage-", "").replace(".json", "") for call in calls]
    resume_positions = [i for i, kind in enumerate(kinds) if kind.startswith("resume")]
    overfit_positions = [i for i, kind in enumerate(kinds) if kind.startswith("overfit")]
    assert kinds[0] == "smoke-attempt1"
    assert resume_positions and overfit_positions
    assert max(resume_positions) < min(overfit_positions)
    assert kinds[-1].startswith("overfit")
    # Every stage JSON carries the preregistered provenance keys.
    for stage in (smoke, resume, overfit, extrapolate):
        for key in ("git_sha", "revision", "split_name", "n", "seed", "device", "dtype", "timestamp_utc"):
            assert key in stage, f"missing provenance key {key!r} in stage {stage['kind']}"


def test_main_stops_at_first_failed_stage(package_dir, fake_run, fake_generation, monkeypatch, tmp_path, capsys):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    monkeypatch.setattr(run_stage1_t4, "stage_resume", lambda *a, **k: {"passed": False, "status": "x"})
    argv = [str(PKG / "run.py"), "all", "--checkpoint", str(checkpoint),
            "--patch-root", "patches", "--out-dir", str(tmp_path / "out"),
            "--package-root", str(package_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    from run import main as run_main
    with pytest.raises(SystemExit) as excinfo:
        run_main()
    assert excinfo.value.code == 1
    summary = json.loads(((tmp_path / "out") / "session-summary.json").read_text())
    assert summary["stages_attempted"] == ["verify", "smoke", "resume"]
    assert summary["all_passed"] is False


def test_package_manifest_integrity():
    manifest = json.loads((PKG / "package-manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = PKG / name
        assert path.is_file(), f"missing package file: {name}"
        assert path.stat().st_size == entry["bytes"], f"size mismatch: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"hash mismatch: {name}"


def test_real_package_smoke_script_exposes_resume_kwargs():
    """The real run() must accept the loop-7 control arguments."""
    import inspect
    signature = inspect.signature(run_stage1_smoke.run)
    assert {"stop_after_step", "max_steps", "resume_from", "shuffle_images"} <= set(signature.parameters)


def test_memory_matrix_records_control_oom_and_variant(package_dir, fake_run, tmp_path, monkeypatch):
    for name in ("config-last4.yaml", "config-last4-vision-fp32.yaml"):
        (package_dir / name).write_bytes((PKG / name).read_bytes())
    original = run_stage1_t4.run
    calls = []
    def run(config, checkpoint, patches, output, **kwargs):
        calls.append(config.name)
        if "last4" not in config.name:
            raise RuntimeError("CUDA out of memory (synthetic)")
        return original(config, checkpoint, patches, output, **kwargs)
    monkeypatch.setattr(run_stage1_t4, "run", run)
    result = run_stage1_t4.stage_memory_matrix(package_dir, tmp_path, tmp_path, tmp_path / "matrix")
    assert calls == ["config.yaml", "config-vision-fp32.yaml", "config-last4.yaml"]
    assert not result["variants"]["control"]["passed"]
    assert result["variants"]["control"]["oom"]
    assert result["variants"]["last4"]["passed"]
    assert result["variants"]["last4"]["peak_memory_bytes"] == 1
    assert (tmp_path / "matrix/control/stage-smoke-attempt1-failure.json").exists()
    assert (tmp_path / "matrix/control/stage-smoke-attempt2-failure.json").exists()


def test_explicit_last4_fp32_reaches_resume_and_overfit(package_dir, fake_run, fake_generation, tmp_path, monkeypatch):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    out = tmp_path / "variant"
    monkeypatch.setattr(sys, "argv", ["runner", "--checkpoint", str(checkpoint),
        "--patch-root", "patches", "--package-root", str(package_dir), "--out-dir", str(out),
        "--config", "config-last4-vision-fp32.yaml"])
    run_stage1_t4.main()
    assert {call["config"] for call in fake_run} == {"config-last4-vision-fp32.yaml"}
    assert json.loads((out / "session-summary.json").read_text())["all_passed"]
    record = json.loads((out / "stage-resume.json").read_text())
    assert record["vision_fp32"] and record["vision_trainable_last_n_blocks"] == 4
    assert record["loss_continuity"]["steps_compared"] == list(range(26, 51))


def test_resume_only_runs_exact_variant_and_keeps_previous_evidence(package_dir, fake_run, tmp_path, monkeypatch):
    checkpoint = _checkpoint_with_matching_hashes(package_dir)
    prior = tmp_path / "failed-control"
    prior.mkdir()
    evidence = prior / "stage-resume-attempt1-failure.json"
    evidence.write_text('{"passed": false, "oom": true}')
    before = evidence.read_bytes()
    out = tmp_path / "resume-last4"
    argv = ["runner", "--checkpoint", str(checkpoint), "--patch-root", "patches",
            "--package-root", str(package_dir), "--out-dir", str(out), "--mode", "resume",
            "--config", "config-last4-vision-fp32.yaml"]
    monkeypatch.setattr(sys, "argv", argv)
    run_stage1_t4.main()
    assert [call["end"] for call in fake_run] == [25, 50, 50]
    assert {call["config"] for call in fake_run} == {"config-last4-vision-fp32.yaml"}
    assert evidence.read_bytes() == before
    assert json.loads((out / "session-summary.json").read_text())["stages_attempted"] == ["verify", "resume"]
    with pytest.raises(FileExistsError, match="fresh output"):
        run_stage1_t4.main()


def test_resume_execution_failure_is_preserved(package_dir, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic CUDA out of memory")
    monkeypatch.setattr(run_stage1_t4, "run", fail)
    out = tmp_path / "failed"
    result = run_stage1_t4.stage_resume(package_dir, "config-last4-vision-fp32.yaml", "weights", "patches", out)
    assert not result["passed"]
    assert (out / "stage-resume-attempt1-failure.json").is_file()


def test_updated_package_runner_matches_source():
    for name in ("run_stage1_smoke.py", "run_stage1_t4.py"):
        assert (PKG / "src/scripts" / name).read_bytes() == (ROOT / "scripts" / name).read_bytes()
