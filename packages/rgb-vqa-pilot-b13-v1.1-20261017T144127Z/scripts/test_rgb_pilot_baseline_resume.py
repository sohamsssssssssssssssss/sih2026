"""Synthetic CPU tests for durable baseline progress and safe resume."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import rgb_pilot_package as package


class FakeProcessor:
    chat_template = "synthetic chat template"
    image_processor = SimpleNamespace(to_dict=lambda: {"synthetic": True})


class FakeModel:
    generation_config = SimpleNamespace(to_dict=lambda: {"do_sample": False})

    def eval(self):
        return self

    def parameters(self):
        return iter(())


def install_model_doubles(monkeypatch):
    processor = FakeProcessor()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **k: processor)
    transformers.Qwen2_5_VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeModel()
    )
    vision = ModuleType("qwen_vl_utils")
    vision.process_vision_info = object()
    evaluator = ModuleType("scripts.evaluate_remote_sensing_adapter")
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision)
    monkeypatch.setitem(sys.modules, "scripts.evaluate_remote_sensing_adapter", evaluator)
    return processor, evaluator


def fixtures():
    rows = [
        {"sample_id": "synthetic-1", "question_id": 1, "image_id": 10, "image": "10.tif",
         "question_type": "presence", "question": "Synthetic presence?", "answer": "yes"},
        {"sample_id": "synthetic-2", "question_id": 2, "image_id": 20, "image": "20.tif",
         "question_type": "count", "question": "Synthetic count?", "answer": "2"},
    ]
    examples = [SimpleNamespace(sample_id=row["sample_id"]) for row in rows]
    return rows, examples


def setup_baseline(monkeypatch, tmp_path):
    rows, examples = fixtures()
    monkeypatch.setattr(package, "validate_pilot", lambda *a: (rows, examples, {
        "manifest_identity": "synthetic-manifest", "selected_image_sha256": {"synthetic.tif": "synthetic-hash"}
    }))
    monkeypatch.setattr(package, "binding", lambda root: "synthetic-package-identity")
    processor, evaluator = install_model_doubles(monkeypatch)
    args = SimpleNamespace(output=tmp_path, weights=tmp_path / "synthetic-weights", resume=False)
    return rows, processor, evaluator, args


def test_interruption_resumes_interleaved_jsonl_without_duplicates(tmp_path, monkeypatch):
    rows, _, evaluator, args = setup_baseline(monkeypatch, tmp_path)
    calls = []

    def interrupt_after_three(model, processor, vision, example, size, *, blind=False):
        calls.append((example.sample_id, blind))
        if len(calls) == 4:
            raise RuntimeError("synthetic interruption")
        return rows[0 if example.sample_id == "synthetic-1" else 1]["answer"]

    evaluator._answer = interrupt_after_three
    fsyncs = []
    real_fsync = package.os.fsync
    monkeypatch.setattr(package.os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1])
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        package.baseline(package.ROOT, args)
    progress_path = tmp_path / "baseline-predictions.jsonl"
    partial = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [(r["sample_id"], r["condition"]) for r in partial] == [
        ("synthetic-1", "base_real"), ("synthetic-1", "base_constant"), ("synthetic-2", "base_real")
    ]
    assert len(fsyncs) >= len(partial)
    assert not (tmp_path / "result.json").exists()

    args.resume = True
    evaluator._answer = lambda model, processor, vision, example, size, *, blind=False: (
        rows[0 if example.sample_id == "synthetic-1" else 1]["answer"]
    )
    report = package.baseline(package.ROOT, args)
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert len(records) == 4
    assert [(r["sample_id"], r["condition"]) for r in records] == [
        ("synthetic-1", "base_real"), ("synthetic-1", "base_constant"),
        ("synthetic-2", "base_real"), ("synthetic-2", "base_constant")
    ]
    assert report["completed_predictions"] == report["expected_predictions"] == 4
    assert report["conditions"]["base_real"]["summary"]["n"] == 2
    package._write_durable_json(tmp_path / "result.json", report)
    with pytest.raises(ValueError, match="already complete"):
        package.baseline(package.ROOT, args)


def test_resume_rejects_changed_run_configuration(tmp_path, monkeypatch):
    _, processor, evaluator, args = setup_baseline(monkeypatch, tmp_path)
    evaluator._answer = lambda *a, **k: "yes"
    package.baseline(package.ROOT, args)
    args.resume = True
    processor.chat_template = "changed synthetic template"
    with pytest.raises(ValueError, match="configuration mismatch"):
        package.baseline(package.ROOT, args)


def test_resume_discards_only_unterminated_tail(tmp_path):
    rows, _ = fixtures()
    path = tmp_path / "baseline-predictions.jsonl"
    good = {"sample_id": "synthetic-1", "question_id": 1, "image_id": 10, "image": "10.tif",
        "question_type": "presence", "question": "Synthetic presence?", "condition": "base_real",
        "expected": "yes", "prediction": "yes", "correct": True, "malformed": False}
    path.write_bytes((json.dumps(good) + "\n{partial").encode())
    assert package._read_baseline_progress(path, rows) == [good]
    assert path.read_text() == json.dumps(good) + "\n"


def test_cli_resume_reuses_output_and_charges_same_budget_ledger(tmp_path, monkeypatch):
    output = tmp_path / "baseline"
    ledger = tmp_path / "budget.json"
    monkeypatch.setattr(package, "preflight", lambda root: {})
    monkeypatch.setattr(package, "binding", lambda root: "synthetic-package-identity")
    monkeypatch.setattr(package, "supervise", lambda args: {
        "exit_code": 124, "status": "synthetic interrupted worker"
    })
    common = ["pilot.py", "baseline", "--weights", str(tmp_path / "weights"),
        "--output", str(output), "--budget-ledger", str(ledger), "--max-seconds", "1200"]
    monkeypatch.setattr(sys, "argv", common)
    with pytest.raises(SystemExit) as first:
        package.main()
    assert first.value.code == 124
    (output / "baseline-run.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", common + ["--resume"])
    with pytest.raises(SystemExit) as second:
        package.main()
    assert second.value.code == 124
    attempts = json.loads(ledger.read_text())
    assert len(attempts) == 2
    assert {entry["output"] for entry in attempts} == {str(output.resolve())}
    assert (output / "session.json").is_file()
