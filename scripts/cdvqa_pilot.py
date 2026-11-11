"""CDVQA two-image pilot stages: preflight, token-preflight, train, evaluate.

Mirrors the B13 runner's safety behaviour: shared budget ledger with a lock, --max-seconds per
invocation, checkpoints every 16 optimizer steps and at the final step, resume only from a verified
incomplete checkpoint of the same package/config/manifest identity, training-report.json statuses,
exactly one visible CUDA GPU. Stages that need weights or CUDA are not exercised by CPU tests.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cdvqa_pairs import (  # noqa: E402
    SCORING_VERSION, assert_no_pair_leak,
    digest, evaluate_conditions, load_choices, load_pair_manifest, load_vocabulary,
    majority_accuracy, paired_differences, summarise)
from training.remote_sensing import (  # noqa: E402
    ANSWER_INSTRUCTION, BASE_MODEL_ID, PAIR_ORDER_INSTRUCTION, QwenVQACollator, TrainingConfig,
    load_training_components, messages, sha256_file)

MANIFESTS = "data/manifests/cdvqa"
TRAIN_MANIFEST = f"{MANIFESTS}/train-8k.v2.jsonl"
PROVENANCE = f"{MANIFESTS}/provenance.json"
EVAL_MANIFESTS = {"test1": f"{MANIFESTS}/eval-test1-400.v1.jsonl", "test2": f"{MANIFESTS}/eval-test2-400.v1.jsonl"}
TRAIN_STEPS = 1000
SAVE_EVERY = 16
IMAGE_SIZE = 392
SEED = 17
BOOTSTRAP = 2000
LIMITS = {"train": 3600, "evaluate": 2400}
LEDGER_TOTAL_DEFAULT = 7200  # B13 inherited shared-ledger cap, used when pilot-config has no budget block
REPORT_NAME = "training-report.json"
RESULT_NAME = "result.json"
REQUIRED_CHECKPOINT_FILES = frozenset(("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth",
                                       "adapter_config.json", "adapter_model.safetensors"))


def load_plan(root):
    return json.loads((root / "pilot-config.json").read_text())


def ledger_total(plan, stage, override=None):
    """Shared-ledger cap in seconds: CLI override, else pilot-config budget, else the B13 default 7200."""
    if override is not None:
        return override
    totals = (plan.get("budget") or {}).get("ledger_total_seconds") or {}
    return int(totals.get(stage, LEDGER_TOTAL_DEFAULT))


def check_reservation(entries, seconds, total):
    """Refuse a new reservation that would push the shared ledger over its cap."""
    spent = sum(e["charged_seconds"] for e in entries)
    if spent + seconds > total:
        raise ValueError(f"Shared budget exhausted: {spent} s charged + {seconds} s requested > {total} s cap; "
                         "new approval required")


def pair_examples(manifest, image_root, root):
    return load_pair_manifest(Path(manifest), Path(image_root), root / PROVENANCE)[0]


def train_config(root, weights, output, image_root, total_steps=TRAIN_STEPS):
    plan = dict(load_plan(root)["training"])
    plan["lora_target_modules"] = tuple(plan["lora_target_modules"])
    plan["max_steps"] = total_steps
    return TrainingConfig(model_path=Path(weights), dataset_manifest=root / TRAIN_MANIFEST,
                          image_root=Path(image_root), output_dir=Path(output), **plan)


def binding(root):
    """Identity every checkpoint and report is tied to: package payload, train manifest, config."""
    package = json.loads((root / "package-manifest.json").read_text())
    return digest({"payload_sha256": package["payload_sha256"],
                   "train_manifest_sha256": sha256_file(root / TRAIN_MANIFEST),
                   "provenance_sha256": sha256_file(root / PROVENANCE),
                   "config_sha256": sha256_file(root / "pilot-config.json")})


def check_payload(root):
    package = json.loads((root / "package-manifest.json").read_text())
    for name, expected in package["payload_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Package payload mismatch: {name}")
    provenance = json.loads((root / PROVENANCE).read_text())
    for name, expected in provenance["manifest_sha256"].items():
        if sha256_file(root / MANIFESTS / name) != expected:
            raise ValueError(f"Manifest hash mismatch: {name}")
    return package


def verify_base(root, weights):
    from scripts.stage1_runtime import verify_checkpoint
    spec = json.loads((root / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    proof = verify_checkpoint(weights, spec["revision"], spec["weight_sha256"])
    if not proof["passed"]:
        raise ValueError("Pinned base revision/weights missing or mismatched")
    return proof


def trainable_guard(model):
    """Only rank-16 language attention LoRA A/B matrices may train."""
    expected = {"q_proj", "k_proj", "v_proj", "o_proj"}
    found, names = set(), []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        target = next((part for part in parts if part in expected), None)
        if ("visual" in parts or "layers" not in parts or target is None
                or not any(part in ("lora_A", "lora_B") for part in parts)
                or parameter.ndim != 2
                or (parameter.shape[0] if "lora_A" in parts else parameter.shape[1]) != 16):
            raise ValueError(f"Unexpected trainable parameter: {name}")
        found.add(target)
        names.append(name)
    if found != expected:
        raise ValueError("Missing intended language attention LoRA parameters")
    return names


def rgb_projection_ok(model):
    """The original Qwen patch projection must take three channels (RGB pilot requirement)."""
    projections = [m for n, m in model.named_modules() if n.endswith("patch_embed.proj")]
    return bool(projections) and all(getattr(m, "in_channels", None) == 3 for m in projections)


def verify_resume(checkpoint, identity, total_steps):
    evidence = json.loads((checkpoint / "pilot-checkpoint.json").read_text())
    if evidence.get("run_identity") != identity:
        raise ValueError("Resume package/config/manifest identity mismatch")
    hashes = evidence.get("files_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Resume checkpoint hash inventory missing")
    files = {}
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Resume checkpoint symlink rejected")
        if path.is_file() and path.name != "pilot-checkpoint.json":
            files[str(path.relative_to(checkpoint))] = sha256_file(path)
    if set(files) != set(hashes):
        raise ValueError("Resume checkpoint file inventory differs")
    missing = REQUIRED_CHECKPOINT_FILES - set(hashes)
    if missing:
        raise ValueError(f"Resume state missing: {sorted(missing)}")
    scaler_expected = evidence.get("scaler_expected")
    if type(scaler_expected) is not bool or ("scaler.pt" in hashes) != scaler_expected:
        raise ValueError("Resume scaler state is missing or inconsistent")
    for name, value in hashes.items():
        if files[name] != value:
            raise ValueError(f"Resume checkpoint hash mismatch: {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state.get("max_steps") != total_steps or not 0 < int(state["global_step"]) < total_steps:
        raise ValueError("Resume must keep the original schedule and an incomplete checkpoint")
    if evidence.get("step", state["global_step"]) != state["global_step"]:
        raise ValueError("Resume checkpoint step differs from Trainer state")
    return evidence


def train_pairs(root, weights, output, image_root, *, max_seconds=None, began=None, resume_from=None,
                total_steps=TRAIN_STEPS, save_every=SAVE_EVERY, expected_examples=8000,
                extra_callbacks=(), component_loader=None, collator_factory=None):
    """Two-image QLoRA training on the frozen v2 manifest, with B13-style checkpoints and statuses."""
    from scripts import train_remote_sensing_adapter as runner
    from transformers import Trainer, TrainerCallback
    began = time.monotonic() if began is None else began
    identity = binding(root)
    resume_evidence = verify_resume(Path(resume_from), identity, total_steps) if resume_from else None
    accumulation = load_plan(root)["training"]["gradient_accumulation_steps"]
    trainer_ref = {}
    deadline = (max_seconds - 120) if max_seconds is not None else None

    class Budget(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step == total_steps:
                control.should_save = True
            if deadline is not None and time.monotonic() - began >= deadline:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args_, state, control, **kwargs):
            checkpoint = Path(args_.output_dir) / f"checkpoint-{state.global_step}"
            hashes = {str(p.relative_to(checkpoint)): sha256_file(p)
                      for p in checkpoint.rglob("*") if p.is_file() and p.name != "pilot-checkpoint.json"}
            scaler_expected = getattr(trainer_ref["trainer"].accelerator, "scaler", None) is not None
            if ("scaler.pt" in hashes) != scaler_expected:
                raise ValueError("Trainer scaler save behaviour differs from runtime")
            (checkpoint / "pilot-checkpoint.json").write_text(json.dumps(
                {"run_identity": identity, "step": state.global_step, "scaler_expected": scaler_expected,
                 "files_sha256": hashes}, indent=2))

    class PairTrainer(Trainer):
        def __init__(self, **kwargs):
            examples = kwargs["train_dataset"]
            if len(examples) != expected_examples or len({e.sample_id for e in examples}) != expected_examples:
                raise ValueError("Pair training selection has the wrong size or duplicate ids")
            arguments = kwargs["args"]
            if (arguments.max_steps, arguments.per_device_train_batch_size,
                    arguments.gradient_accumulation_steps) != (total_steps, 1, accumulation):
                raise ValueError("Trainer differs from the frozen schedule")
            arguments.save_steps = save_every
            arguments.save_total_limit = None
            super().__init__(**kwargs, callbacks=[*extra_callbacks, Budget()])
            trainer_ref["trainer"] = self

        def train(self, *a, **kw):
            if deadline is not None and time.monotonic() - began >= deadline:
                raise TimeoutError("No remaining training budget after setup")
            if resume_evidence is not None:
                if resume_evidence.get("scaler_expected") != (getattr(self.accelerator, "scaler", None) is not None):
                    raise ValueError("Resume runtime scaler differs from saved checkpoint")
            return super().train(*a, resume_from_checkpoint=str(resume_from) if resume_from else None, **kw)

    def components(config):
        model, processor = (component_loader or load_training_components)(config)
        if not rgb_projection_ok(model):
            raise ValueError("Original RGB projection required")
        names = trainable_guard(model)
        Path(output, "trainable-parameters.json").write_text(json.dumps(names, indent=2))
        return model, processor

    config = train_config(root, weights, output, image_root, total_steps)
    report = runner.run(config, component_loader=components, trainer_factory=PairTrainer,
                        collator_factory=collator_factory,
                        example_loader=lambda c: pair_examples(c.dataset_manifest, c.image_root, root))
    output = Path(output)
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    step = json.loads((checkpoints[-1] / "trainer_state.json").read_text())["global_step"] if checkpoints else 0
    report.update(status="training_complete" if step == total_steps else "budget_stopped_incomplete",
                  global_step=step, run_identity=identity, training_manifest_identity=digest(
                      {"manifest_sha256": sha256_file(root / TRAIN_MANIFEST)}),
                  base_revision=load_plan(root)["base_revision"])
    (output / REPORT_NAME).write_text(json.dumps(report, indent=2))
    return report


def preflight(root, image_root=None):
    """CPU-only: package/manifest hashes, leakage, vocabulary, and (with images) pair resolution."""
    check_payload(root)
    assert_no_pair_leak(root / TRAIN_MANIFEST, root / EVAL_MANIFESTS["test1"])
    assert_no_pair_leak(root / TRAIN_MANIFEST, root / EVAL_MANIFESTS["test2"])
    report = {"payload": "PASS", "manifests": "PASS", "leakage": "PASS", "run_identity": binding(root)}
    if image_root is not None:
        examples = pair_examples(root / TRAIN_MANIFEST, image_root, root)
        report["train_pairs_resolved"] = len(examples)
    return report


def token_check(plan, examples, processor, process_vision_info=None):
    """Every example must encode with an answer-only label mask and fit the sequence limit.

    `plan` is the shipped training block; it must carry image_size and max_sequence_length.
    """
    collator = QwenVQACollator(processor, plan["image_size"], process_vision_info=process_vision_info,
                               max_sequence_length=plan["max_sequence_length"])
    failures = []
    for example in examples:
        try:
            batch = collator([example])
            if "pixel_values" not in batch or not batch["pixel_values"].numel():
                raise ValueError("image-token path missing")
            if batch["input_ids"].shape[1] > plan["max_sequence_length"]:
                raise ValueError("exceeds sequence limit")
        except Exception as exc:  # collect every failure, do not stop at the first
            failures.append({"sample_id": example.sample_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"examples_checked": len(examples), "max_sequence_length": plan["max_sequence_length"],
            "all_fit": not failures, "failures": failures[:50]}


def token_preflight(root, weights, image_root):
    """Real processor, CPU: checks every training example with the pinned base processor."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    check_payload(root)
    proof = verify_base(root, Path(weights))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    examples = pair_examples(root / TRAIN_MANIFEST, image_root, root)
    report = token_check(load_plan(root)["training"], examples, processor)
    report["base_revision"] = proof["measured_revision"]
    report["revision_source"] = proof["revision_source"]
    return report


def evaluate_gpu(root, weights, adapter, image_root, eval_name, output, *, bootstrap=BOOTSTRAP):
    """Base-only (no adapter) or base + adapter under the four paired pair conditions."""
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    manifest = root / EVAL_MANIFESTS[eval_name]
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = pair_examples(manifest, image_root, root)
    if len(rows) != len(examples):
        raise ValueError("Row/example alignment mismatch")
    provenance = json.loads((root / PROVENANCE).read_text())
    vocab = load_vocabulary(root / PROVENANCE)
    spec = json.loads((root / "configs/model_artifacts.json").read_text())["providers"]["qwen2.5vl-3b"]
    proof = verify_base(root, Path(weights))
    processor = AutoProcessor.from_pretrained(str(weights), local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(weights), local_files_only=True, torch_dtype=torch.float16, device_map="auto").eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    def answerer(model):
        return lambda ex: answer_pair(model, processor, process_vision_info, ex, IMAGE_SIZE)

    conditions = {"base_real": evaluate_conditions(rows, examples, answerer(base), vocab, {"base_real": "real"})["base_real"]}
    if adapter is not None:
        from peft import PeftModel
        adapted = PeftModel.from_pretrained(base, str(adapter), local_files_only=True).eval()
        for name, kind in (("adapter_real", "real"), ("adapter_identical", "identical_pair"),
                           ("adapter_swapped", "swapped_order")):
            conditions[name] = evaluate_conditions(rows, examples, answerer(adapted), vocab,
                                                   {name: kind})[name]
    result = build_pair_result(
        mode="adapter" if adapter is not None else "base_only", eval_name=eval_name,
        eval_manifest_sha256=sha256_file(manifest), rows=rows, conditions=conditions,
        provenance=provenance, base_meta={"base_model_id": BASE_MODEL_ID, "base_revision": spec["revision"],
                                          "base_verification": proof,
                                          "revision_proof": revision_proof_note(proof)},
        adapter_meta=None if adapter is None else {n: sha256_file(Path(adapter) / n)
                                                   for n in ("adapter_config.json", "adapter_model.safetensors")},
        run_identity=binding(root), bootstrap=bootstrap)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / RESULT_NAME).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def revision_proof_note(proof):
    """How the pinned revision was proven. Kaggle: weight hashes match + restored HF metadata."""
    source = proof.get("revision_source")
    if source == "config.json":
        return "config.json _commit_hash + weight sha256 match"
    if source == "hf_download_metadata":
        return "restored .cache/huggingface/download/config.json.metadata (pinned commit) + weight sha256 match"
    return "no revision proof found"


def build_pair_result(*, mode, eval_name, eval_manifest_sha256, rows, conditions, provenance, base_meta,
                      adapter_meta, run_identity, bootstrap=BOOTSTRAP):
    """B13-shaped result.json for pair conditions. Pure: no model, no GPU."""
    summaries = {name: summarise(records) for name, records in conditions.items()}
    diffs = paired_differences(conditions, seed=SEED, repetitions=bootstrap) if len(conditions) > 1 else None
    return {
        "schema_version": 1,
        "mode": mode,
        "scoring": {"version": SCORING_VERSION,
                    "policy": "exact match after strip+casefold against the question type's Train answer vocabulary; anything else is malformed and counted"},
        "eval_manifest": eval_name,
        "eval_manifest_sha256": eval_manifest_sha256,
        **base_meta,
        "adapter_identity": adapter_meta,
        "preprocessing": {"image_size": IMAGE_SIZE, "order": "before image, after image, question"},
        "prompt": {"order_instruction": PAIR_ORDER_INSTRUCTION, "answer_instruction": ANSWER_INSTRUCTION,
                   "allowed_answers": {t: list(v) for t, v in load_choices_from(provenance).items()}},
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "conditions": {name: {"summary": summaries[name], "records": records} for name, records in conditions.items()},
        "paired_differences": diffs,
        "majority_baseline": {"majority_answer": provenance["majority_answer"],
                              **majority_accuracy(rows, provenance["majority_answer"])},
        "run_identity": run_identity,
        "claim": "Exploratory pilot on the frozen CDVQA evaluation manifest; no official benchmark claim.",
    }


def load_choices_from(provenance):
    return {t: tuple(v) for t, v in provenance["answer_choices"].items()}


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def supervise(output, argv, max_seconds, resume):
    with (Path(output) / "worker.log").open("a" if resume else "x") as log:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, "--worker"],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=max_seconds)
            return {"exit_code": result.returncode,
                    "status": "completed" if result.returncode == 0 else "failed; outputs preserved"}
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "status": "hard runtime limit; partial outputs preserved"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["preflight", "token-preflight", "evaluate", "train"])
    p.add_argument("--weights", type=Path)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--adapter", type=Path, help="evaluate only; omit for base-only (G1) evaluation")
    p.add_argument("--eval", choices=sorted(EVAL_MANIFESTS), default="test1")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--budget-ledger", type=Path)
    p.add_argument("--max-seconds", type=int)
    p.add_argument("--ledger-total", type=int, help="Override the shared-ledger cap (seconds)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if args.stage == "preflight":
        print(json.dumps(preflight(ROOT, args.image_root), indent=2))
        return
    if args.stage == "token-preflight":
        if not args.weights or not args.image_root:
            p.error("token-preflight requires --weights and --image-root")
        report = token_preflight(ROOT, args.weights, args.image_root)
        print(json.dumps(report, indent=2))
        if report["failures"]:
            raise SystemExit(1)
        return
    if not args.worker:
        if not args.budget_ledger or not args.weights or not args.image_root or not args.output or args.max_seconds is None:
            p.error("GPU stages require --weights, --image-root, a fresh --output, --budget-ledger and --max-seconds")
        if not 121 <= args.max_seconds <= LIMITS[args.stage]:
            p.error(f"--max-seconds must be within 121..{LIMITS[args.stage]} for {args.stage}")
        if args.stage == "train" and args.adapter:
            p.error("--adapter is evaluate-only")
        if args.stage == "evaluate" and args.resume_from:
            p.error("--resume-from is train-only")
        preflight(ROOT, args.image_root)
        ledger = args.budget_ledger
        ledger.parent.mkdir(parents=True, exist_ok=True)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "session.json").write_text(json.dumps({
            "stage": args.stage, "run_identity": binding(ROOT), "budget_seconds": args.max_seconds,
            "budget_ledger": str(ledger.resolve()), "resume_from": str(args.resume_from) if args.resume_from else None},
            indent=2))
        lock = ledger.with_suffix(".lock")
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        started = time.monotonic()
        try:
            entries = json.loads(ledger.read_text()) if ledger.exists() else []
            check_reservation(entries, args.max_seconds,
                              ledger_total(load_plan(ROOT), args.stage, args.ledger_total))
            entries.append({"output": str(args.output.resolve()), "stage": args.stage,
                            "charged_seconds": args.max_seconds})
            ledger.write_text(json.dumps(entries, indent=2))
            status = supervise(args.output, argv, args.max_seconds, resume=False)
            entries[-1]["charged_seconds"] = min(args.max_seconds, int(time.monotonic() - started) + 1)
            ledger.write_text(json.dumps(entries, indent=2))
        finally:
            lock.unlink()
        (args.output / "exit.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status))
        raise SystemExit(status["exit_code"])
    began = time.monotonic()
    plan = load_plan(ROOT)
    mismatches = [n for n, v in plan["dependency_pins"].items() if installed(n) is None or installed(n).split("+")[0] != v]
    if mismatches:
        raise ValueError(f"Target dependency pins not provisioned: {mismatches}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("GPU stage requires exactly one visible CUDA GPU")
    if args.stage == "train":
        train_pairs(ROOT, args.weights, args.output, args.image_root, max_seconds=args.max_seconds,
                    began=began, resume_from=args.resume_from)
    else:
        evaluate_gpu(ROOT, args.weights, args.adapter, args.image_root, args.eval, args.output)


if __name__ == "__main__":
    main()
