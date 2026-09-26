"""Kaggle T4 runner for the SatQuery VLM program (research brief Days 1-8).

Every GPU stage shells out to the existing training/evaluation scripts in a fresh
process (each run starts with an empty CUDA context), then writes one immutable
registry record per experiment under ``<work>/registry/``. Bring that directory and the
small JSON reports back into ``eval/registry/`` and ``eval/results/``; never adapters,
checkpoints, weights or dataset pixels.

Research code: no stage makes a capability claim. See docs/research/vlm-kaggle-runbook.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.registry import REGISTRY_DIR, write_record  # noqa: E402
from eval.rsvqa_research import paired_comparison  # noqa: E402
from training.remote_sensing import sha256_file  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MODEL_FILES = ROOT / "configs" / "research" / "qwen2.5-vl-3b-instruct@66285546.files.json"
DATASET, DATASET_VERSION = "RSVQA-LR", "Zenodo record 6344334"
MANIFEST_SHA256 = "23a74c573026a26620262eb2440a7effa10168f77010f40dee81a77996fed3e1"
DEV_SUBSET = ROOT / "eval" / "subsets" / "rsvqa-lr-validation-dev1000.v1.json"
DEV_SUBSET_SHA256 = "e3b43319645767c2c559fba82413f66818193689d4316e067f7339eb9ec4e20c"
AUDIT = ROOT / "eval" / "results" / "rsvqa-lr-spatial-audit.v1.json"
TEST_SUBSET_SHA256 = "b21e5046fef76f0f21ececcfe8c3c6cc9c3a844dd054c8c00ad1ca78875057b5"
# 256 px RSVQA-LR patches -> 252: the multiple of 28 qwen_vl_utils also picks for the
# un-resized provider path, so no pixels are invented by upsampling.
IMAGE_SIZE = 252
SEED = 17
FIXED = {
    "epochs": 1.0,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "precision": "fp16",
    "quantization": "4bit",
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
    "image_size": IMAGE_SIZE,
    "seed": SEED,
}
BASELINE_HP = {"lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4}
LADDER = (100, 500, 1000, 5000)
# alpha/rank is held at 2 so the rank arms change capacity, not the LoRA update scale.
# Rank 16 at 2e-4 is not repeated: it is the ladder run at the same sample count.
MATRIX = (
    {"lora_rank": 8, "lora_alpha": 16, "learning_rate": 2e-4},
    {"lora_rank": 32, "lora_alpha": 64, "learning_rate": 2e-4},
    {"lora_rank": 16, "lora_alpha": 32, "learning_rate": 1e-4},
    {"lora_rank": 16, "lora_alpha": 32, "learning_rate": 4e-4},
)
# ASSUMED, NOT MEASURED (low, high) seconds on a T4. Replace with the measured values from
# the base-eval and ladder N=100 records before trusting any later session plan.
ASSUMED_EVAL_SECONDS_PER_SAMPLE = (0.25, 0.6)
ASSUMED_TRAIN_SECONDS_PER_SAMPLE = (0.6, 1.5)
ASSUMED_MODEL_LOAD_SECONDS = (30, 120)
PACKAGES = ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes",
            "qwen-vl-utils", "huggingface-hub", "safetensors", "numpy", "pillow")
BASE_V0 = "rsvqa-qwen-base-v0.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - content identity


def verify_model_files(model_dir: Path, manifest: Path = MODEL_FILES) -> dict:
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    problems = []
    for name, spec in sorted(expected["files"].items()):
        path = model_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif path.stat().st_size != spec["size"]:
            problems.append(f"size mismatch {name}")
        elif "sha256" in spec and sha256_file(path) != spec["sha256"]:
            problems.append(f"sha256 mismatch {name}")
        elif "git_blob_sha1" in spec and blob_sha1(path) != spec["git_blob_sha1"]:
            problems.append(f"blob sha1 mismatch {name}")
    return {"revision": expected["revision"], "verified": not problems, "problems": problems}


def next_experiment_id(date: str, directories: list[Path]) -> str:
    used = {
        int(path.stem[-3:])
        for directory in directories
        if directory.is_dir()
        for path in directory.glob(f"SQ-{date}-1??.json")
    }
    free = next((number for number in range(100, 200) if number not in used), None)
    if free is None:
        raise RuntimeError(f"VLM experiment block 100-199 is exhausted for {date}")
    return f"SQ-{date}-{free}"


def estimate_hours(train_samples: int, eval_samples: int, loads: int) -> list[float]:
    return [
        round((train_samples * train + eval_samples * evaluation + loads * load) / 3600, 2)
        for train, evaluation, load in zip(
            ASSUMED_TRAIN_SECONDS_PER_SAMPLE, ASSUMED_EVAL_SECONDS_PER_SAMPLE, ASSUMED_MODEL_LOAD_SECONDS
        )
    ]


def session_plan(matrix_samples: int = 1000, final_samples: int = 20000) -> list[dict]:
    dev = len(json.loads(DEV_SUBSET.read_text(encoding="utf-8"))["samples"])
    test = 10004
    sessions = [
        ("S1 Day 1-3", "env, load-check, dry-run, base-eval", 1, 1 + dev, 3),
        ("S2 Day 4-5", "ladder " + "/".join(map(str, LADDER)), sum(LADDER), dev * len(LADDER), 2 * len(LADDER)),
        ("S3 Day 6", f"matrix {len(MATRIX)} runs at N={matrix_samples}", matrix_samples * len(MATRIX),
         dev * len(MATRIX), 2 * len(MATRIX)),
        ("S4 Day 7", f"train-eval N={final_samples} (chosen from S2/S3 evidence)", final_samples, dev, 2),
        ("S5 Day 8", "lock + final-test base and adapter on the full official test split", 0, 2 * test, 1),
    ]
    return [
        {"session": name, "stages": stages, "estimated_hours_low_high": estimate_hours(train, evaluation, loads)}
        for name, stages, train, evaluation, loads in sessions
    ]


def capture_environment() -> dict:
    import torch

    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    smi = None
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    return {
        "captured_at": now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi_name_driver_memory": smi,
        "packages": packages,
    }


class Program:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work = args.work_dir
        self.registry = self.work / "registry"

    # --- helpers ---------------------------------------------------------

    def environment(self) -> dict:
        path = self.work / "environment.json"
        if not path.is_file():
            raise RuntimeError("run the env stage first in every Kaggle session")
        environment = json.loads(path.read_text(encoding="utf-8"))
        if not environment.get("preflight_passed"):
            raise RuntimeError("the env stage failed in this session; see its registry record")
        return environment

    def new_id(self) -> str:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return next_experiment_id(date, [self.registry, REGISTRY_DIR])

    def run(self, command: list[str], log: Path) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, check=False).returncode

    def register(self, experiment_id: str, title: str, *, split: str, started: str, status: str,
                 metrics: dict, commands: list, artifacts: dict, hyperparameters=None,
                 adapter_revision=None, notes: str = "", environment=None) -> Path:
        record = {
            "experiment_id": experiment_id,
            "title": title,
            "dataset": DATASET,
            "dataset_version": DATASET_VERSION,
            "dataset_checksum": {"manifest_sha256": MANIFEST_SHA256},
            "split": split,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapter_revision": adapter_revision,
            "seed": SEED,
            "hyperparameters": hyperparameters,
            "command": {"runner": " ".join(sys.argv), "subprocesses": commands},
            "started_at": started,
            "ended_at": now(),
            "metrics": metrics,
            "artifact_paths": {key: str(value) for key, value in artifacts.items()},
            "status": status,
            "notes": notes,
            "environment": environment or self.environment(),
        }
        path = write_record(record, registry_dir=self.registry)
        print(f"[{status}] {experiment_id} {title} -> {path}", flush=True)
        return path

    def train_command(self, out: Path, max_samples: int, hp: dict, dry_run: bool = False) -> list[str]:
        settings = {**FIXED, **hp, "max_samples": max_samples}
        command = [
            sys.executable, str(ROOT / "scripts" / "train_remote_sensing_adapter.py"),
            "--model-path", str(self.args.model_dir), "--model-revision", MODEL_REVISION,
            "--dataset-manifest", str(self.args.manifest), "--image-root", str(self.args.rsvqa_root),
            "--output-dir", str(out),
        ]
        command += [item for key, value in settings.items() for item in (f"--{key.replace('_', '-')}", str(value))]
        if self.args.resume_from_checkpoint and self.args.stage == "train-eval":
            command += ["--resume-from-checkpoint", str(self.args.resume_from_checkpoint)]
        return command + (["--dry-run"] if dry_run else [])

    def eval_command(self, subset: Path, expected_sha256: str | None, out: Path,
                     adapter: Path | None = None, skip_base: bool = False) -> list[str]:
        command = [
            sys.executable, str(ROOT / "scripts" / "evaluate_remote_sensing_adapter.py"),
            "--model-path", str(self.args.model_dir), "--model-revision", MODEL_REVISION,
            "--dataset-manifest", str(self.args.manifest), "--image-root", str(self.args.rsvqa_root),
            "--subset", str(subset), "--image-size", str(IMAGE_SIZE), "--out", str(out),
        ]
        if expected_sha256:
            command += ["--expected-subset-sha256", expected_sha256]
        if adapter:
            command += ["--adapter-path", str(adapter)]
        return command + (["--skip-base"] if skip_base else [])

    @staticmethod
    def eval_metrics(report: dict, key: str) -> dict:
        part = report[key]
        summary = part["summary"]
        metrics = {
            name: summary.get(name)
            for name in ("n", "strict_accuracy", "strict_wilson_95", "lenient_accuracy", "invalid_rate",
                         "type_weighted_strict_accuracy", "pred_yes_rate_on_binary", "degenerate_binary",
                         "per_type")
        }
        metrics["eval_seconds"] = part["latency_seconds"]
        metrics["eval_peak_cuda_memory_bytes"] = part["peak_cuda_memory_bytes"]
        if "footprint_disjoint_summary" in part:
            metrics["footprint_disjoint"] = {
                name: part["footprint_disjoint_summary"][name] for name in ("n", "strict_accuracy", "lenient_accuracy")
            }
        return metrics

    @staticmethod
    def status_for(metrics: dict) -> str:
        return "INCONCLUSIVE" if metrics.get("degenerate_binary") else "PASSED"

    # --- stages ------------------------------------------------------------

    def stage_plan(self) -> None:
        runs = [{"stage": "ladder", "max_samples": n, **BASELINE_HP} for n in LADDER]
        runs += [{"stage": "matrix", "max_samples": self.args.matrix_samples, **hp} for hp in MATRIX]
        print(json.dumps({
            "fixed": FIXED,
            "runs": runs,
            "assumed_seconds_not_measured": {
                "eval_per_sample": ASSUMED_EVAL_SECONDS_PER_SAMPLE,
                "train_per_sample": ASSUMED_TRAIN_SECONDS_PER_SAMPLE,
                "model_load": ASSUMED_MODEL_LOAD_SECONDS,
            },
            "sessions": session_plan(self.args.matrix_samples, self.args.max_samples or 20000),
        }, indent=2))

    def stage_env(self) -> None:
        started, experiment_id = now(), self.new_id()
        environment = capture_environment()
        model = verify_model_files(self.args.model_dir)
        manifest_sha256 = sha256_file(self.args.manifest)
        passed = environment["cuda_available"] and model["verified"] and manifest_sha256 == MANIFEST_SHA256
        environment["preflight_passed"] = passed
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
        self.register(
            experiment_id, "Day 1 GPU environment and pinned artifact verification",
            split="all", started=started, status="PASSED" if passed else "FAILED",
            metrics={"model_files": model, "manifest_sha256": manifest_sha256,
                     "manifest_matches_reference": manifest_sha256 == MANIFEST_SHA256},
            commands=[], artifacts={"environment": self.work / "environment.json"},
            environment=environment,
        )

    def stage_load_check(self) -> None:
        started, experiment_id = now(), self.new_id()
        out = self.work / experiment_id
        dev = json.loads(DEV_SUBSET.read_text(encoding="utf-8"))
        one = out / "one-sample-subset.json"
        out.mkdir(parents=True, exist_ok=True)
        one.write_text(json.dumps({**dev, "samples": dev["samples"][:1]}), encoding="utf-8")
        command = self.eval_command(one, None, out / "eval-report.json")
        code = self.run(command, out / "eval.log")
        metrics = {"exit_code": code}
        if code == 0:
            report = json.loads((out / "eval-report.json").read_text(encoding="utf-8"))
            metrics |= {"model_load_seconds": report["model_load_seconds"],
                        "answer": report["base"]["results"][0]["prediction"],
                        "peak_cuda_memory_bytes": report["base"]["peak_cuda_memory_bytes"]}
        self.register(experiment_id, "Day 1 offline model load and one-sample generation",
                      split="validation", started=started, status="PASSED" if code == 0 else "FAILED",
                      metrics=metrics, commands=[command], artifacts={"dir": out},
                      notes="Operational check only; one sample is not a measurement.")

    def stage_dry_run(self) -> None:
        started, experiment_id = now(), self.new_id()
        out = self.work / experiment_id
        command = self.train_command(out, 1, BASELINE_HP, dry_run=True)
        code = self.run(command, out / "train.log")
        report_path = out / "training-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        adapter_emitted = (out / "adapter").exists()
        passed = code == 0 and report.get("status") == "dry_run_passed" and not adapter_emitted
        self.register(
            experiment_id, "Day 2 QLoRA one-sample gradient dry run", split="train", started=started,
            status="PASSED" if passed else "FAILED",
            metrics={"exit_code": code, "report_status": report.get("status"), "adapter_emitted": adapter_emitted,
                     "peak_cuda_memory_bytes": report.get("peak_cuda_memory_bytes"),
                     "wall_clock_seconds": report.get("wall_clock_seconds")},
            commands=[command], artifacts={"dir": out}, hyperparameters={**FIXED, **BASELINE_HP, "max_samples": 1},
            notes="Checks image loading, processor, tokenisation, collation, QLoRA attachment, finite loss "
                  "and finite adapter gradients; performs no optimizer step.",
        )

    def stage_base_eval(self) -> None:
        started, experiment_id = now(), self.new_id()
        out = self.work / experiment_id
        command = self.eval_command(DEV_SUBSET, DEV_SUBSET_SHA256, out / "eval-report.json")
        code = self.run(command, out / "eval.log")
        metrics, status = {"exit_code": code}, "FAILED"
        if code == 0:
            report = json.loads((out / "eval-report.json").read_text(encoding="utf-8"))
            metrics = self.eval_metrics(report, "base")
            status = self.status_for(metrics)
            shutil.copyfile(out / "eval-report.json", self.work / BASE_V0)
        self.register(experiment_id, "Day 3 BASELINE V0: base Qwen on the validation dev subset",
                      split="validation", started=started, status=status, metrics=metrics,
                      commands=[command], artifacts={"report": out / "eval-report.json"},
                      notes=f"Subset {DEV_SUBSET.name} sha256 {DEV_SUBSET_SHA256}.")

    def train_and_evaluate(self, title: str, max_samples: int, hp: dict) -> None:
        started, experiment_id = now(), self.new_id()
        out = self.work / experiment_id
        train = self.train_command(out, max_samples, hp)
        commands, metrics, status, adapter_sha = [train], {}, "FAILED", None
        code = self.run(train, out / "train.log")
        metrics["train_exit_code"] = code
        if code == 0:
            report = json.loads((out / "training-report.json").read_text(encoding="utf-8"))
            adapter_sha = sha256_file(out / "adapter" / "adapter_model.safetensors")
            metrics |= {
                "train_loss": report["trainer_metrics"].get("train_loss"),
                "train_runtime_seconds": report["trainer_metrics"].get("train_runtime"),
                "train_peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
                "adapter_size_bytes": report["adapter_size_bytes"],
                "loss_history": [
                    {"step": entry["step"], "loss": entry["loss"]}
                    for entry in report["trainer_log_history"] if "loss" in entry
                ],
            }
            evaluation = self.eval_command(DEV_SUBSET, DEV_SUBSET_SHA256, out / "eval-report.json",
                                           adapter=out / "adapter", skip_base=True)
            commands.append(evaluation)
            code = self.run(evaluation, out / "eval.log")
            metrics["eval_exit_code"] = code
            if code == 0:
                report = json.loads((out / "eval-report.json").read_text(encoding="utf-8"))
                metrics |= self.eval_metrics(report, "adapted")
                status = self.status_for(metrics)
                base_v0 = self.work / BASE_V0
                if base_v0.is_file():
                    base = json.loads(base_v0.read_text(encoding="utf-8"))["base"]["results"]
                    metrics["paired_vs_base_v0"] = paired_comparison(base, report["adapted"]["results"])
        self.register(
            experiment_id, title, split="train->validation", started=started, status=status, metrics=metrics,
            commands=commands, artifacts={"dir": out}, hyperparameters={**FIXED, **hp, "max_samples": max_samples},
            adapter_revision=adapter_sha,
            notes="Training loss is not accuracy; validation dev subset is for selection only.",
        )

    def stage_ladder(self) -> None:
        for count in LADDER:
            self.train_and_evaluate(f"Day 4-5 training-size ladder N={count}", count, BASELINE_HP)

    def stage_matrix(self) -> None:
        for hp in MATRIX:
            self.train_and_evaluate(
                f"Day 6 LoRA matrix r={hp['lora_rank']} lr={hp['learning_rate']} N={self.args.matrix_samples}",
                self.args.matrix_samples, hp,
            )

    def stage_train_eval(self) -> None:
        if not self.args.max_samples:
            raise SystemExit("train-eval needs --max-samples and the chosen --lora-rank/--lora-alpha/--learning-rate")
        hp = {"lora_rank": self.args.lora_rank, "lora_alpha": self.args.lora_alpha,
              "learning_rate": self.args.learning_rate}
        self.train_and_evaluate(f"Day 7 selected configuration N={self.args.max_samples}", self.args.max_samples, hp)

    def stage_lock(self) -> None:
        adapter = self.args.adapter_path
        if not adapter or not self.args.selected_from or not self.args.rationale:
            raise SystemExit("lock needs --adapter-path, --selected-from and --rationale")
        lock = {
            "locked_at": now(),
            "adapter_path": str(adapter.resolve()),
            "adapter_weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "training_report": json.loads((adapter.parent / "training-report.json").read_text(encoding="utf-8")),
            "selected_from": self.args.selected_from,
            "rationale": self.args.rationale,
        }
        with (self.work / "locked-config.json").open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)
        print(f"locked {lock['adapter_weights_sha256']}")

    def stage_final_test(self) -> None:
        lock = json.loads((self.work / "locked-config.json").read_text(encoding="utf-8"))
        adapter = Path(lock["adapter_path"])
        if sha256_file(adapter / "adapter_model.safetensors") != lock["adapter_weights_sha256"]:
            raise RuntimeError("adapter changed after it was locked")
        started, experiment_id = now(), self.new_id()
        out = self.work / experiment_id
        subset = self.work / "rsvqa-lr-test-locked.v1.json"
        commands = []
        if not subset.is_file():
            build = [sys.executable, "-m", "eval.rsvqa_research", "subset", "--dataset-root", str(self.args.rsvqa_root),
                     "--split", "test", "--seed", "26167", "--audit", str(AUDIT), "--leakage", "flag", "--out", str(subset)]
            commands.append(build)
            if self.run(build, out / "subset.log"):
                raise RuntimeError("could not build the locked test subset")
        command = self.eval_command(subset, TEST_SUBSET_SHA256, out / "eval-report.json", adapter=adapter)
        commands.append(command)
        code = self.run(command, out / "eval.log")
        metrics, status = {"exit_code": code}, "FAILED"
        if code == 0:
            report = json.loads((out / "eval-report.json").read_text(encoding="utf-8"))
            metrics = {"base": self.eval_metrics(report, "base"), "adapted": self.eval_metrics(report, "adapted"),
                       "paired": report["paired"]}
            status = "INCONCLUSIVE" if "INCONCLUSIVE" in (
                self.status_for(metrics["base"]), self.status_for(metrics["adapted"])) else "PASSED"
        self.register(
            experiment_id, "Day 8 locked final: base vs adapter on the official RSVQA-LR test split",
            split="test", started=started, status=status, metrics=metrics, commands=commands,
            artifacts={"report": out / "eval-report.json", "lock": self.work / "locked-config.json"},
            hyperparameters=lock["training_report"]["configuration"], adapter_revision=lock["adapter_weights_sha256"],
            notes="Official test split, run once after locking. Footprint-disjoint numbers exclude the 4 test "
                  "images whose footprints overlap train patches (eval/results/rsvqa-lr-spatial-audit.v1.json).",
        )


STAGES = ("plan", "env", "load-check", "dry-run", "base-eval", "ladder", "matrix", "train-eval", "lock", "final-test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/satquery-vlm"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--rsvqa-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--matrix-samples", type=int, default=1000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lora-rank", type=int, default=BASELINE_HP["lora_rank"])
    parser.add_argument("--lora-alpha", type=int, default=BASELINE_HP["lora_alpha"])
    parser.add_argument("--learning-rate", type=float, default=BASELINE_HP["learning_rate"])
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--selected-from", nargs="+")
    parser.add_argument("--rationale")
    args = parser.parse_args(argv)
    needs_paths = args.stage not in ("plan", "lock")
    if needs_paths and not (args.model_dir and args.rsvqa_root and args.manifest):
        parser.error(f"{args.stage} needs --model-dir, --rsvqa-root and --manifest")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    program = Program(args)
    getattr(program, f"stage_{args.stage.replace('-', '_')}")()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
