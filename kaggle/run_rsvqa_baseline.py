"""Self-contained Kaggle T4 runner for the frozen Qwen RSVQA-LR baseline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    import torch

    assert torch.cuda.is_available(), "CUDA GPU not found. Select a T4 accelerator in Kaggle first."
    print(f"torch={torch.__version__}; gpu={torch.cuda.get_device_name(0)}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=os.environ.get("SIH26167_REPO_URL"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/sih26167"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.49",
            "qwen-vl-utils",
            "accelerate",
        ]
    )
    if not (args.repo_dir / "eval" / "eval.py").is_file():
        if not args.repo_url:
            raise RuntimeError(
                "Set SIH26167_REPO_URL to a public GitHub clone URL, or copy a Kaggle dataset "
                "upload to /kaggle/working/sih26167 before running this script."
            )
        run(["git", "clone", args.repo_url, str(args.repo_dir)])

    output = Path("/kaggle/working/results.json")
    command = [
        sys.executable,
        "eval/eval.py",
        "--model",
        "qwen2.5vl-3b",
        "--suite",
        "rsvqa",
        "--out",
        str(output),
    ]
    if args.full:
        command.append("--full")
    run(command, cwd=args.repo_dir)
    report = json.loads(output.read_text(encoding="utf-8"))
    print(f"accuracy={report['accuracy']:.6f}; n_samples={report['n_samples']}")
    print(f"Kaggle output artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
