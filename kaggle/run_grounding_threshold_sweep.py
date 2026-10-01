"""Self-contained Kaggle T4 runner for the zero-shot threshold sweep.

One forward pass per expression (400 total); all 12 threshold combinations
are re-scored offline from the captured raw queries. No training.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle.run_grounding_dior_rsvg import prepare_dataset  # noqa: E402


def main() -> int:
    import torch

    assert torch.cuda.is_available(), "CUDA GPU not found. Select a T4 accelerator in Kaggle first."
    print(f"torch={torch.__version__}; gpu={torch.cuda.get_device_name(0)}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=os.environ.get("SIH26167_REPO_URL"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/sih26167"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/kaggle/temp/dior-rsvg-official")
    )
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=26167)
    parser.add_argument("--git-sha", default=os.environ.get("SIH26167_GIT_SHA"))
    parser.add_argument(
        "--working-tree-sha256", default=os.environ.get("SIH26167_WORKING_TREE_SHA256")
    )
    args = parser.parse_args()

    run = lambda command, cwd=None: (
        print("+", " ".join(command), flush=True),
        subprocess.run(command, cwd=cwd, check=True),
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "gdown",
            "huggingface-hub",
            "groundingdino-py==0.4.0",
            "transformers==4.49.0",
        ]
    )
    if not (
        args.repo_dir
        / "eval"
        / "suites"
        / "grounding_dior_rsvg_threshold_sweep.py"
    ).is_file():
        if args.source_dir:
            shutil.copytree(args.source_dir, args.repo_dir, dirs_exist_ok=True)
        elif args.repo_url:
            run(["git", "clone", args.repo_url, str(args.repo_dir)])
        else:
            raise RuntimeError(
                "Pass --source-dir for an uploaded working tree or set SIH26167_REPO_URL"
            )

    dataset_root = prepare_dataset(args.data_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path("/kaggle/working/results")
        / f"grounding-dino-swint__dior-rsvg__threshold-sweep__{stamp}.json"
    )
    command = [
        sys.executable,
        "eval/suites/grounding_dior_rsvg_threshold_sweep.py",
        "--data-root",
        str(dataset_root),
        "--out",
        str(output),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
    ]
    if args.git_sha:
        command.extend(("--git-sha", args.git_sha))
    if args.working_tree_sha256:
        command.extend(("--working-tree-sha256", args.working_tree_sha256))
    run(command, cwd=args.repo_dir)

    report = json.loads(output.read_text(encoding="utf-8"))
    print("=== SWEEP SUMMARY ===", flush=True)
    for combo in report["combinations"]:
        print(
            f"box={combo['box_threshold']:.2f} text={combo['text_threshold']:.2f} "
            f"Pr@0.5={combo['pr_at_0_5']:.4f} mean_IoU={combo['mean_iou']:.4f}",
            flush=True,
        )
    print(f"Sweep artifact: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
