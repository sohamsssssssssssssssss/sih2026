"""Self-contained Kaggle T4 runner for zero-shot DIOR-RSVG grounding."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_DATASET_URL = (
    "https://drive.google.com/drive/folders/"
    "1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_?usp=sharing"
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _safe_extract(archive: Path, destination: Path) -> None:
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        if any(
            not (destination / member.filename).resolve().is_relative_to(resolved)
            for member in bundle.infolist()
        ):
            raise ValueError(f"Unsafe archive member in {archive}")
        bundle.extractall(destination)


def _find_dataset_root(root: Path, split_dir: str = ".") -> Path | None:
    depth = len(Path(split_dir).parts)
    for candidate in [root, *(path.parents[depth] for path in root.rglob("test.txt") if len(path.parents) > depth)]:
        if (
            (candidate / split_dir / "test.txt").is_file()
            and (candidate / "Annotations").is_dir()
            and (candidate / "JPEGImages").is_dir()
        ):
            return candidate
    return None


def prepare_dataset(root: Path) -> Path:
    existing = _find_dataset_root(root)
    if existing:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            OFFICIAL_DATASET_URL,
            "-O",
            str(root),
        ]
    )
    for archive_name in ("Annotations.zip", "JPEGImages.zip"):
        archives = list(root.rglob(archive_name))
        if len(archives) != 1:
            raise FileNotFoundError(f"Expected one {archive_name}, found {len(archives)}")
        _safe_extract(archives[0], archives[0].parent)
    dataset_root = _find_dataset_root(root)
    if dataset_root is None:
        raise FileNotFoundError("Downloaded DIOR-RSVG layout is incomplete")
    return dataset_root


def main() -> int:
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

    import torch

    assert torch.cuda.is_available(), "CUDA GPU not found. Select a T4 accelerator in Kaggle first."
    print(f"torch={torch.__version__}; gpu={torch.cuda.get_device_name(0)}", flush=True)
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
    if not (args.repo_dir / "eval" / "suites" / "grounding_dior_rsvg.py").is_file():
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
        / f"grounding-dino-swint__dior-rsvg__{stamp}.json"
    )
    command = [
        sys.executable,
        "eval/suites/grounding_dior_rsvg.py",
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
    print(
        f"FINAL Pr@0.5={report['pr_at_0_5']:.6f}; "
        f"mean_IoU={report['mean_iou']:.6f}; n={report['dataset']['n_sampled']}",
        flush=True,
    )
    print(f"Kaggle output artifact: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
