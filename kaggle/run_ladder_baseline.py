"""Self-contained Kaggle T4 runner for the frozen Qwen resolution ladder baseline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def link_contents(source: Path, destination: Path) -> None:
    """Expose a read-only Kaggle Dataset inside the working repository."""
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        destination_path = destination / source_path.name
        if destination_path.exists() or destination_path.is_symlink():
            continue
        destination_path.symlink_to(
            source_path.resolve(), target_is_directory=source_path.is_dir()
        )


def has_loveda_images(root: Path) -> bool:
    return root.is_dir() and any(
        path.parent.name == "images_png" for path in root.rglob("*.png")
    )


def find_kaggle_dataset(name: str) -> Path:
    direct = Path("/kaggle/input") / name
    if direct.is_dir():
        return direct
    slug = name.rsplit("/", 1)[-1]
    versioned = [
        path
        for path in Path("/kaggle/input/datasets").glob(f"*/{slug}/versions/*")
        if path.is_dir()
    ]
    if versioned:
        return max(versioned, key=lambda path: int(path.name) if path.name.isdigit() else -1)
    raise FileNotFoundError(
        f"Kaggle Dataset input {name!r} was not found under /kaggle/input"
    )


def main() -> int:
    import torch

    assert torch.cuda.is_available(), "CUDA GPU not found. Select a T4 accelerator in Kaggle first."
    print(f"torch={torch.__version__}; gpu={torch.cuda.get_device_name(0)}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default=os.environ.get("SIH26167_REPO_URL"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/sih26167"))
    parser.add_argument(
        "--ladder-source",
        choices=("kaggle-dataset", "regenerate"),
        default="kaggle-dataset",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.environ.get("SIH26167_LADDER_DATASET"),
        help=(
            "Kaggle input name under /kaggle/input; contains rendered ladder data in "
            "kaggle-dataset mode or extracted LoveDA data in regenerate mode"
        ),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")

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

    manifest = args.repo_dir / "data" / "ladder" / "manifest.jsonl"
    if not manifest.is_file():
        if args.ladder_source == "kaggle-dataset":
            if not args.dataset_name:
                raise RuntimeError(
                    "Pass --dataset-name NAME for the Kaggle Dataset mounted at "
                    "/kaggle/input/NAME containing manifest.jsonl and rendered ladder tiles."
                )
            dataset_root = find_kaggle_dataset(args.dataset_name)
            manifests = sorted(dataset_root.rglob("manifest.jsonl"))
            if len(manifests) != 1:
                raise RuntimeError(
                    f"Expected exactly one manifest.jsonl under {dataset_root}, "
                    f"found {len(manifests)}: {manifests}"
                )
            link_contents(manifests[0].parent, manifest.parent)
        else:
            raw_root = args.repo_dir / "data" / "raw" / "loveda"
            if not has_loveda_images(raw_root):
                if not args.dataset_name:
                    raise RuntimeError(
                        "LoveDA is not present under data/raw/loveda. Pass --dataset-name NAME "
                        "for a Kaggle Dataset mounted at /kaggle/input/NAME containing the "
                        "extracted LoveDA Train/ and Val/ directories."
                    )
                dataset_root = find_kaggle_dataset(args.dataset_name)
                raw_root.parent.mkdir(parents=True, exist_ok=True)
                raw_root.symlink_to(dataset_root, target_is_directory=True)
            if not has_loveda_images(raw_root):
                raise FileNotFoundError(
                    f"No extracted LoveDA images_png directories found under {raw_root}. "
                    "Upload or mount LoveDA with Train.zip and Val.zip already extracted."
                )
            run(
                [sys.executable, "eval/ladder.py", "--limit", str(args.limit)],
                cwd=args.repo_dir,
            )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Ladder manifest was not created at {manifest}. For --ladder-source "
            "kaggle-dataset, mount a dataset containing manifest.jsonl plus its rendered "
            "tiles and pass --dataset-name NAME. For --ladder-source regenerate, mount "
            "extracted LoveDA Train/ and Val/ data and pass --dataset-name NAME."
        )

    output = Path("/kaggle/working/results_ladder.json")
    run(
        [
            sys.executable,
            "eval/eval.py",
            "--model",
            "qwen2.5vl-3b",
            "--suite",
            "ladder",
            "--out",
            str(output),
        ],
        cwd=args.repo_dir,
    )
    curve = Path("/kaggle/working/ladder_curve.png")
    run(
        [sys.executable, "eval/plot_ladder.py", str(output), "--out", str(curve)],
        cwd=args.repo_dir,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    print("Resolution ladder summary:")
    for rung, values in sorted(report["per_rung"].items(), key=lambda item: float(item[0])):
        print(f"  {rung} m/pixel: accuracy={values['accuracy']:.6f}; n={values['n']}")
    print(f"Kaggle output artifacts: {output}, {curve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
