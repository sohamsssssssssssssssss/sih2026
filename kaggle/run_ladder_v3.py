"""Run the registered ladder from read-only, hash-checked Kaggle inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from eval.ladder import RUNGS, degrade
from eval.ladder_v3 import BOOT_SEED, REPS, SEED, TYPES, canonical_answer, ci, metrics, prompt, sha

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/kaggle/working/ladder-v3/results/ladder-v3")
MANIFEST = ROOT / "data/manifests/ladder-v3/questions.jsonl"
GATE = ROOT / "docs/gates/LADDER-V3.md"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
BASE_HASHES = {
    "config.json": "7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5",
    "preprocessor_config.json": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "tokenizer_config.json": "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "model-00001-of-00002.safetensors": "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8",
    "model-00002-of-00002.safetensors": "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4",
}
ADAPTER_HASHES = {
    "adapter_model.safetensors": "7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a",
    "adapter_config.json": "716251f9cfbb73ed5fb408e41f2eb2b30281b8f98fb7c6a92a1dd0b9204e3644",
}
CONDITIONS = ("base_real", "base_constant", "b13_real", "b13_constant")


def require(condition, message):
    if not condition:
        raise SystemExit(f"HARD STOP: {message}")


def find_input(filename, dataset, parent=None):
    mount = Path("/kaggle/input")
    matches = [p for p in mount.rglob(filename) if dataset in p.relative_to(mount).parts
               and (parent is None or p.parent.name == parent)]
    require(len(matches) == 1, f"expected one {filename} in {dataset}, found {matches}")
    return matches[0]


def validate_hashes(root, expected):
    for filename, digest in expected.items():
        path = root / filename
        require(path.is_file() and sha(path) == digest, f"failed hash check for {path}")


def summarize(results, tiles):
    draws = np.random.default_rng(BOOT_SEED).integers(len(tiles), size=(REPS, len(tiles)))
    by_key = defaultdict(list)
    for row in results:
        by_key[row["condition"], row["gsd"], row["type"]].append(row)
        if row["type"].startswith("presence:"):
            by_key[row["condition"], row["gsd"], "presence"].append(row)
    summaries, accuracy = {}, {}
    for (cond, gsd, task), rows in by_key.items():
        labels = TYPES.get(task, ("yes", "no"))
        if task == "dominant":
            labels = tuple(label for label in labels if any(r["gold"] == label for r in rows))
        report, distribution = metrics(rows, tiles, draws, labels)
        key = cond, gsd, task
        summaries[key] = report
        accuracy[key] = distribution
    comparisons = []
    for cond, gsd, task in summaries:
        if not cond.endswith("_real"):
            continue
        control = cond.replace("_real", "_constant")
        real, constant = summaries[cond, gsd, task], summaries[control, gsd, task]
        interval = ci(accuracy[cond, gsd, task] - accuracy[control, gsd, task])
        comparisons.append(dict(condition=cond, gsd=gsd, type=task,
                                difference=real["accuracy"] - constant["accuracy"],
                                ci95=interval, image_dependent=bool(interval and (interval[0] > 0 or interval[1] < 0))))
    cells = [dict(condition=c, gsd=g, type=t, **m) for (c, g, t), m in summaries.items()]
    return cells, comparisons


def write_report(rows, comparisons, commit, config_hash):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["# Ladder v3", "", f"Git commit `{commit}`. Gate SHA-256 `{config_hash}`.", "",
             "See results.json for intervals and per-question results.", "",
             "| Condition | GSD | Task | Accuracy | Balanced accuracy | Yes rate | Degenerate |",
             "|---|---:|---|---:|---:|---:|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    for row in rows:
        lines.append("| {condition} | {gsd:g} | {type} | {acc} | {bal} | {yes} | {degenerate} |".format(
            **row, acc=fmt(row["accuracy"]), bal=fmt(row["balanced_accuracy"]), yes=fmt(row["yes_rate"])))
    lines.extend(["", "| Real condition | GSD | Task | Real minus grey [95% CI] | Image-dependent |",
                  "|---|---:|---|---|---|"])
    for row in comparisons:
        lines.append(f"| {row['condition']} | {row['gsd']:g} | {row['type']} | "
                     f"{row['difference']:+.3f} {row['ci95']} | {row['image_dependent']} |")
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, task in zip(axes, ("presence", "dominant", "domain")):
        for cond in CONDITIONS:
            selected = sorted((r for r in rows if r["type"] == task and r["condition"] == cond), key=lambda r: r["gsd"])
            ax.plot([r["gsd"] for r in selected], [r["accuracy"] for r in selected], marker="o", label=cond)
        ax.set(title=task, xlabel="GSD (m/pixel)", ylim=(0, 1))
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "curve.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", required=True)
    args = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(args.commit == head, "checkout differs from requested commit")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "checkout has local changes")
    require(sha(MANIFEST) in GATE.read_text(), "manifest hash differs from registration")
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")

    tiles = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    require(len(tiles) == len({r["tile"] for r in tiles}) == 200, "invalid cohort")
    example = find_input(Path(tiles[0]["image"]).name, "loveda", "images_png")
    require(example.parent.name == "images_png", f"unexpected LoveDA layout: {example}")
    source = example.parent.parent
    for tile in tiles:
        for key in ("image", "mask"):
            path = source.joinpath(*Path(tile[key]).parts[-2:])
            require(path.is_file() and sha(path) == tile[f"{key}_sha256"], f"LoveDA input mismatch: {path}")
    base = find_input("model-00001-of-00002.safetensors", "satquery-qwen-pinned").parent
    validate_hashes(base, BASE_HASHES)
    adapter = find_input("adapter_model.safetensors", "b13-final-750-zip", "adapter").parent
    require(adapter.parent.name == "b13-resume-112-b", "expected the final B13 adapter")
    validate_hashes(adapter, ADAPTER_HASHES)

    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    require(torch.cuda.is_available(), "select a Kaggle CUDA accelerator")
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    OUTPUT.mkdir(parents=True)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True, revision=REVISION)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, local_files_only=True, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    results = []

    def run_pair(model, prefix):
        rng = np.random.default_rng(SEED)
        for index, tile in enumerate(tiles, 1):
            with Image.open(source.joinpath(*Path(tile["image"]).parts[-2:])) as image:
                rgb = np.asarray(image.convert("RGB"))
            for gsd in RUNGS:
                pixels, _ = degrade(rgb, gsd, rng)
                variants = {"real": Image.fromarray(pixels),
                            "constant": Image.new("RGB", (pixels.shape[1], pixels.shape[0]), (127, 127, 127))}
                for variant, image in variants.items():
                    for row in tile["questions"]:
                        messages = [{"role": "user", "content": [
                            {"type": "image", "image": image, "resized_height": 392, "resized_width": 392},
                            {"type": "text", "text": prompt(row["type"])},
                        ]}]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        images, videos = process_vision_info(messages)
                        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
                        with torch.inference_mode():
                            ids = model.generate(**inputs, do_sample=False, max_new_tokens=8)
                        raw = processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
                        results.append(dict(tile=tile["tile"], gsd=gsd, condition=f"{prefix}_{variant}",
                                            raw=raw, answer=canonical_answer(raw, TYPES[row["type"]]), **row))
            print(f"{prefix}: tile {index}/{len(tiles)}", flush=True)

    run_pair(model, "base")
    run_pair(PeftModel.from_pretrained(model, adapter, is_trainable=False).eval(), "b13")
    expected = sum(len(tile["questions"]) for tile in tiles) * len(RUNGS) * len(CONDITIONS)
    require(len(results) == expected, "missing predictions")
    cells, comparisons = summarize(results, [tile["tile"] for tile in tiles])
    data = dict(commit=head, config_sha256=sha(GATE), manifest_sha256=sha(MANIFEST), base_revision=REVISION,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                metrics=cells, contrasts=comparisons, predictions=results)
    (OUTPUT / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    write_report(cells, comparisons, head, data["config_sha256"])
    print("CELL DONE", flush=True)


if __name__ == "__main__":
    main()
