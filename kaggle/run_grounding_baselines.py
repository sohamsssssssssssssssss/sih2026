"""Produce raw grounding-v1 predictions from a checksum-pinned private pack."""
import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
from io import BytesIO
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import zipfile

from PIL import Image

SEED = 17
DINO_ID = "ShilongLiu/GroundingDINO"
DINO_REVISION = "a94c9b567a2a374598f05c584e96798a170c56fb"
DINO_WEIGHT = "groundingdino_swint_ogc.pth"
DINO_SHA = "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
GRID = tuple(product((0.20, 0.25, 0.30, 0.35), (0.15, 0.20, 0.25)))
PROMPT = ("Locate the object described by: {expression}. Return exactly one JSON array "
          "[x_min, y_min, x_max, y_max] with integer coordinates normalized to 0–1000. Return no other text.")
CAPTION = "Describe this aerial image in one detailed paragraph."
REPO = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(f"HARD STOP: {message}")


def pack_items(pack):
    manifest = json.loads(pack.read("MANIFEST.json"))
    names = [row["file"] for row in manifest]
    require(len(names) == len(set(names)) and sorted(pack.namelist()) == sorted([*names, "MANIFEST.json"]), "pack file list mismatch")
    for row in manifest:
        data = pack.read(row["file"])
        require(len(data) == row["bytes"] and sha256(data).hexdigest() == row["sha256"], row["file"])
    frozen = REPO / "results/grounding"
    require(pack.read("frozen/MANIFEST.json") == (frozen / "MANIFEST.json").read_bytes(), "frozen manifest mismatch")
    items = [json.loads(line) for line in pack.read("items.jsonl").splitlines()]
    for dataset, split in product(("dior-rsvg", "vrsbench"), ("select", "eval")):
        name = f"{dataset}-{split}.jsonl"
        data = pack.read(f"frozen/{name}")
        require(data == (frozen / name).read_bytes(), name)
        expected = [json.loads(line) for line in data.splitlines()]
        actual = [row for row in items if (row["dataset"], row["split"]) == (dataset, split)]
        require(len(expected) == len(actual), f"item count mismatch: {name}")
        for source, row in zip(expected, actual):
            source["image"] = f"images/{dataset}/{source['image']}"
            require(all(row[key] == val for key, val in source.items()), f"item mismatch: {row['id']}")
    require(len({row["id"] for row in items}) == len(items), "duplicate IDs")
    return items


def parse_box(text):
    try:
        box = json.loads(text)
    except ValueError:
        return None
    if (isinstance(box, list) and len(box) == 4 and all(type(v) is int and 0 <= v <= 1000 for v in box)
            and box[0] < box[2] and box[1] < box[3]):
        return [v / 1000 for v in box]
    return None


def shuffled(rows):
    donors = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        ordered = sorted((row for row in rows if row["dataset"] == dataset),
                         key=lambda row: sha256(f"17:shuffle:{row['parent']}".encode()).digest())
        for row, donor in zip(ordered, ordered[1:] + ordered[:1]):
            require(row["parent"] != donor["parent"], "shuffle requires distinct parents")
            donors[row["id"]] = donor
    return donors


def qwen_backend(path):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
    ).eval().to("cuda")

    def predict(image, prompt, tokens):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        return processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return predict


def dino_backend():
    import torch
    import groundingdino.datasets.transforms as transforms
    from groundingdino.util.inference import predict
    from huggingface_hub import hf_hub_download
    from models.grounding_dino.model import GroundingDINOModel

    weight = Path(hf_hub_download(DINO_ID, DINO_WEIGHT, revision=DINO_REVISION))
    require(digest(weight) == DINO_SHA, "DINO checkpoint SHA-256")
    os.environ["SATQUERY_GROUNDING_CHECKPOINT"] = str(weight)
    provider = GroundingDINOModel()
    provider._load()
    transform = transforms.Compose([
        transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def detections(image, expression):
        tensor, _ = transform(image, None)

        class CachedForward:
            output = None

            def to(self, device):
                return self

            def __call__(self, *args, **kwargs):
                if self.output is None:
                    self.output = provider._model(*args, **kwargs)
                return self.output

        cached = CachedForward()
        for box_threshold, text_threshold in GRID:
            with torch.inference_mode():
                boxes, scores, phrases = predict(cached, tensor, expression, box_threshold, text_threshold, device="cuda")
            items = []
            for (cx, cy, w, h), score, phrase in zip(boxes.tolist(), scores.tolist(), phrases):
                items.append({"box": [cx-w/2, cy-h/2, cx+w/2, cy+h/2], "score": score, "phrase": phrase})
            yield {"box_threshold": box_threshold, "text_threshold": text_threshold, "detections": items}
    return detections


def run(args):
    import torch
    random.seed(SEED)
    torch.manual_seed(SEED)
    require(args.stub or torch.cuda.is_available(), "CUDA required")
    require(digest(args.pack) == args.pack_sha256, "pack SHA-256")
    config = json.loads((REPO / "configs/model_artifacts.json").read_bytes())["providers"]["qwen2.5vl-3b"]
    if not args.stub:
        require(args.qwen_dir and args.qwen_hashes, "Qwen directory and checksum list required")
        hashes = json.loads(args.qwen_hashes.read_bytes())
        require(all(hashes.get(name) == sha for name, sha in config["weight_sha256"].items()), "Qwen shard pins")
        require(set(config["required_files"] + ["model.safetensors.index.json"]) <= hashes.keys(), "incomplete Qwen checksum list")
        files = {p.name for p in args.qwen_dir.iterdir() if p.is_file()}
        require(files == hashes.keys(), "Qwen file list mismatch")
        for name, sha in hashes.items():
            require(digest(args.qwen_dir / name) == sha, f"Qwen file mismatch: {name}")

    with zipfile.ZipFile(args.pack) as pack:
        items = pack_items(pack)
        if args.stub:
            items = [row for dataset, split, task, count in
                     (("dior-rsvg", "eval", "grounding", 5), ("vrsbench", "eval", "grounding", 5),
                      ("dior-rsvg", "select", "grounding", 3), ("vrsbench", "select", "grounding", 3),
                      ("vrsbench", "eval", "caption", 4))
                     for row in [r for r in items if (r["dataset"], r["split"], r["task"]) == (dataset, split, task)][:count]]
        ground = [r for r in items if r["split"] == "eval" and r["task"] == "grounding"]
        select = [r for r in items if r["split"] == "select"]
        captions = [r for r in items if r["task"] == "caption"]
        donors = shuffled(ground)
        args.output.mkdir()
        for model in ("qwen", "dino"):
            infer = None if args.stub else (qwen_backend(args.qwen_dir) if model == "qwen" else dino_backend())
            jobs = [("eval", ground), ("shuffled", ground), ("grey", ground)]
            jobs.append(("caption", captions) if model == "qwen" else ("select", select))
            for condition, rows in jobs:
                with (args.output / f"{model}-{condition}.jsonl").open("x") as output:
                    for index, row in enumerate(rows, 1):
                        image = Image.open(BytesIO(pack.read(row["image"]))).convert("RGB")
                        if condition == "grey":
                            image = Image.new("RGB", image.size, (128, 128, 128))
                        donor = donors[row["id"]] if condition == "shuffled" else row
                        expression = donor["expression"]
                        result = {"id": row["id"], "donor_id": donor["id"],
                                  "donor_same_wording": expression == row["expression"], "dataset": row["dataset"]}
                        if model == "qwen":
                            prompt = CAPTION if condition == "caption" else PROMPT.format(expression=expression)
                            raw = "[250,250,750,750]" if args.stub else infer(image, prompt, 256 if condition == "caption" else 64)
                            result["raw_text"] = raw
                            if condition != "caption":
                                result["box"] = parse_box(raw)
                        else:
                            result["grid"] = list(infer(image, expression)) if infer else [
                                {"box_threshold": b, "text_threshold": t, "detections": []} for b, t in GRID]
                        output.write(json.dumps(result, allow_nan=False) + "\n")
                        if index % 25 == 0 or index == len(rows):
                            print(f"{model}-{condition} {index}/{len(rows)}", flush=True)
            del infer
            if not args.stub:
                torch.cuda.empty_cache()
    info = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "seed": SEED, "stub": args.stub, "gpu": None if args.stub else torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "pack_sha256": args.pack_sha256, "qwen_revision": config["revision"],
            "dino_revision": DINO_REVISION, "versions": {name: version(name) for name in
                (("torch", "Pillow") if args.stub else ("torch", "Pillow", "transformers", "groundingdino-py", "huggingface-hub"))}}
    (args.output / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
    files = sorted(args.output.iterdir())
    require(sum(path.stat().st_size for path in files) < 200_000_000, "raw output exceeds 200 MB")
    (args.output / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path)
    parser.add_argument("--qwen-hashes", type=Path, help="Pinned JSON mapping filename to SHA-256")
    parser.add_argument("--stub", action="store_true", help="Exercise serialization with 20 selected inputs")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
