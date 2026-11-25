"""Produce raw grounding-v1 predictions from a checksum-pinned private pack."""
import argparse
from contextlib import contextmanager
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

from kaggle.grounding_qwen_inputs import verify_qwen_directory

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


PACK_NAME = "satquery-grounding-baseline-pack"
MANIFEST_SHA = "0feb52a6f339d2c4404c6cc8022550593f569ae52de2b78a2c832c67a046f292"


def discover_pack(root, pack_name=PACK_NAME):
    archives, directories = [], []
    def fail(error):
        raise ValueError(f"HARD STOP: cannot scan input mounts: {error}") from error

    for directory, _, files in os.walk(root, followlinks=True, onerror=fail):
        path = Path(directory)
        if pack_name + ".zip" in files:
            archives.append(path / (pack_name + ".zip"))
        if "MANIFEST.json" in files and any(pack_name in part for part in path.relative_to(root).parts):
            # The pack also ships a frozen/MANIFEST.json. Only the pack root qualifies.
            if path.name != "frozen":
                directories.append(path)
    candidates = archives + directories
    require(len(candidates) == 1, f"attach exactly one baseline pack (found {len(archives)} ZIPs and {len(directories)} extracted packs)")
    return candidates[0]


class ExtractedPack:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root / name).read_bytes()

    def namelist(self):
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*")
                if not path.is_dir() or path.is_symlink()]


@contextmanager
def open_pack(path, zip_sha, manifest_sha=MANIFEST_SHA):
    try:
        if path.is_dir():
            pack = ExtractedPack(path)
            require(digest(path / "MANIFEST.json") == manifest_sha, "extracted MANIFEST.json SHA-256 mismatch")
            require(not any(p.is_symlink() for p in path.rglob("*")), "unexpected symlink in extracted pack")
            yield pack
        else:
            require(digest(path) == zip_sha, "pack ZIP SHA-256 mismatch")
            with zipfile.ZipFile(path) as pack:
                yield pack
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"HARD STOP: cannot read pack: {error}") from error


def verify_manifest(pack):
    manifest = json.loads(pack.read("MANIFEST.json"))
    names = [row["file"] for row in manifest]
    actual = pack.namelist()
    expected = set(names) | {"MANIFEST.json"}
    require(len(actual) == len(set(actual)) and len(names) == len(set(names))
            and set(actual) == expected,
            f"pack file list mismatch (missing={sorted(expected-set(actual))}, extra={sorted(set(actual)-expected)})")
    for row in manifest:
        data = pack.read(row["file"])
        require(len(data) == row["bytes"] and sha256(data).hexdigest() == row["sha256"], row["file"])


def pack_items(pack, splits=("select", "eval")):
    verify_manifest(pack)
    frozen = REPO / "results/grounding"
    require(pack.read("frozen/MANIFEST.json") == (frozen / "MANIFEST.json").read_bytes(), "frozen manifest mismatch")
    items = [json.loads(line) for line in pack.read("items.jsonl").splitlines()]
    for dataset, split in product(("dior-rsvg", "vrsbench"), splits):
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


def qwen_backend(path, adapter=None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
    ).eval().to("cuda")

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True).eval()

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
    path = args.pack if args.pack else discover_pack(args.input_root)
    config = json.loads((REPO / "configs/model_artifacts.json").read_bytes())["providers"]["qwen2.5vl-3b"]
    if not args.stub:
        require(args.qwen_dir and args.qwen_hashes, "Qwen directory and checksum list required")
        hashes = json.loads(args.qwen_hashes.read_bytes())
        require(all(hashes.get(name, {}).get("sha256") == sha for name, sha in config["weight_sha256"].items()), "Qwen shard pins")
        require(set(config["required_files"] + ["model.safetensors.index.json"]) <= hashes.keys(), "incomplete Qwen checksum list")
        verify_qwen_directory(args.qwen_dir, hashes)

    if args.adapter_eval and not args.stub:
        require(args.adapter_dir and args.adapter_receipt_sha256, "adapter directory and receipt pin required")
        from kaggle.train_grounding_adapter import verify_tree
        verify_tree(args.adapter_dir, args.adapter_receipt_sha256)
    with open_pack(path, args.pack_sha256, args.manifest_sha256) as pack:
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
        for model in (("adapter",) if args.adapter_eval else ("qwen", "dino")):
            infer = None if args.stub else (qwen_backend(args.qwen_dir, args.adapter_dir) if model != "dino" else dino_backend())
            jobs = [("eval", ground), ("shuffled", ground), ("grey", ground)]
            jobs.append(("caption", captions) if model != "dino" else ("select", select))
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
                        if model != "dino":
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
            "dino_revision": None if args.adapter_eval else DINO_REVISION, "adapter_receipt_sha256": args.adapter_receipt_sha256, "versions": {name: version(name) for name in
                (("torch", "Pillow") if args.stub else ("torch", "Pillow", "transformers", "peft", "huggingface-hub") if args.adapter_eval else ("torch", "Pillow", "transformers", "groundingdino-py", "huggingface-hub"))}}
    (args.output / "run-info.json").write_text(json.dumps(info, indent=2) + "\n")
    files = sorted(args.output.iterdir())
    require(sum(path.stat().st_size for path in files) < 200_000_000, "raw output exceeds 200 MB")
    (args.output / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack", type=Path)
    source.add_argument("--input-root", type=Path)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA, choices=[MANIFEST_SHA])
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path)
    parser.add_argument("--qwen-hashes", type=Path, help="Pinned JSON mapping filename to size and SHA-256")
    parser.add_argument("--adapter-eval", action="store_true")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--adapter-receipt-sha256")
    parser.add_argument("--stub", action="store_true", help="Exercise serialization with 20 selected inputs")
    args = parser.parse_args()
    require(not args.adapter_dir or args.adapter_eval, "--adapter-dir requires --adapter-eval")
    run(args)


if __name__ == "__main__":
    main()
