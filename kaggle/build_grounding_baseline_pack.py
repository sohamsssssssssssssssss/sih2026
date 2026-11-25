"""Build a private grounding ZIP from verified grounding-v1 sources."""
import argparse
from contextlib import ExitStack
from hashlib import sha256, file_digest
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

FROZEN = Path(__file__).resolve().parents[1] / "results/grounding"


def verify_train_separation(train_rows, held_out_rows, train_hashes, held_out_hashes):
    if ({row["parent"] for row in train_rows} & {row["parent"] for row in held_out_rows}
            or {row["image"] for row in train_rows} & {row["image"] for row in held_out_rows}
            or set(train_hashes) & set(held_out_hashes)):
        raise ValueError("HARD STOP: train image overlaps eval/select")


def train_preflight(root, sources):
    baseline = root / "satquery-grounding-baseline-pack.zip"
    expected = json.loads((FROZEN / "BASELINE-PACK.json").read_bytes())["sha256"]
    with baseline.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("HARD STOP: baseline pack SHA-256 mismatch")
    with zipfile.ZipFile(baseline) as pack:
        held_out = [json.loads(line) for line in pack.read("items.jsonl").splitlines()]
        held_out_hashes = [item["sha256"] for item in json.loads(pack.read("MANIFEST.json"))
                           if item["file"].startswith("images/")]
    members, train_rows = {}, []
    for dataset, archive, folder in (("dior-rsvg", "JPEGImages.zip", "JPEGImages"),
                                     ("vrsbench", "Images_train.zip", "Images_train")):
        for line in (FROZEN / f"{dataset}-train.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            path = f"images/{dataset}/{row['image']}"
            members[path] = sources[f"{dataset}/{archive}"], f"{folder}/{row['image']}"
            train_rows.append(dict(row, image=path))
    verify_train_separation(train_rows, held_out, (), ())
    size = sum(archive.getinfo(member).file_size for archive, member in members.values())
    print(f"Training archive image payload before zipping: {size:,} bytes across {len(members):,} images", flush=True)
    if size >= 15_000_000_000:
        raise ValueError("HARD STOP: train pack would exceed 15 GB")
    train_hashes = [sha256(archive.read(member)).hexdigest() for archive, member in members.values()]
    verify_train_separation(train_rows, held_out, train_hashes, held_out_hashes)


def build(root, output, *, train=False):
    frozen_manifest = (FROZEN / "MANIFEST.json").read_bytes()
    if train and sha256(frozen_manifest).hexdigest() != "6970604866b45d25849a256c432f11d968ff345e076e9b59c07fd131441337df":
        raise ValueError("HARD STOP: frozen manifest SHA-256 mismatch")
    manifest = json.loads(frozen_manifest)
    receipt = []
    written = set()
    with ExitStack() as stack:
        sources = {}
        for name, info in manifest["source_files"].items():
            prefix, filename = name.split("/")
            directory = root / ("dior-rsvg/xet" if prefix == "dior-rsvg" else "vrsbench")
            path = directory / filename
            with path.open("rb") as stream:
                if path.stat().st_size != info["bytes"] or file_digest(stream, "sha256").hexdigest() != info["sha256"]:
                    raise ValueError(f"Source mismatch: {name}")
            sources[name] = stack.enter_context(zipfile.ZipFile(path)) if path.suffix == ".zip" else path
        if train:
            for dataset in ("dior-rsvg", "vrsbench"):
                name = f"{dataset}-train.jsonl"
                if sha256((FROZEN / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
                    raise ValueError(f"HARD STOP: frozen train split mismatch: {name}")
            train_preflight(root, sources)
        destination = stack.enter_context(zipfile.ZipFile(output, "x"))

        def add(name, data):
            destination.writestr(name, data)
            written.add(name)
            receipt.append({"file": name, "bytes": len(data), "sha256": sha256(data).hexdigest()})

        add("README.md", b"Private academic research pack. DIOR-RSVG is CC BY-NC 4.0. VRSBench text annotations are CC BY 4.0 and DOTA-derived imagery is academic-use only. See results/grounding/DATA-CHECK.md for sources and terms.\n")
        add("frozen/MANIFEST.json", (FROZEN / "MANIFEST.json").read_bytes())
        annotations = {key: json.loads(path.read_bytes()) for key, path in sources.items()
                       if isinstance(path, Path) and path.suffix == ".json"}
        rows = []
        for dataset in ("dior-rsvg", "vrsbench"):
            for split in (("train",) if train else ("select", "eval")):
                name = f"{dataset}-{split}.jsonl"
                frozen = (FROZEN / name).read_bytes()
                if sha256(frozen).hexdigest() != manifest["files"][name]["sha256"]:
                    raise ValueError(f"Frozen manifest mismatch: {name}")
                add(f"frozen/{name}", frozen)
                for line in frozen.splitlines():
                    row = json.loads(line)
                    image = f"images/{dataset}/{row['image']}"
                    source = f"{dataset}/{row['source']}"
                    if dataset == "dior-rsvg":
                        obj = ET.fromstring(sources[source].read(row["member"])).findall("object")[row["object"]]
                        expression = obj.findtext("description") or obj.findtext("phrase")
                        box = obj.find("bndbox")
                        gold = [int(box.findtext(k)) / 800 for k in ("xmin", "ymin", "xmax", "ymax")]
                        archive, folder = "JPEGImages.zip", "JPEGImages"
                    else:
                        item = annotations[source][row["index"]]
                        archive, folder = ("Images_train.zip", "Images_train") if split in ("select", "train") else ("Images_val.zip", "Images_val")
                        expression = item.get("question")
                        gold = item.get("ground_truth")
                        if split in ("select", "train"):
                            prompt, gold = [turn["value"] for turn in item["conversations"]]
                            expression = re.search(r"<p>(.*?)</p>", prompt, re.S)[1] if row["task"] == "grounding" else None
                        if row["task"] == "grounding":
                            gold = [int(v) / 100 for v in re.findall(r"<(\d+)>", gold)]
                    if image not in written:
                        add(image, sources[f"{dataset}/{archive}"].read(f"{folder}/{row['image']}"))
                    rows.append(dict(row, dataset=dataset, split=split, image=image, expression=expression, gold=gold,
                                     id=f"{dataset}:{row['source']}:{row['index']}"))
        add("items.jsonl", "".join(json.dumps(row) + "\n" for row in rows).encode())
        destination.writestr("MANIFEST.json", json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    build(args.root, args.output, train=args.train)
