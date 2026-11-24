"""Freeze grounding v1 sample IDs from the publisher annotations (no inference)."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile


SEED = 17
DIOR_SPLITS = {"train": 26991, "val": 3829, "test": 7500}
VRS_TRAIN = "VRSBench_train.json"
VRS_REF = "VRSBench_EVAL_referring.json"
VRS_CAP = "VRSBench_EVAL_Cap.json"


def vrs_parent(image):
    match = re.fullmatch(r"(P[0-9]+|[0-9]{5})_[0-9]{4}\.png", image)
    if match is None:
        raise ValueError(f"Unexpected VRSBench image name {image!r}")
    parent = match[1]
    return f"dota-v2:{parent}" if parent.startswith("P") else f"dior:{parent}"


def score_key(salt, value):
    return sha256(f"{SEED}:{salt}:{value}".encode()).digest()


def choose(rows, count, salt, excluded=()):
    """Choose one annotation per parent, then sample parents without replacement."""
    by_parent = {}
    excluded = set(excluded)
    for row in rows:
        parent = row["parent"]
        if parent in excluded:
            continue
        rank = score_key(salt, f'{row["source"]}:{row["index"]}')
        if parent not in by_parent or rank < by_parent[parent][0]:
            by_parent[parent] = rank, row
    if len(by_parent) < count:
        raise ValueError(f"{salt} needs {count} parents, found {len(by_parent)}")
    ordered = sorted(by_parent, key=lambda parent: score_key(salt, parent))
    return [by_parent[parent][1] for parent in ordered[:count]]


def validate_disjoint(splits):
    owner = {}
    for split, rows in splits.items():
        for row in rows:
            parent = row["parent"]
            if owner.setdefault(parent, split) != split:
                raise ValueError(f"Image parent {parent} assigned to {owner[parent]} and {split}")


def load_dior(root):
    index_to_split = {}
    for split, expected in DIOR_SPLITS.items():
        indices = [int(value) for value in (root / f"{split}.txt").read_text().split()]
        if len(indices) != expected or len(set(indices)) != expected:
            raise ValueError(f"Unexpected number of DIOR-RSVG {split} indices")
        if index_to_split.keys() & set(indices):
            raise ValueError("DIOR-RSVG source lists repeat expression indices")
        index_to_split.update(dict.fromkeys(indices, split))
    rows = {split: [] for split in DIOR_SPLITS}
    index = 0
    with ZipFile(root / "Annotations.zip") as archive:
        names = sorted(n for n in archive.namelist() if n.startswith("Annotations/") and n.endswith(".xml"))
        for name in names:
            annotation = ElementTree.fromstring(archive.read(name))
            image = annotation.findtext("filename")
            if not image or not re.fullmatch(r"[0-9]{5}\.jpg", image):
                raise ValueError(f"Unexpected image name in {name}")
            for object_index, obj in enumerate(annotation.findall("object")):
                split = index_to_split[index]
                box = obj.find("bndbox")
                left, top, right, bottom = (int(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
                if not (0 <= left < right <= 800 and 0 <= top < bottom <= 800):
                    raise ValueError(f"Invalid DIOR box in {name} object {object_index}")
                if not (obj.findtext("description") or obj.findtext("phrase") or "").strip():
                    raise ValueError(f"Empty expression in {name} object {object_index}")
                rows[split].append({
                    "task": "grounding", "image": image, "parent": f"dior:{Path(image).stem}",
                    "source": "Annotations.zip", "member": name, "object": object_index,
                    "index": index, "upstream_split": split,
                })
                index += 1
    if len(names) != 17402 or index != len(index_to_split):
        raise ValueError("DIOR-RSVG archive and published split indices disagree")
    return rows


def load_vrs(root):
    def load(filename, expected):
        data = json.loads((root / filename).read_text())
        if len(data) != expected:
            raise ValueError(f"Unexpected {filename} row count")
        return data

    def make_row(task, image, name, index, split):
        return {"task": task, "image": image, "parent": vrs_parent(image),
                "source": name, "index": index, "upstream_split": split}

    def valid_box(text):
        values = re.fullmatch(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}", text)
        if not values:
            return False
        left, top, right, bottom = map(int, values.groups())
        return 0 <= left < right <= 100 and 0 <= top < bottom <= 100

    published_counts = Counter()
    omitted = Counter()
    train_ref, train_cap = [], []
    for index, row in enumerate(load(VRS_TRAIN, 142390)):
        prompt, response = (turn["value"] for turn in row["conversations"])
        task = prompt.split("\n", 1)[1].split("]", 1)[0][1:]
        published_counts[task] += 1
        if task == "refer":
            if valid_box(response) and re.search(r"<p>\s*\S.*?</p>", prompt, re.S):
                train_ref.append(make_row("grounding", row["image"], VRS_TRAIN, index, "train"))
            else:
                omitted["train_grounding"] += 1
        elif task == "caption":
            if not response.strip():
                raise ValueError(f"Empty VRSBench train caption at {index}")
            train_cap.append(make_row("caption", row["image"], VRS_TRAIN, index, "train"))
    if published_counts != {"vqa": 85813, "refer": 36313, "caption": 20264}:
        raise ValueError(f"VRSBench train task counts disagree: {published_counts}")

    eval_ref, eval_cap = [], []
    for index, row in enumerate(load(VRS_REF, 16159)):
        if row["question_id"] != index:
            raise ValueError("VRSBench referring question IDs changed")
        if valid_box(row["ground_truth"]) and row["question"].strip():
            eval_ref.append(make_row("grounding", row["image_id"], VRS_REF, index, "test"))
        else:
            omitted["eval_grounding"] += 1
    for index, row in enumerate(load(VRS_CAP, 9350)):
        if row["question_id"] != index or not row["ground_truth"].strip():
            raise ValueError("Invalid VRSBench test caption or question ID")
        eval_cap.append(make_row("caption", row["image_id"], VRS_CAP, index, "test"))
    return train_ref, train_cap, eval_ref, eval_cap, dict(omitted)


def parent_ids(rows):
    return {row["parent"] for row in rows}


def freeze(dior, vrs, images):
    aliases = canonical_parents(images)

    def canonicalize(rows):
        return [{**row, "parent": aliases[row["parent"]]} for row in rows]

    dior = {split: canonicalize(rows) for split, rows in dior.items()}
    train_ref, train_cap, eval_ref, eval_cap = map(canonicalize, vrs[:4])
    dior_test = parent_ids(dior["test"])
    vrs_test = parent_ids(eval_cap)
    upstream_test = dior_test | vrs_test

    dior_eval = choose(dior["test"], 1000, "dior-eval")
    vrs_eval = choose(eval_ref, 1000, "vrs-eval", parent_ids(dior_eval))
    dior_select = choose(dior["val"], 500, "dior-select", upstream_test)
    vrs_select = choose(train_ref, 500, "vrs-select", upstream_test | parent_ids(dior_select))

    reserved = upstream_test | parent_ids(dior_select) | parent_ids(vrs_select)
    dior_train = choose(dior["train"], 5000, "dior-train", reserved | parent_ids(dior["val"]))
    vrs_train = choose(train_ref, 2500, "vrs-train", reserved)
    vrs_train_images = {row["image"] for row in vrs_train}
    vrs_eval_images = {row["image"] for row in vrs_eval}
    vrs_train += choose(
        (row for row in train_cap if row["image"] in vrs_train_images), 2500, "vrs-caption-train",
    )
    vrs_cap_eval = [row for row in eval_cap if row["image"] in vrs_eval_images]
    if len(vrs_cap_eval) != 1000 or {row["image"] for row in vrs_cap_eval} != vrs_eval_images:
        raise ValueError("Expected one caption per VRSBench evaluation image")
    vrs_eval += vrs_cap_eval

    validate_disjoint({"train": dior_train + vrs_train, "select": dior_select + vrs_select,
                       "eval": dior_eval + vrs_eval})
    return {"dior-rsvg": {"train": dior_train, "select": dior_select, "eval": dior_eval},
            "vrsbench": {"train": vrs_train, "select": vrs_select, "eval": vrs_eval}}


def zip_images(path, suffix):
    with ZipFile(path) as archive:
        entries = [info for info in archive.infolist()
                   if re.fullmatch(rf"[A-Za-z0-9_]+\.{suffix}", Path(info.filename).name)
                   and "__MACOSX" not in Path(info.filename).parts]
    images = {Path(info.filename).name: info for info in entries}
    if len(entries) != len(images):
        raise ValueError(f"Duplicate image basenames in {path.name}")
    return images


def canonical_parents(images):
    """Group identical archive members and all crops that share a source image."""
    by_fingerprint = {}
    parent_ids = {}

    def root(parent):
        while parent_ids[parent] != parent:
            parent_ids[parent] = parent_ids[parent_ids[parent]]
            parent = parent_ids[parent]
        return parent

    for (dataset, _), members in images.items():
        for filename, info in members.items():
            parent = f"dior:{Path(filename).stem}" if dataset == "dior-rsvg" else vrs_parent(filename)
            parent_ids.setdefault(parent, parent)
            fingerprint = info.file_size, info.CRC
            other = by_fingerprint.setdefault(fingerprint, parent)
            left, right = sorted((root(parent), root(other)))
            parent_ids[right] = left

    return {parent: root(parent) for parent in parent_ids}


def verify_images(splits, images):
    fingerprints = {}
    for dataset, groups in splits.items():
        for split, rows in groups.items():
            for row in rows:
                key = (dataset, row["upstream_split"] if dataset == "vrsbench" else "all")
                image = images[key][row["image"]]
                fingerprint = image.file_size, image.CRC
                # Equal bytes must share both ZIP metadata fields. Fail closed on a collision.
                if fingerprints.setdefault(fingerprint, split) != split:
                    raise ValueError(f"Image data fingerprint shared across splits: {row['image']}")


def write_manifests(output, splits, omitted, sources):
    manifest = {"seed": SEED, "source_files": sources, "invalid_annotation_rows": omitted, "files": {}}
    for dataset, groups in splits.items():
        for split, rows in groups.items():
            filename = f"{dataset}-{split}.jsonl"
            content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
            (output / filename).write_bytes(content)
            manifest["files"][filename] = {
                "items": len(rows), "images": len({row["image"] for row in rows}),
                "parent_images": len(parent_ids(rows)),
                "tasks": dict(Counter(row["task"] for row in rows)),
                "sha256": sha256(content).hexdigest(),
            }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dior", type=Path, required=True)
    parser.add_argument("--vrsbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vrs = load_vrs(args.vrsbench)
    images = {
        ("dior-rsvg", "all"): zip_images(args.dior / "JPEGImages.zip", "jpg"),
        ("vrsbench", "train"): zip_images(args.vrsbench / "Images_train.zip", "png"),
        ("vrsbench", "test"): zip_images(args.vrsbench / "Images_val.zip", "png"),
    }
    if [len(entries) for entries in images.values()] != [17402, 20264, 9350]:
        raise ValueError("Image archive counts differ from published counts")
    splits = freeze(load_dior(args.dior), vrs, images)
    verify_images(splits, images)

    sources = {}
    for dataset, directory, names in (
        ("dior-rsvg", args.dior, ("Annotations.zip", "JPEGImages.zip", "train.txt", "val.txt", "test.txt")),
        ("vrsbench", args.vrsbench, (VRS_TRAIN, VRS_REF, VRS_CAP, "Images_train.zip", "Images_val.zip")),
    ):
        for name in names:
            path = directory / name
            with path.open("rb") as file:
                digest = file_digest(file, "sha256").hexdigest()
            sources[f"{dataset}/{name}"] = {"bytes": path.stat().st_size, "sha256": digest}
    write_manifests(args.output, splits, vrs[-1], sources)


if __name__ == "__main__":
    main()
