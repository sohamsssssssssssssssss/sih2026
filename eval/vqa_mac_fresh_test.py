"""Freeze an official RSVQA-LR test sample and run the registered MLX evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import threading
import time
import zipfile

import numpy as np
from PIL import Image

from eval.suites.rsvqa import FILES, _md5
from eval.vqa_mac_headroom import app_up, available_bytes, soak_rows
from eval.vqa_mac_parity import ADAPTER_SHA, KINDS, exact, make_model, memory, predict, sha, valid

CONVERSION_SHA = "e4ed41344ad53c218047253edf61d0e60a2c3d151651977ef8aff18094c99e6f"
INSTRUCTION = "Answer with a single word or number only. No explanation."
DATA = Path("data/manifests/rsvqa/mac-fresh-test")


def read_split(root, split):
    def load(key):
        return json.loads((root / f"LR_split_{split}_{key}.json").read_text())[key]

    images = {item["id"] for item in load("images") if item["active"]}
    answers = {item["question_id"]: exact(str(item["answer"])) for item in load("answers") if item["active"]}
    rows = []
    for q in load("questions"):
        if q["active"] and q.get("type") in KINDS and q["img_id"] in images and q["id"] in answers:
            answer = answers[q["id"]]
            if not valid(answer, q["type"]):
                raise ValueError(f"Invalid official gold answer at {split}/{q['id']}: {answer!r}")
            rows.append(dict(id=q["id"], cluster=q["img_id"], image=f"{q['img_id']}.tif",
                             question=q["question"], question_type=q["type"], answer=answer))
    return rows


def freeze(root, out):
    files = [f"LR_split_{split}_{key}.json" for split in ("train", "val", "test")
             for key in ("questions", "answers", "images")] + ["Images_LR.zip"]
    hashes = {}
    for name in files:
        path = root / name
        if (path.stat().st_size, _md5(path)) != FILES[name]:
            raise ValueError(f"Official archive checksum mismatch: {name}")
        hashes[name] = sha(path)
    train, val, test = (read_split(root, split) for split in ("train", "val", "test"))
    clusters = sorted({r["cluster"] for r in test})
    if len(clusters) != 100 or set(clusters) & {r["cluster"] for r in train + val}:
        raise ValueError("Expected 100 test-only image clusters")
    group = defaultdict(list)
    for row in test:
        group[row["cluster"], row["question_type"]].append(row)
    rng = random.Random(17)
    selected = [rng.choice(sorted(group[cluster, kind], key=lambda r: r["id"]))
                for cluster in clusters for kind in KINDS]
    majority = {}
    for kind in KINDS:
        counts = Counter(r["answer"] for r in train if r["question_type"] == kind)
        answer = min(counts, key=lambda label: (-counts[label], label))
        majority[kind] = {"answer": answer, "count": counts[answer], "n": sum(counts.values())}
    image_hashes = {}
    with zipfile.ZipFile(root / "Images_LR.zip") as archive:
        for name in sorted({r["image"] for r in selected}):
            image_hashes[name] = hashlib.sha256(archive.read(f"Images_LR/{name}")).hexdigest()
            if sha(root / "Images_LR" / name) != image_hashes[name]:
                raise ValueError(f"Extracted image mismatch: {name}")
    with (out / "questions.jsonl").open("x") as f:
        for row in selected:
            f.write(json.dumps({**row, "image_sha256": image_hashes[row["image"]]}, sort_keys=True) + "\n")
    with (out / "sources.json").open("x") as f:
        json.dump({"zenodo_record": "6344334", "sha256": hashes, "train_majority": majority}, f, indent=2)
        f.write("\n")


def snapshot():
    vm = subprocess.check_output(["vm_stat"], text=True)
    return {"vm_stat": vm, "available_bytes": available_bytes(vm),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def run(args):
    import mlx.core as mx

    manifest = DATA / "questions.jsonl"
    gate = Path("docs/gates/VQA-MAC-FRESH-TEST.md")
    if sha(manifest) not in gate.read_text() or sha(DATA / "sources.json") not in gate.read_text():
        raise ValueError("Frozen inputs differ from the gate")
    if sha(args.adapter) != ADAPTER_SHA or sha(args.base / "MANIFEST") != CONVERSION_SHA:
        raise ValueError("Weights differ from the pinned identity")
    for line in (args.base / "MANIFEST").read_text().split("SHA-256 (conversion payload):\n", 1)[1].splitlines():
        digest, filename = line.split(maxsplit=1)
        if sha(args.base / filename) != digest:
            raise ValueError(f"Conversion member hash mismatch: {filename}")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    images = {r["image"]: args.images / r["image"] for r in rows}
    grey = args.out.parent / "grey"
    grey.mkdir(exist_ok=True)
    for row in rows:
        path = images[row["image"]]
        if sha(path) != row["image_sha256"]:
            raise ValueError(f"Test asset has changed: {path}")
        if args.mode == "constant":
            with Image.open(path) as original:
                output = grey / f"{original.width}x{original.height}.png"
                if not output.exists():
                    Image.new("RGB", original.size, (127, 127, 127)).save(output)
            row["grey"] = output
    if args.mode == "soak" and not app_up():
        raise RuntimeError("Demo servers are not reachable")
    before_load = snapshot()
    mx.reset_peak_memory()
    start = time.perf_counter()
    model, processor = make_model(mx, args.base, None if args.mode == "base" else args.adapter)
    load_s = time.perf_counter() - start
    system_samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            system_samples.append(snapshot())
            stop.wait(.25)

    soak = args.mode == "soak"
    baseline = memory(mx) if soak else None
    sampler = threading.Thread(target=poll, daemon=True)
    if soak:
        sampler.start()
    records, mem, crashes = [], [], []
    try:
        for row in soak_rows(rows) if soak else rows:
            start = time.perf_counter()
            try:
                answer = predict(mx, model, processor, row.get("grey", images[row["image"]]),
                                 row["question"], INSTRUCTION)
            except Exception as error:
                crashes.append({"id": row["id"], "error": repr(error)})
                break
            elapsed = time.perf_counter() - start
            malformed = not valid(exact(answer), row["question_type"])
            records.append(dict(id=row["id"], answer=answer, malformed=malformed,
                                correct=not malformed and exact(answer) == row["answer"], seconds=elapsed))
            if soak:
                mem.append(memory(mx))
            if len(records) % 100 == 0:
                print(args.mode, len(records), flush=True)
    finally:
        if soak:
            stop.set()
            sampler.join()
    with args.out.open("x") as handle:
        json.dump(dict(mode=args.mode, load_s=load_s, rows=records, crashes=crashes,
                       peak_mlx_bytes=mx.get_peak_memory(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       baseline=baseline, soak_memory=mem, system_before_load=before_load,
                       system_during_soak=system_samples, app_up_after=app_up() if soak else None,
                       conversion_manifest_sha256=CONVERSION_SHA, adapter_sha256=ADAPTER_SHA,
                       manifest_sha256=sha(manifest), sources_sha256=sha(DATA / "sources.json"),
                       gate_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()),
                  handle, indent=2)
        handle.write("\n")


def bootstrap(matrix, rows):
    # One question per type per image. Resample whole clusters and keep draws paired.
    clusters = sorted({r["cluster"] for r in rows})
    indices = {(r["cluster"], r["question_type"]): i for i, r in enumerate(rows)}
    order = [[indices[c, k] for k in KINDS] for c in clusters]
    draws = np.random.default_rng(17).integers(len(clusters), size=(2000, len(clusters)))
    return {name: np.asarray(values)[order][draws].mean(axis=(1, 2))
            for name, values in matrix.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("freeze")
    build.add_argument("source", type=Path)
    build.add_argument("out", type=Path)
    for choice in ("base", "b13", "constant", "soak"):
        runner = sub.add_parser(choice)
        for name in ("base", "adapter", "images", "out"):
            runner.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.mode == "freeze":
        freeze(args.source, args.out)
    else:
        run(args)
