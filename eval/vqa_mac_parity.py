"""Run the frozen Mac VQA parity protocol using the separate MLX environment."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import time

import numpy as np
from PIL import Image

KINDS = ("presence", "comp", "count", "rural_urban")
MANIFEST_SHA = "b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c"
CUDA_SHA = "93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd"
ADAPTER_SHA = "7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def exact(text):
    return text.strip().casefold()


def valid(answer, kind):
    if kind == "count":
        return re.fullmatch(r"0|[1-9][0-9]*", answer) is not None
    return answer in (("rural", "urban") if kind == "rural_urban" else ("yes", "no"))


def make_model(mx, base, adapter):
    from mlx_vlm import load
    from mlx_vlm.trainer.peft.lora_layers import LoRALinear

    model, processor = load(base, trust_remote_code=False)
    if adapter is not None:
        weights = mx.load(str(adapter))
        used = set()
        for i, layer in enumerate(model.language_model.model.layers):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lora = LoRALinear.from_base(getattr(layer.self_attn, projection), r=16, scale=2.0)
                stem = f"base_model.model.model.layers.{i}.self_attn.{projection}.lora_"
                a, b = stem + "A.weight", stem + "B.weight"
                lora.lora_a, lora.lora_b = weights[a].T.astype(mx.float16), weights[b].T.astype(mx.float16)
                setattr(layer.self_attn, projection, lora)
                used.update((a, b))
        if used != set(weights):
            raise ValueError("B13 tensor coverage mismatch")
    model.eval()
    mx.eval(model.parameters())
    return model, processor


def predict(mx, model, processor, image_path, question, instruction):
    from mlx_vlm import generate

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path.as_uri()},
        {"type": "text", "text": question + " " + instruction},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((392, 392))
    inputs = processor(text=[prompt], images=[rgb], return_tensors="np")
    params = {k: mx.array(v) for k, v in inputs.items()}
    params["mask"] = params.pop("attention_mask")
    return generate(model, processor, prompt, temperature=0, max_tokens=32,
                    repetition_penalty=1.05, repetition_context_size=4096, **params).text


def memory(mx):
    rss_kib = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]))
    return {"mlx_bytes": mx.get_active_memory() + mx.get_cache_memory(), "rss_bytes": rss_kib * 1024}


def run(args):
    import mlx.core as mx

    manifest, cuda_file = args.manifest, args.cuda
    if sha(manifest) != MANIFEST_SHA or sha(cuda_file) != CUDA_SHA:
        raise ValueError("Frozen data or CUDA reference hash mismatch")
    if sha(args.adapter) != ADAPTER_SHA:
        raise ValueError("B13 hash mismatch")
    conversion = args.base / "MANIFEST"
    payload = {}
    for line in conversion.read_text().split("SHA-256 (conversion payload):\n", 1)[1].splitlines():
        digest, name = line.split(maxsplit=1)
        payload[name.removeprefix("./")] = digest
    actual = {p.name: sha(p) for p in args.base.iterdir() if p.is_file() and p != conversion}
    if actual != payload:
        raise ValueError("MLX conversion hash mismatch")

    references = json.loads(cuda_file.read_text())
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != 389 or len({r["sample_id"] for r in rows}) != 389:
        raise ValueError("Expected 389 unique rows")
    for name in {r["image"] for r in rows}:
        if sha(args.images / name) != references["selected_image_sha256"][name]:
            raise ValueError(f"Image hash mismatch: {name}")
    key = "adapter_real" if args.mode == "b13" else "base_real"
    cuda_rows = references["conditions"][key]["records"]
    if [(r["sample_id"], r["question"], r["image"], r["expected"]) for r in cuda_rows] != [
        (r["sample_id"], r["question"], r["image"], r["answer"]) for r in rows
    ]:
        raise ValueError("CUDA and manifest rows differ")

    mx.reset_peak_memory()
    start = time.perf_counter()
    model, processor = make_model(mx, args.base, args.adapter if args.mode == "b13" else None)
    load_s = time.perf_counter() - start
    instruction = references["prompt"]["answer_instruction"]
    out = []
    for row, ref in zip(rows, cuda_rows, strict=True):
        start = time.perf_counter()
        answer = predict(mx, model, processor, args.images / row["image"], row["question"], instruction)
        seconds = time.perf_counter() - start
        kind = row["question_type"]
        malformed = not valid(exact(answer), kind)
        out.append(dict(sample_id=row["sample_id"], image=row["image"], kind=kind,
                        answer=answer, gold=row["answer"], cuda=ref["prediction"],
                        correct=not malformed and exact(answer) == exact(row["answer"]),
                        cuda_correct=ref["correct"], malformed=malformed, cuda_malformed=ref["malformed"],
                        agreement=exact(answer) == exact(ref["prediction"]), seconds=seconds))
        if len(out) % 50 == 0:
            print(args.mode, len(out), flush=True)
    soak = None
    if args.mode == "b13":
        rng = random.Random(17)
        selected = []
        for kind, count in zip(KINDS, (8, 8, 7, 7)):
            group = [r for r in rows if r["question_type"] == kind]
            rng.shuffle(group)
            selected.extend(group[:count])
        rng.shuffle(selected)
        baseline = memory(mx)
        samples = []
        for row in selected:
            predict(mx, model, processor, args.images / row["image"], row["question"], instruction)
            samples.append(memory(mx))
        soak = dict(n=len(samples), crashes=0, baseline=baseline,
                    peak={k: max(baseline[k], *(s[k] for s in samples)) for k in baseline},
                    sample_ids=[r["sample_id"] for r in selected])
    times = [r["seconds"] for r in out]
    result = dict(mode=args.mode, rows=out, load_s=load_s, first_s=times[0],
                  median_s=float(np.median(times)), p95_s=float(np.quantile(times, .95)),
                  peak_mlx_bytes=mx.get_peak_memory(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  crashes=0, soak=soak, manifest_sha256=MANIFEST_SHA, cuda_sha256=CUDA_SHA,
                  conversion_manifest_sha256=sha(conversion),
                  gate_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def paired_ci(base, b13):
    images = sorted({r["image"] for r in base})
    cluster = {im: i for i, im in enumerate(images)}
    totals = np.zeros((len(images), len(KINDS)), dtype=int)
    delta = np.zeros_like(totals)
    for a, b in zip(base, b13, strict=True):
        if a["sample_id"] != b["sample_id"]:
            raise ValueError("Unpaired evaluation rows")
        i, k = cluster[a["image"]], KINDS.index(a["kind"])
        totals[i, k] += 1
        delta[i, k] += int(b["correct"]) - int(a["correct"])
    draws = np.random.default_rng(17).integers(len(images), size=(2000, len(images)))
    n, d = totals[draws].sum(axis=1), delta[draws].sum(axis=1)
    valid_draws = (n > 0).all(axis=1)
    return np.quantile((d[valid_draws] / n[valid_draws]).mean(axis=1), [.025, .975]).tolist()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("base", "b13"))
    for name in ("base", "adapter", "manifest", "cuda", "images", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
