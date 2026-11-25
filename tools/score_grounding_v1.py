"""Score frozen grounding v1 predictions on CPU."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np

from kaggle.run_grounding_baselines import GRID, digest, open_pack, pack_items, parse_box, require

CENTER = [0.25, 0.25, 0.75, 0.75]
DATASETS = ("dior-rsvg", "vrsbench")
CONDITIONS = ("eval", "shuffled", "grey")


def valid(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def iou(box, gold):
    if not valid(box):
        return 0.0
    overlap = math.prod(max(0, min(box[i+2], gold[i+2]) - max(box[i], gold[i])) for i in (0, 1))
    return overlap / (math.prod(box[i+2]-box[i] for i in (0, 1))
                      + math.prod(gold[i+2]-gold[i] for i in (0, 1)) - overlap)


def prediction(row):
    raw = row.get("raw_text")
    box = parse_box(raw) if isinstance(raw, str) else None
    require(row.get("box") == box, f"parsed box mismatch for {row['id']}")
    return box, "valid" if box else ("invalid" if raw else "missing")


def dino(row, threshold):
    entries = row["grid"]
    keys = [(e["box_threshold"], e["text_threshold"]) for e in entries]
    require(tuple(keys) == GRID, f"DINO grid mismatch for {row['id']}")
    detections = entries[keys.index(threshold)]["detections"]
    if not detections:
        return None, "missing"
    require(all(math.isfinite(d["score"]) for d in detections), "nonfinite DINO confidence")
    box = max(detections, key=lambda d: d["score"])["box"]
    return box, "valid" if valid(box) else "invalid"


def select_threshold(rows, predictions):
    return max(GRID, key=lambda pair: (sum(iou(dino(predictions[r["id"]], pair)[0], r["gold"]) >= .5
                                          for r in rows), *pair))


def metrics(values):
    count = len(values)
    missing = sum(status == "missing" for _, status in values)
    invalid = sum(status == "invalid" for _, status in values)
    return {"n": count, "acc@0.5": sum(v >= .5 for v, _ in values)/count,
            "acc@0.25": sum(v >= .25 for v, _ in values)/count,
            "mean_iou": sum(v for v, _ in values)/count, "malformed_rate": (missing+invalid)/count,
            "missing": missing, "invalid": invalid}


def draws_for(sizes):
    rng = np.random.default_rng(17)
    return [rng.integers(n, size=(10000, n)) for n in sizes]


def paired(deltas, draws):
    point = float(np.mean([np.mean(delta) for delta in deltas]))
    sample = np.mean([np.asarray(delta)[indices].mean(axis=1)
                      for delta, indices in zip(deltas, draws, strict=True)], axis=0)
    lo, hi = np.percentile(sample, [2.5, 97.5], method="linear")
    return {"delta": point, "ci95": [float(lo), float(hi)]}


def control_pass(comparison):
    return comparison["delta"] > 0


def positive_ci(comparison):
    return control_pass(comparison) and comparison["ci95"][0] > 0


def read_predictions(directory, name, rows, stub):
    index = {}
    for line in (directory / f"{name}.jsonl").read_text().splitlines():
        row = json.loads(line)
        require(row["id"] not in index, f"duplicate prediction: {row['id']}")
        index[row["id"]] = row
    expected = {r["id"]: r for r in rows}
    require(index.keys() == expected.keys() or (stub and index.keys() <= expected.keys()),
            f"prediction coverage mismatch: {name}")
    require(all(row["dataset"] == expected[key]["dataset"] for key, row in index.items()), f"dataset mismatch: {name}")
    return index


def verify_output(directory, names, pack_sha):
    names = {f"{name}.jsonl" for name in names} | {"run-info.json"}
    seen = set()
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        require(name in names and name not in seen and digest(directory / name) == checksum, f"output checksum: {name}")
        seen.add(name)
    require(seen == names, "incomplete output checksums")
    info = json.loads((directory / "run-info.json").read_text())
    require(info["pack_sha256"] == pack_sha and info["seed"] == 17, "prediction provenance mismatch")
    return info["stub"]


def caption_scores(rows, predictions):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    require(version("pycocoevalcap") == "1.2" and version("pycocotools") == "2.0.8",
            "install the pinned caption metric packages")
    tokenizer = PTBTokenizer()
    gold = tokenizer.tokenize({r["id"]: [{"caption": r["gold"]}] for r in rows})
    generated = tokenizer.tokenize({r["id"]: [{"caption": predictions[r["id"]]["raw_text"]}] for r in rows})
    cider, values = Cider().compute_score(gold, generated)
    bleu, _ = Bleu(4).compute_score(gold, generated)
    meteor, _ = Meteor().compute_score(gold, generated)
    return {"CIDEr": float(cider), "BLEU-4": float(bleu[3]), "METEOR": float(meteor)}, values


def score(items, raw, pack_sha, adapter=None):
    baseline_names = [f"{model}-{condition}" for model in ("qwen", "dino") for condition in CONDITIONS]
    stub = verify_output(raw, baseline_names + ["dino-select", "qwen-caption"], pack_sha)
    select = [r for r in items if r["split"] == "select"]
    ground = [r for r in items if r["split"] == "eval" and r["task"] == "grounding"]
    captions = [r for r in items if r["task"] == "caption"]
    selected = read_predictions(raw, "dino-select", select, stub)
    threshold = select_threshold([r for r in select if r["id"] in selected], selected)
    inputs = {name: read_predictions(raw, name, ground, stub) for name in baseline_names}
    if adapter:
        require(not stub and not verify_output(adapter, [f"adapter-{c}" for c in CONDITIONS] + ["adapter-caption"], pack_sha),
                "adapter scoring requires real predictions")
        inputs.update({f"adapter-{c}": read_predictions(adapter, f"adapter-{c}", ground, False) for c in CONDITIONS})
    ground = [r for r in ground if r["id"] in inputs["qwen-eval"]]
    require(all(set(predictions) == {r["id"] for r in ground} for predictions in inputs.values()), "unaligned eval conditions")
    by_dataset = [[r for r in ground if r["dataset"] == dataset] for dataset in DATASETS]
    draws = draws_for([len(rows) for rows in by_dataset])
    summaries, hits = {}, {}
    for name in [*inputs, "centre"]:
        summaries[name], hits[name] = {}, []
        for dataset, rows in zip(DATASETS, by_dataset, strict=True):
            values = []
            for row in rows:
                if name == "centre":
                    box, status = CENTER, "valid"
                elif name.startswith("dino-"):
                    box, status = dino(inputs[name][row["id"]], threshold)
                else:
                    box, status = prediction(inputs[name][row["id"]])
                values.append((iou(box, row["gold"]), status))
            summaries[name][dataset] = metrics(values)
            hits[name].append(np.array([float(value >= .5) for value, _ in values]))
        keys = ("acc@0.5", "acc@0.25", "mean_iou", "malformed_rate")
        summaries[name]["equal_weight"] = {key: float(np.mean([summaries[name][d][key] for d in DATASETS])) for key in keys}

    def compare(left, right):
        return paired([a-b for a, b in zip(hits[left], hits[right], strict=True)], draws)

    controls = {}
    for model in ("qwen", "dino", "adapter") if adapter else ("qwen", "dino"):
        controls[model] = {c: compare(f"{model}-eval", f"{model}-{c}") for c in CONDITIONS[1:]}
        controls[model]["same_wording"] = sum(r["donor_same_wording"] for r in inputs[f"{model}-shuffled"].values())
    caption_input = read_predictions(raw, "qwen-caption", captions, stub)
    captions = [r for r in captions if r["id"] in caption_input]
    qwen_caption, base_values = caption_scores(captions, caption_input)
    result = {"stub": stub, "dino_threshold": threshold, "grounding": summaries, "controls": controls,
              "base_vs_dino": compare("qwen-eval", "dino-eval"), "captioning": {"qwen": qwen_caption}}
    if adapter:
        adapter_caption, values = caption_scores(captions, read_predictions(adapter, "adapter-caption", captions, False))
        delta = paired([values-base_values], draws_for([len(captions)]))
        comparisons = {model: compare("adapter-eval", f"{model}-eval") for model in ("qwen", "dino")}
        result["captioning"].update(adapter=adapter_caption, cider_gain=delta, passed=positive_ci(delta))
        result["adapter"] = {"comparisons": comparisons, "passed": all(map(positive_ci, comparisons.values()))
                             and all(control_pass(controls["adapter"][c]) for c in CONDITIONS[1:])}
    return result


def report(result):
    lines = ["# Grounding v1", ""]
    if result["stub"]:
        lines += ["Stub dry run. These values exercise the scoring pipeline only.", ""]
    elif "adapter" not in result:
        lines += ["Baseline measurements only. An adapter verdict requires adapter predictions.", ""]
    lines += [f"DINO thresholds selected on SELECT: {result['dino_threshold']}.", "",
              "| Condition | Dataset | Acc@0.5 | Acc@0.25 | Mean IoU | Malformed | Missing | Invalid |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, groups in result["grounding"].items():
        for dataset, row in groups.items():
            values = " | ".join(f"{row[key]:.4f}" for key in ("acc@0.5", "acc@0.25", "mean_iou", "malformed_rate"))
            lines.append(f"| {name} | {dataset} | {values} | {row.get('missing', '—')} | {row.get('invalid', '—')} |")
    lines += ["", "| Model | Control | Accuracy drop | Paired 95% CI | Same wording |",
              "|---|---|---:|---|---:|"]
    for model, controls in result["controls"].items():
        for condition in CONDITIONS[1:]:
            row = controls[condition]
            same = controls["same_wording"] if condition == "shuffled" else "—"
            lines.append(f"| {model} | {condition} | {row['delta']:.4f} | {row['ci95']} | {same} |")
    lines += ["", f"Base minus DINO: {result['base_vs_dino']}.", "",
              f"Caption metrics: {result['captioning']}."]
    if "adapter" in result:
        lines += ["", f"Grounding adapter decision and paired comparisons: {result['adapter']}."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with open_pack(args.pack, args.pack_sha256) as pack:
        result = score(pack_items(pack), args.raw, args.pack_sha256, args.adapter)
    args.output.mkdir()
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(report(result))


if __name__ == "__main__":
    main()
