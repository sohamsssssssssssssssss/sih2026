"""Build the frozen LoveDA questions and score matched ladder predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.ladder import CLASS_NAMES, QUESTIONS, RUNGS, label_answers

SEED = 26167
BOOT_SEED = 17
REPS = 2000
CLASSES = tuple(CLASS_NAMES.values())
ALIASES = {
    "buildings": "building", "roads": "road", "bare land": "barren",
    "woodland": "forest", "cropland": "agriculture", "farmland": "agriculture",
    "1": "yes", "0": "no",
}
TYPES = {
    **{f"presence:{name}": ("yes", "no") for name in CLASSES},
    "dominant": CLASSES,
    "domain": ("rural", "urban"),
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_answer(text, labels):
    text = text.strip().casefold().rstrip(".!?").strip()
    text = ALIASES.get(text, text)
    return text if text in labels else None


def tile_questions(tiles):
    """Balance each presence class by deterministic sampling without replacement."""
    questions = {tile["tile"]: [] for tile in tiles}
    for index, name in CLASS_NAMES.items():
        positive = [tile for tile in tiles if index in tile["classes"]]
        negative = [tile for tile in tiles if index not in tile["classes"]]
        count = min(len(positive), len(negative))
        for label, pool in (("yes", positive), ("no", negative)):
            ranked = sorted(pool, key=lambda tile: hashlib.sha256(
                f"{SEED}:{name}:{label}:{tile['tile']}".encode()).digest())
            for tile in ranked[:count] if count else ranked:
                questions[tile["tile"]].append({"type": f"presence:{name}", "gold": label})
    for tile in tiles:
        questions[tile["tile"]].extend([
            {"type": "dominant", "gold": tile["dominant"]},
            {"type": "domain", "gold": tile["domain"]},
        ])
    return questions


def build(v2, source):
    prior = json.loads(v2.read_text())["results"]
    identities = {}
    for row in prior:
        tile, _, _ = row["tile_id"].rpartition("_gsd")
        if row["gsd"] == RUNGS[0]:
            identities.setdefault(tile, {})[row["question"]] = row["expected_answer"]
    if len(identities) != 200 or len(prior) != 200 * len(RUNGS) * len(QUESTIONS):
        raise ValueError("Unexpected v2 tile or row count")
    tiles = []
    for tile, gold in identities.items():
        stem = tile.removeprefix("loveda_LoveDA_images_png_")
        if not stem.isdecimal():
            raise ValueError(f"Unexpected legacy tile ID: {tile}")
        domain = Path("Train/Rural")
        image = domain / "images_png" / f"{stem}.png"
        mask = domain / "masks_png" / f"{stem}.png"
        with Image.open(source / mask) as raw:
            labels = np.unique(np.asarray(raw)).tolist()
        if set(labels) - set(range(8)) or not set(labels).intersection(CLASS_NAMES):
            raise ValueError(f"Invalid LoveDA mask: {mask}")
        if label_answers(source / mask) != gold:
            raise ValueError(f"V2 mask answers differ: {tile}")
        tiles.append({
            "tile": tile, "image": image.as_posix(), "mask": mask.as_posix(),
            "image_sha256": sha(source / image), "mask_sha256": sha(source / mask),
            "classes": sorted(set(labels).intersection(CLASS_NAMES)),
            "dominant": gold[QUESTIONS[0]], "domain": domain.name.lower(),
        })
    questions = tile_questions(tiles)
    return [{**tile, "questions": questions[tile["tile"]]} for tile in tiles]


def prompt(task):
    if task.startswith("presence:"):
        question = f"Does this tile contain the LoveDA class {task.split(':')[1]}?"
    elif task == "dominant":
        question = "Which LoveDA class occupies the largest area of this tile?"
    else:
        question = "Is this tile rural or urban?"
    return question + " Reply with exactly one label: " + ", ".join(TYPES[task]) + "."


def ci(values):
    valid = values[np.isfinite(values)]
    return np.quantile(valid, [.025, .975]).tolist() if len(valid) else None


def metrics(records, tile_ids, draws, labels):
    # Accumulate by source tile so repeated questions on the same tile stay paired.
    tile_index = {tile: i for i, tile in enumerate(tile_ids)}
    totals = np.zeros((len(tile_ids), len(labels)), dtype=int)
    hits = np.zeros_like(totals)
    yes = np.zeros(len(tile_ids), dtype=int)
    for row in records:
        t, label = tile_index[row["tile"]], labels.index(row["gold"])
        totals[t, label] += 1
        hits[t, label] += row["answer"] == row["gold"]
        yes[t] += row["answer"] == "yes"
    sampled_total, sampled_hits = totals[draws].sum(axis=1), hits[draws].sum(axis=1)
    n = int(totals.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = sampled_hits.sum(axis=1) / sampled_total.sum(axis=1)
        balanced = (sampled_hits / sampled_total).mean(axis=1)
    point_ba = float((hits.sum(axis=0) / totals.sum(axis=0)).mean()) if totals.sum(axis=0).all() else None
    bal_ci = ci(balanced)
    yes_rate = float(yes.sum() / n) if labels == ("yes", "no") else None
    return ({
        "n": n, "accuracy": float(hits.sum() / n), "accuracy_ci95": ci(accuracy),
        "balanced_accuracy": point_ba, "balanced_accuracy_ci95": bal_ci,
        "yes_rate": yes_rate,
        "degenerate": (yes_rate > .85 or bool(bal_ci and bal_ci[0] <= .5 <= bal_ci[1]))
        if yes_rate is not None else None,
        "malformed": sum(row["answer"] is None for row in records),
    }, accuracy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("v2", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("manifest", type=Path)
    args = p.parse_args()
    with args.manifest.open("x") as handle:
        for tile in build(args.v2, args.source):
            handle.write(json.dumps(tile, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
