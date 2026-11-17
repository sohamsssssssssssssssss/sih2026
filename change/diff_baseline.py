"""Run the preregistered T18 experiment using local CDVQA inputs and saved G3 scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 17
REPS = 2000
TYPES = ("change_or_not", "increase_or_not", "decrease_or_not")
SPLITS = {"train": "Train", "val": "Val", "test1": "Test", "test2": "Test2"}
CONDITIONS = ("real", "identical", "swapped")
ARMS = ("majority", "question-only", "diff-baseline", "oracle-mask")
G3 = "89fc5bf"
G3_MANIFEST = {
    "test1": "4689eb8a45fecb6e0ca42f3c209d64425f3b9068091337469ece8da588e16993",
    "test2": "f79da8027377422e15af62edf4797c412c1ed61241d10b2ee55a685362314425",
}
ROOT = Path(__file__).resolve().parents[1]


def features(before, after):
    """Global and 4×4 RGB statistics in DN, with no resampling."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3 or min(before.shape[:2]) < 4:
        raise ValueError("Expected aligned RGB images of equal shape, at least 4×4")
    delta = after.astype(np.float32) - before
    absolute = np.abs(delta)
    changed = absolute > 20

    def stats(index):
        return np.concatenate([x[index].mean(axis=(0, 1)) for x in (absolute, changed, delta)])

    ys, xs = (np.linspace(0, size, 5, dtype=int) for size in delta.shape[:2])
    grid = [stats(np.s_[y0:y1, x0:x1]) for y0, y1 in zip(ys, ys[1:]) for x0, x1 in zip(xs, xs[1:])]
    return np.concatenate([stats(np.s_[:, :]), *grid])


def read_split(root, official_name):
    def read(kind):
        path = root / "qa" / f"{official_name}_{kind}.json"
        return json.loads(path.read_text())[kind]

    images = {i["id"]: i for i in read("images")}
    answers = {a["id"]: a for a in read("answers")}
    rows = {}
    for q in read("questions"):
        img = images[q["img_id"]]
        (answer_id,) = q["answers_ids"]
        answer = answers[answer_id]
        gold = answer["answer"].strip().casefold()
        name = img["file_name"]
        if (answer["question_id"] != q["id"] or q["id"] not in img["questions_ids"]
                or not gold or Path(name).name != name or not name.endswith(".png") or q["id"] in rows):
            raise ValueError(f"Invalid QA join in {official_name} question {q['id']}")
        rows[q["id"]] = (q["type"], name, q["question"], gold)
    return rows


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def mask_areas(path, shape):
    with Image.open(path) as im:
        if im.size != shape[::-1]:
            raise ValueError(f"Mask and image shapes disagree: {path}")
        colors = im.convert("RGB").getcolors(maxcolors=256)
    if colors is None:
        raise ValueError(f"Too many mask colors: {path}")
    return {color: n / np.prod(shape) for n, color in colors if color != (255, 255, 255)}


def make_features(image_root, filenames, include_masks):
    pixel, masks = {}, {}
    for name in sorted(filenames):
        before, after = (read_rgb(image_root / f"im{date}" / name) for date in (1, 2))
        real = features(before, after).reshape(-1, 9)
        swapped = real.copy()
        swapped[:, 6:] *= -1
        pixel[name] = np.stack([real.ravel(), np.zeros_like(real).ravel(), swapped.ravel()])
        if include_masks:
            masks[name] = tuple(mask_areas(image_root / f"label{date}" / name, before.shape[:2])
                                for date in (1, 2))
    return pixel, masks


def rows_of_type(rows, question_type):
    selected = [(qid, name, question, gold) for qid, (kind, name, question, gold) in rows.items()
                if kind == question_type]
    if any(gold not in ("yes", "no") for _, _, _, gold in selected):
        raise ValueError(f"Non-binary label for {question_type}")
    return selected


def matrix(rows, arm, condition, pixel, masks, palette):
    if arm == "question-only":
        return [row[2] for row in rows]
    index = CONDITIONS.index(condition)
    x = np.stack([pixel[row[1]][index] for row in rows])
    if arm == "oracle-mask":
        dates = ((0, 1), (0, 0), (1, 0))[index]
        areas = [[masks[row[1]][date].get(color, 0) for date in dates for color in palette] for row in rows]
        x = np.column_stack((x, areas))
    return x


def tune(train, val, y_train, y_val, arm):
    transform = TfidfVectorizer(ngram_range=(1, 2)) if arm == "question-only" else StandardScaler()
    x_train, x_val = transform.fit_transform(train), transform.transform(val)
    candidates = []
    for c in (.01, .1, 1., 10.):
        model = LogisticRegression(C=c, random_state=SEED, max_iter=1000).fit(x_train, y_train)
        candidates.append((model.score(x_val, y_val), model))
    score, model = max(candidates, key=lambda item: item[0])
    return transform, model, {"C": model.C, "val_accuracy": score}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_g3(split, official):
    manifest = (ROOT / f"data/manifests/cdvqa/eval-{split}-400.v1.jsonl").read_bytes()
    if sha(manifest) != G3_MANIFEST[split]:
        raise ValueError(f"{split}: G3 manifest hash mismatch")
    frozen = {r["question_id"]: r for r in map(json.loads, manifest.splitlines())}
    raw = subprocess.check_output(["git", "show", f"{G3}:results/cdvqa-g3/{split}-result.json"], cwd=ROOT)
    result = json.loads(raw)
    predictions = {}
    for condition in CONDITIONS:
        records = result["conditions"][f"adapter_{condition}"]["records"]
        by_id = {r["question_id"]: r for r in records}
        if len(records) != len(frozen) or by_id.keys() != frozen.keys():
            raise ValueError(f"{split}: G3 evaluation IDs do not match")
        for qid, r in by_id.items():
            f = frozen[qid]
            kind, filename, question, gold = official[qid]
            if ((kind, filename, question, gold) != (f["question_type"], f["pair_id"] + ".png", f["question"], f["answer"].strip().casefold())
                    or (r["sample_id"], r["question_type"], r["pair_id"], r["question"], r["expected"].strip().casefold())
                    != (f["id"], kind, f["pair_id"], question, gold)):
                raise ValueError(f"{split}: G3 record differs at question {qid}")
        predictions[condition] = {qid: r["prediction"].strip().casefold() for qid, r in by_id.items()}
    return predictions, {"manifest_sha256": sha(manifest), "adapter_result_sha256": sha(raw)}


def cell(rows, predictions, question_type):
    gold = np.array([row[3] for row in rows])
    names = [row[1] for row in rows]
    pairs, cluster = np.unique(names, return_inverse=True)
    support = Counter(gold)
    supported = min(support.get("yes", 0), support.get("no", 0)) >= 10
    draws = np.random.default_rng(SEED).integers(len(pairs), size=(REPS, len(pairs)))
    weights = np.ones(len(rows), dtype=bool)

    def metric(values, eligible=weights, enough=supported):
        n = int(eligible.sum())
        if not enough or not n:
            return {"value": None, "ci95": None, "n": n}
        hit = np.bincount(cluster, weights=values * eligible, minlength=len(pairs))
        count = np.bincount(cluster, weights=eligible, minlength=len(pairs))
        denom = count[draws].sum(axis=1)
        boot = hit[draws].sum(axis=1)[denom > 0] / denom[denom > 0]
        return {"value": float(hit.sum() / n), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": n}

    results = {}
    for arm, (real, identical, swapped) in predictions.items():
        correct = real == gold
        stats = {f"{condition}_accuracy": metric(pred == gold)
                 for condition, pred in zip(CONDITIONS, (real, identical, swapped))}
        if question_type == TYPES[0]:
            stats["identical_no"] = metric(identical == "no")
            stats["swap_change"] = metric(real != swapped)
        else:
            enough = supported and correct.sum() >= 10 and (correct & (gold == "yes")).sum() >= 10
            flip = ((real == "yes") & (swapped == "no")) | ((real == "no") & (swapped == "yes"))
            stats["swap_flip_correct"] = metric(flip, correct, enough)
            stats["original_correct_yes"] = int((correct & (gold == "yes")).sum())
        results[arm] = stats
    return {"n": len(rows), "pairs": len(pairs), "support": dict(support), "arms": results,
            "gain_vs_majority": metric((predictions["diff-baseline"][0] == gold).astype(int)
                                       - (predictions["majority"][0] == gold).astype(int))}


def evaluate(root):
    rows = {split: read_split(root, name) for split, name in SPLITS.items()}
    filenames = {split: {row[1] for row in data.values()} for split, data in rows.items()}
    test_names = filenames["test1"] | filenames["test2"]
    if filenames["train"] & filenames["val"] or (filenames["train"] | filenames["val"]) & test_names:
        raise ValueError("Train, validation and test pair IDs must be disjoint")
    image_root = root / "images" / "SECOND_train_set"
    train_names = filenames["train"] | filenames["val"]
    names = train_names | test_names
    mask_found = sum((image_root / f"label{date}" / name).is_file() for name in names for date in (1, 2))
    if mask_found not in (0, 2 * len(names)):
        raise ValueError("Incomplete oracle-mask inputs")
    arms = ARMS if mask_found else ARMS[:-1]
    g3 = {split: read_g3(split, rows[split]) for split in ("test1", "test2")}
    print(f"Extracting features for {len(names)} image pairs on CPU", flush=True)
    pixel, masks = make_features(image_root, names, bool(mask_found))
    palette = sorted({color for name in filenames["train"] for mask in masks.get(name, ()) for color in mask})
    if any(color not in palette for dates in masks.values() for mask in dates for color in mask):
        raise ValueError("A mask uses a color absent from train")
    out = {"seed": SEED, "bootstrap_reps": REPS, "sklearn": sklearn.__version__,
           "oracle_palette_rgb": palette, "tuning": {}, "test": {},
           "qa_sha256": {f"{name}_{kind}": sha((root / "qa" / f"{name}_{kind}.json").read_bytes())
                         for name in SPLITS.values() for kind in ("images", "questions", "answers")}}
    for split, (_, provenance) in g3.items():
        out["test"][split] = {"g3_source": provenance, "full": {}, "g3_subset": {}}
    for kind in TYPES:
        print(f"Fitting and evaluating {kind}", flush=True)
        train, val = (rows_of_type(rows[split], kind) for split in ("train", "val"))
        y_train, y_val = ([row[3] for row in group] for group in (train, val))
        counts = Counter(y_train)
        majority = max(("no", "yes"), key=counts.get)
        fitted = {}
        out["tuning"][kind] = {"train_majority": majority, "train_support": dict(counts)}
        for arm in arms[1:]:
            train_x, val_x = (matrix(group, arm, "real", pixel, masks, palette) for group in (train, val))
            transform, model, details = tune(train_x, val_x, y_train, y_val, arm)
            fitted[arm] = (transform, model)
            out["tuning"][kind][arm] = details
        for split, (adapter, _) in g3.items():
            test = rows_of_type(rows[split], kind)
            preds = {"majority": [np.full(len(test), majority)] * len(CONDITIONS)}
            for arm, (transform, model) in fitted.items():
                preds[arm] = [model.predict(transform.transform(matrix(test, arm, condition, pixel, masks, palette)))
                              for condition in CONDITIONS]
            subset = np.array([i for i, row in enumerate(test) if row[0] in adapter["real"]])
            matched = {arm: [p[subset] for p in ps] for arm, ps in preds.items()}
            matched["G3 adapter"] = [np.array([adapter[c][test[i][0]] for i in subset]) for c in CONDITIONS]
            result = out["test"][split]
            result["full"][kind] = cell(test, preds, kind)
            result["g3_subset"][kind] = cell([test[i] for i in subset], matched, kind)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
