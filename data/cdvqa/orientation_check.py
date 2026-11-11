"""Temporal order check for SECOND: which label map (label1 or label2) is the later one.

Uses CDVQA increase/decrease answers (train split) against per-class pixel areas of the colour-coded
SECOND label maps. Writes results/cdvqa-g0/orientation-check.json.
"""
import itertools, json, re
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path.home() / "data/cdvqa"
IMG = Path.home() / "data/cdvqa/images/SECOND_train_set"
REPO = Path(__file__).resolve().parents[2]
COLOURS = {"white": (255, 255, 255), "grey": (128, 128, 128), "dark_green": (0, 128, 0),
           "dark_red": (128, 0, 0), "bright_green": (0, 255, 0), "blue": (0, 0, 255), "red": (255, 0, 0)}
NAMES = list(COLOURS)
PAT = re.compile(r"(?:areas|regions) of (.+?) (?:increased?|decreased?)\?")

qs = json.load(open(BASE / "qa/Train_questions.json"))["questions"]
ans = {x["question_id"]: x["answer"] for x in json.load(open(BASE / "qa/Train_answers.json"))["answers"]}
imgs = json.load(open(BASE / "qa/Train_images.json"))["images"]
rows = []
for x in qs:
    if x["type"] in ("increase_or_not", "decrease_or_not"):
        m = PAT.search(x["question"])
        assert m, x["question"]
        rows.append((imgs[x["img_id"]]["file_name"], m.group(1), x["type"] == "increase_or_not", ans[x["id"]] == "yes"))
classes = sorted({r[1] for r in rows})
files = sorted({r[0] for r in rows})
fi = {f: i for i, f in enumerate(files)}
cache = Path.home() / "data/cdvqa/orientation-counts.npz"
if cache.exists():
    z = np.load(cache, allow_pickle=True)
    assert list(z["files"]) == files
    C1, C2 = z["C1"], z["C2"]
else:
    def counts(path):
        a = np.array(Image.open(path).convert("RGB")).reshape(-1, 3)
        return [int(np.all(a == np.array(COLOURS[n]), axis=1).sum()) for n in NAMES]
    C1 = np.array([counts(IMG / "label1" / f) for f in files])
    C2 = np.array([counts(IMG / "label2" / f) for f in files])
    np.savez(cache, files=np.array(files), C1=C1, C2=C2)
print("questions", len(rows), "pairs", len(files), "classes", classes)
P = np.array([fi[r[0]] for r in rows]); K = np.array([classes.index(r[1]) for r in rows])
INC = np.array([r[2] for r in rows]); YES = np.array([r[3] for r in rows])
out = {}
for name, (early, late) in {"label2_is_later": (C1, C2), "label1_is_later": (C2, C1)}.items():
    best = (-1.0, None)
    for perm in itertools.permutations(range(len(NAMES)), len(classes)):
        k = np.array(perm)[K]
        lt, et = late[P, k], early[P, k]
        pred = np.where(INC, lt > et, lt < et)
        acc = float((pred == YES).mean())
        if acc > best[0]:
            best = (acc, {c: NAMES[perm[i]] for i, c in enumerate(classes)})
    out[name] = {"agreement": best[0], "mapping": best[1]}
    print(name, f"agreement={best[0]*100:.2f}%", best[1])
json.dump({"questions": len(rows), "pairs": len(files), "classes": classes, "result": out}, open(REPO / "results/cdvqa-g0/orientation-check.json", "w"), indent=2)
