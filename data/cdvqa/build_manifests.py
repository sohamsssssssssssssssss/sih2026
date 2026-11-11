"""Build CDVQA G0 manifests from the official QA JSON (CPU only, no images read).

Run: python3.11 data/cdvqa/build_manifests.py
Input: ~/data/cdvqa/qa/{Train,Val,Test,Test2}_{images,questions,answers}.json
Output: data/manifests/cdvqa/{train-8k.v1,eval-test1-400.v1,eval-test2-400.v1}.jsonl + provenance.json
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

RAW = Path.home() / "data/cdvqa/qa"
OUT = Path(__file__).resolve().parents[2] / "data/manifests/cdvqa"
SEED = 17
TYPES = ("change_or_not", "increase_or_not", "decrease_or_not", "smallest_change",
         "largest_change", "change_to_what", "change_ratio", "change_ratio_types")
TRAIN_PER_TYPE = 1000
EVAL_PAIRS = 100
EVAL_PER_TYPE = 50
# Unverified convention: SECOND stores before/after as im1/ and im2/ with the same file name.
BEFORE_DIR, AFTER_DIR = "im1", "im2"


def key(*parts):
    """Deterministic seed-17 sort key (sha256), independent of input order."""
    return hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()


def load(split):
    images = json.loads((RAW / f"{split}_images.json").read_text())["images"]
    questions = json.loads((RAW / f"{split}_questions.json").read_text())["questions"]
    answers = json.loads((RAW / f"{split}_answers.json").read_text())["answers"]
    assert all(image["id"] == i for i, image in enumerate(images)), "image ids are not list indices"
    answer_of = {}
    for item in answers:
        assert item["active"] and item["question_id"] not in answer_of
        answer_of[item["question_id"]] = item["answer"]
    rows = []
    for question in questions:
        assert question["active"] and question["type"] in TYPES
        image = images[question["img_id"]]
        assert question["id"] in image["questions_ids"], "question not listed under its image"
        assert question["id"] in answer_of, "question without answer"
        rows.append({"question_id": question["id"], "question_type": question["type"],
                     "question": question["question"], "answer": answer_of[question["id"]],
                     "pair_id": Path(image["file_name"]).stem, "file_name": image["file_name"]})
    return rows, len(images)


def pair_row(split_name, row):
    return {"id": f"cdvqa-{split_name}-{row['question_id']}", "split": split_name,
            "question_type": row["question_type"], "question": row["question"],
            "answer": row["answer"], "before_image": f"{BEFORE_DIR}/{row['file_name']}",
            "after_image": f"{AFTER_DIR}/{row['file_name']}", "pair_id": row["pair_id"],
            "question_id": row["question_id"]}


def select_train(rows):
    """Walk pairs in seed-17 hash order; keep a QA row while its type quota is open."""
    quota, chosen = Counter(), []
    for row in sorted(rows, key=lambda r: (key("train", r["pair_id"]), r["question_id"])):
        if quota[row["question_type"]] < TRAIN_PER_TYPE:
            quota[row["question_type"]] += 1
            chosen.append(row)
    assert all(quota[t] == TRAIN_PER_TYPE for t in TYPES), dict(quota)
    return chosen


def select_eval(rows, label):
    """Pick EVAL_PAIRS pairs by seed-17 hash, then EVAL_PER_TYPE QA per type inside them."""
    pairs = set(sorted({r["pair_id"] for r in rows}, key=lambda p: key(label, "pair", p))[:EVAL_PAIRS])
    pool = [r for r in rows if r["pair_id"] in pairs]
    chosen = []
    for t in TYPES:
        candidates = sorted((r for r in pool if r["question_type"] == t),
                            key=lambda r: key(label, r["question_id"]))
        assert len(candidates) >= EVAL_PER_TYPE, (label, t, len(candidates))
        chosen += candidates[:EVAL_PER_TYPE]
    return chosen


def select_train_v2(rows, cap=5):
    """1,000 QA per type spread over all train pairs, with at most `cap` QA per pair in total.

    Types are filled scarcest-first. For each type, the next QA comes from the pair with the fewest
    selected QA so far (ties by seed-17 hash) that still has an unused question of that type.
    """
    import heapq
    by_type_pair = {}
    for row in sorted(rows, key=lambda r: key("train-v2", r["question_id"])):
        by_type_pair.setdefault((row["question_type"], row["pair_id"]), []).append(row)
    availability = Counter(r["question_type"] for r in rows)
    pairs = sorted({r["pair_id"] for r in rows})
    count = Counter()
    chosen = []
    for t in sorted(TYPES, key=lambda t: (availability[t], t)):
        heap = [(count[p], key("train-v2", "pair", p), p) for p in pairs
                if by_type_pair.get((t, p))]
        heapq.heapify(heap)
        taken = 0
        while taken < TRAIN_PER_TYPE:
            if not heap:
                raise AssertionError(f"type {t}: only {taken} selectable QA under cap {cap}")
            c, k, p = heapq.heappop(heap)
            if c >= cap:
                continue
            pick = by_type_pair[(t, p)].pop(0)
            chosen.append(pick)
            count[p] += 1
            taken += 1
            if by_type_pair[(t, p)] and count[p] < cap:
                heapq.heappush(heap, (count[p], k, p))
    assert len(chosen) == TRAIN_PER_TYPE * len(TYPES)
    assert len({r["question_id"] for r in chosen}) == len(chosen)
    return chosen


def majority_answer(rows):
    """Most frequent casefolded answer per type over the given rows; ties go to the smaller string."""
    counts = {t: Counter() for t in TYPES}
    for r in rows:
        counts[r["question_type"]][r["answer"].strip().casefold()] += 1
    return {t: min(counts[t], key=lambda a: (-counts[t][a], a)) for t in TYPES}


def write_jsonl(path, rows):
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                   for r in sorted(rows, key=lambda r: r["id"]))
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def counts(rows):
    by_type = Counter(r["question_type"] for r in rows)
    return {"qa": len(rows), "pairs": len({r["pair_id"] for r in rows}),
            "per_type": {t: by_type[t] for t in TYPES}}


def main():
    train_rows, train_images = load("Train")
    test_rows, test_images = load("Test")
    test2_rows, test2_images = load("Test2")
    val_rows, val_images = load("Val")
    vocab = {t: sorted({r["answer"].strip().casefold() for r in train_rows if r["question_type"] == t})
             for t in TYPES}
    for name, rows in (("Val", val_rows), ("Test", test_rows), ("Test2", test2_rows)):
        unseen = {(r["question_type"], r["answer"].strip().casefold()) for r in rows
                  if r["answer"].strip().casefold() not in vocab[r["question_type"]]}
        assert not unseen, f"{name} answers outside Train vocabulary: {sorted(unseen)[:5]}"

    train = select_train(train_rows)
    train_v2 = select_train_v2(train_rows)
    test1 = select_eval(test_rows, "eval-test1")
    test2 = select_eval(test2_rows, "eval-test2")
    train_pairs = {r["pair_id"] for r in train}
    train_universe = {r["pair_id"] for r in train_rows}
    test_pairs = {r["pair_id"] for r in test_rows}
    eval_pairs = {r["pair_id"] for r in test1} | {r["pair_id"] for r in test2}
    assert not (train_pairs & eval_pairs), "train/eval pair leakage"
    assert eval_pairs <= test_pairs and train_pairs <= train_universe
    assert not (train_universe & test_pairs), "Train and Test share image pairs"
    v2_pairs = {r["pair_id"] for r in train_v2}
    assert len(train_v2) == TRAIN_PER_TYPE * len(TYPES) and len({r["question_id"] for r in train_v2}) == len(train_v2)
    assert not (v2_pairs & eval_pairs), "train-v2/eval image-pair leakage"
    assert v2_pairs <= train_universe
    v2_per_pair = Counter(r["pair_id"] for r in train_v2)

    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    manifests["train-8k.v1.jsonl"] = write_jsonl(OUT / "train-8k.v1.jsonl", [pair_row("train", r) for r in train])
    manifests["train-8k.v2.jsonl"] = write_jsonl(OUT / "train-8k.v2.jsonl", [pair_row("train", r) for r in train_v2])
    manifests["eval-test1-400.v1.jsonl"] = write_jsonl(OUT / "eval-test1-400.v1.jsonl", [pair_row("test1", r) for r in test1])
    manifests["eval-test2-400.v1.jsonl"] = write_jsonl(OUT / "eval-test2-400.v1.jsonl", [pair_row("test2", r) for r in test2])
    provenance = {
        "dataset": "CDVQA (Change Detection VQA), QA from YZHJessica/CDVQA (Apache-2.0)",
        "qa_sources": {f"{s}_{k}.json": f"https://raw.githubusercontent.com/YZHJessica/CDVQA/main/{s}_{k}.json"
                       for s in ("Train", "Val", "Test", "Test2") for k in ("images", "questions", "answers")},
        "qa_sha256": {line.split()[1]: line.split()[0]
                      for line in (RAW / "SHA256SUMS").read_text().splitlines() if line.strip()},
        "qa_download_dir": "~/data/cdvqa/qa (not in git)",
        "image_source": "SECOND dataset (Yang et al. 2021), Google Drive; NOT downloaded (manual click required)",
        "image_convention": {"before": f"{BEFORE_DIR}/<file_name>", "after": f"{AFTER_DIR}/<file_name>",
                             "status": "UNVERIFIED until the SECOND archive is extracted"},
        "pair_definition": "pair_id = file_name stem; one bi-temporal image pair per file_name",
        "split_pairs": {"Train": train_images, "Val": val_images, "Test": test_images, "Test2": test2_images},
        "answer_vocabulary_from": "full Train split, casefolded",
        "answer_vocabulary": vocab,
        "seed": SEED,
        "sampling": {
            "train-8k.v1": f"pairs ordered by sha256(seed|train|pair); rows ordered by sha256 pair then question_id; keep row while its type has < {TRAIN_PER_TYPE}",
            "eval-test1-400.v1": f"first {EVAL_PAIRS} pairs by sha256(seed|eval-test1|pair); {EVAL_PER_TYPE} QA per type by sha256(seed|eval-test1|question_id)",
            "eval-test2-400.v1": f"first {EVAL_PAIRS} pairs by sha256(seed|eval-test2|pair); {EVAL_PER_TYPE} QA per type by sha256(seed|eval-test2|question_id)",
        },
        "counts": {"train-8k.v2": counts([pair_row("train", r) for r in train_v2]),
                   "train-8k.v1": counts([pair_row("train", r) for r in train]),
                   "eval-test1-400.v1": counts([pair_row("test1", r) for r in test1]),
                   "eval-test2-400.v1": counts([pair_row("test2", r) for r in test2])},
        "train_v2": {"pairs_available": len(train_universe), "pairs_used": len(v2_pairs),
                     "max_qa_per_pair": max(v2_per_pair.values()), "min_qa_per_pair": min(v2_per_pair.values()),
                     "cap_per_pair": 5, "algorithm": "types filled scarcest-first; each QA taken from the pair with fewest selected QA (ties by sha256(seed|train-v2|pair)) that has an unused question of that type; per-pair cap 5; QA order by sha256(seed|train-v2|question_id)"},
        "majority_answer": majority_answer(train_rows),
        "answer_choices": {t: sorted({r["answer"] for r in train_rows if r["question_type"] == t}) for t in TYPES},
        "leakage_check": {"train_pairs_in_eval": 0, "train_pairs_in_test_universe": 0,
                          "note": "Test and Test2 share the same 968 image pairs; they are separate eval manifests"},
        "manifest_sha256": manifests,
        "builder": "data/cdvqa/build_manifests.py",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    for name, digest in manifests.items():
        print(f"{name} sha256={digest}")
    for name in ("train-8k.v2", "train-8k.v1", "eval-test1-400.v1", "eval-test2-400.v1"):
        print(name, provenance["counts"][name])


if __name__ == "__main__":
    main()
