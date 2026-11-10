"""Freeze offline RGB VQA pilot manifests; never select test questions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.remote_sensing import load_manifest, sha256_file

TYPES = ("presence", "comp", "count", "rural_urban")
ALGORITHM = "image-balanced-sha256-v1"
ALGORITHM_DETAIL = (
    "At each draw, choose a feasible image with the fewest previously selected QA pairs "
    "across all types; break ties by SHA256 of seed|split|image|image_id. Within that "
    "image take the first feasible question ordered by SHA256 of "
    "seed|split|question|question_id. Feasible means unseen and below its type quota "
    "(validation only). Stop at the total limit or exhaustion. Hash ties use numeric IDs. "
    "Input order is irrelevant. No replacement; test questions are never read or selected."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_rows(rows, split, seed, limit, per_type=None, quotas=None):
    """Balance images globally, subject to remaining question/type availability."""
    if quotas is not None and (per_type is not None or set(quotas) != set(TYPES)
                               or any(type(n) is not int or n < 0 for n in quotas.values())
                               or sum(quotas.values()) != limit):
        raise ValueError("Exact per-type quotas must cover all types and sum to the limit")
    groups = defaultdict(list)
    def order(kind, identity):
        return hashlib.sha256(f"{seed}|{split}|{kind}|{identity}".encode()).digest(), identity
    for row in rows:
        if row["split"] != split or split not in ("train", "validation"):
            raise ValueError("Pilot selection permits only the requested train/validation split")
        groups[row["image_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: order("question", r["question_id"]))
    counts, type_counts, selected = Counter(), Counter(), []
    while len(selected) < limit:
        feasible = {}
        for image_id, group in groups.items():
            candidate = next((r for r in group if (quotas is None and per_type is None)
                              or type_counts[r["question_type"]] < (quotas[r["question_type"]]
                              if quotas is not None else per_type)), None)
            if candidate is not None:
                feasible[image_id] = candidate
        if not feasible:
            break
        image_id = min(feasible, key=lambda i: (counts[i], order("image", i)))
        row = feasible[image_id]
        groups[image_id].remove(row)
        selected.append(row)
        counts[image_id] += 1
        type_counts[row["question_type"]] += 1
    return selected


def read_split(metadata, split, exclusions):
    prefix = "val" if split == "validation" else split
    def read(kind):
        return json.loads((metadata / f"LR_split_{prefix}_{kind}.json").read_text())[kind]
    images = {int(r["id"]) for r in read("images") if r.get("active")}
    if not set(exclusions) <= images:
        raise ValueError("Exclusion IDs are absent from active source images")
    images -= set(exclusions)
    answers = {}
    for a in read("answers"):
        if a.get("active"):
            qid = int(a["question_id"])
            if qid in answers:
                raise ValueError("Duplicate active answer question ID")
            answers[qid] = str(a["answer"])
    rows, seen = [], set()
    for q in read("questions"):
        if not q.get("active"):
            continue
        qid = int(q["id"])
        if qid in seen:
            raise ValueError("Duplicate active question ID")
        seen.add(qid)
        image_id = int(q["img_id"])
        if image_id not in images or qid not in answers:
            continue
        if q["type"] not in TYPES:
            raise ValueError(f"Unknown question type: {q['type']}")
        rows.append({"image": f"{image_id}.tif", "image_id": image_id,
                     "question_id": qid, "question_type": q["type"],
                     "question": q["question"], "answer": answers[qid],
                     "instruction": q["question"], "response": answers[qid],
                     "dataset": "RSVQA-LR", "source": "RSVQA-LR, Zenodo record 6344334",
                     "split": split, "sample_id": f"{split}-{qid}"})
    return rows


def summary(rows):
    by_image = Counter(r["image_id"] for r in rows)
    return {"qa": len(rows), "images": len(by_image),
            "per_type": {t: sum(r["question_type"] == t for r in rows) for t in TYPES},
            "per_image": {str(i): n for i, n in sorted(by_image.items())}}


def build(metadata, images, policy_path, output):
    policy = json.loads(policy_path.read_text())
    if policy["test_selection_permitted"] or policy["excluded_image_ids"]["test"]:
        raise ValueError("Test must remain unchanged and excluded from pilot selection")
    sources = {p.name: sha256_file(p) for p in sorted(metadata.glob("LR_split_*.json"))}
    required = {f"LR_split_{s}_{k}.json" for s in ("train", "val", "test") for k in ("images", "questions", "answers")}
    if not required <= sources.keys():
        raise FileNotFoundError("Missing official split metadata")
    eligible = {s: read_split(metadata, s, policy["excluded_image_ids"][s]) for s in ("train", "validation")}
    train_ids = {r["image_id"] for r in eligible["train"]}
    val_ids = {r["image_id"] for r in eligible["validation"]}
    if train_ids & val_ids:
        raise ValueError("Train/validation image IDs overlap")
    all_qids = [r["question_id"] for rows in eligible.values() for r in rows]
    if len(set(all_qids)) != len(all_qids):
        raise ValueError("Train/validation QA IDs overlap")
    for s, rows in eligible.items():
        actual = summary(rows)
        if any(actual[k] != v for k, v in policy["expected_remaining"][s].items()):
            raise ValueError(f"Remaining counts differ for {s}: {actual['images']} images / {actual['qa']} QA")
    selected = {
        "train": select_rows(eligible["train"], "train", policy["seed"], policy["train_qa_limit"]),
        "validation": select_rows(eligible["validation"], "validation", policy["seed"],
                                  len(TYPES) * policy["validation_per_type_limit"], policy["validation_per_type_limit"])}
    if len(selected["train"]) != policy["train_qa_limit"]:
        raise ValueError("Insufficient training QA pairs")
    image_hashes = {f"{i}.tif": sha256_file(images / f"{i}.tif")
                    for i in sorted({r["image_id"] for rows in selected.values() for r in rows})}
    payloads = {f"{s}.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
                for s, rows in selected.items()}
    report = {"schema_version": 1, "policy": policy, "policy_sha256": sha256_file(policy_path),
              "algorithm": ALGORITHM, "algorithm_detail": ALGORITHM_DETAIL,
              "builder_sha256": sha256_file(Path(__file__)),
              "image_path_base": "Explicit image_root pointing to the canonical Images_LR directory",
              "image_hash_algorithm": "SHA256 of complete source TIFF bytes",
              "source_metadata_sha256": sources, "selected_image_sha256": image_hashes,
              "eligible": {s: summary(rows) for s, rows in eligible.items()},
              "selected": {s: summary(rows) for s, rows in selected.items()},
              "manifest_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in payloads.items()},
              "test_selected_qa": 0}
    report["manifest_identity"] = digest(report)
    if any(sha256_file(metadata / n) != h for n, h in sources.items()):
        raise ValueError("Source metadata changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (output / name).write_bytes(content)
    # Exercise the unchanged RGB loader, including cross-manifest separation.
    combined = output / "pilot.jsonl"
    combined.write_bytes(payloads["train.jsonl"] + payloads["validation.jsonl"])
    load_manifest(combined, images)
    report["manifest_sha256"]["pilot.jsonl"] = sha256_file(combined)
    report.pop("manifest_identity")
    report["manifest_identity"] = digest(report)
    (output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=ROOT / "data/manifests/rsvqa/pilot-exclusion-policy.v1.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.metadata_root, args.image_root, args.policy, args.output_dir)
    print(json.dumps({"manifest_identity": report["manifest_identity"], "selected": report["selected"]}))


if __name__ == "__main__":
    main()
