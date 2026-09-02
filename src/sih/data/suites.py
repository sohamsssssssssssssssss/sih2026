"""Benchmark suite loaders.

VERIFY THE DATASET PATHS BELOW BEFORE RELYING ON THEM. RSVQA, VRSBench and
CDVQA all have multiple community mirrors on HuggingFace with different field
names. Run inspect_suite() first, look at the actual keys, then fix the
adapter here — not in a notebook.

Verification log (2026-09-02, `datasets==3.5.0`):

  rsvqa     dmarsili/RSVQA-LR-2k  validation  2000 rows  (question/answer/image)
  hrvqa     dmarsili/HRVQA-2k     validation  2000 rows  (question/answer/image)
  vrsbench  xiang709/VRSBench     JSON only, routed through ADAPTERS
  lrsvqa    ll-13/LRS-VQA         BLOCKED, images ship as a 57 GB 13-part .7z
  bentxt    unverified — PS-provided, distribution at txt.bigearth.net
  cdvqa     unverified — not on the HF Hub at all

SPLIT NAMES ARE NOT NEGOTIABLE. `datasets.get_dataset_split_names` reports
exactly ['validation'] for BOTH dmarsili mirrors — they are 2k eval-only
subsets and publish no `train` split. Do not "fix" these to `train`; the
split does not exist and load_dataset will raise. If you need RSVQA/HRVQA
training data, that is a different, larger mirror and needs its own
inspect_suite pass before it goes in here.
"""

from __future__ import annotations

SUITES = {
    # name -> (hf_path, split, question_key, answer_key, image_key)

    # PRIMARY. The PS-provided corpus (BigEarthNet.txt, arXiv 2603.29630,
    # CC-BY-4.0). Its manually-verified benchmark split is the measurement
    # surface for every track — NOBODY TRAINS ON IT, ever, and it is never
    # subsampled. Path unresolved: distribution is at txt.bigearth.net, and
    # the HF mirror (if one exists) has not been confirmed. Do not guess it.
    "bentxt":    ("<VERIFY-txt.bigearth.net>", "benchmark", "question", "answer", "image"),

    # Verified. CC-BY-4.0. validation-only 2k subset — see module docstring.
    "rsvqa":     ("dmarsili/RSVQA-LR-2k", "validation", "question", "answer", "image"),

    # Verified. CC-BY-NC — non-commercial, fine for the hackathon, flag it if
    # anything downstream ships. validation-only 2k subset.
    "hrvqa":     ("dmarsili/HRVQA-2k", "validation", "question", "answer", "image"),

    # Ships as flat JSON + a separate image zip, so it does not fit this
    # tuple at all — routed through ADAPTERS below. Entry kept here so the
    # registry lists every suite in one place.
    "vrsbench":  ("xiang709/VRSBench", "<ADAPTER>", "question", "ground_truth", "image_id"),

    # CC-BY-NC. Independent high-resolution probe: it is NOT part of the
    # BEN.txt corpus, so it is the one suite that can tell us whether Stage 1
    # gains are real or just fitting BEN.txt's annotation style. Ultra-high-res
    # (4000x4000+) makes it our domain-gap canary. Currently blocked — see
    # load_suite.
    "lrsvqa":    ("ll-13/LRS-VQA", "<BLOCKED-7z>", "text", "ground_truth", "image"),

    # A2's BLOCKER. Bi-temporal change VQA is A2's entire track and there is
    # no PS-provided source for it and no CDVQA mirror on the HF Hub. Until
    # this resolves, A2 has nothing to train or evaluate against.
    "cdvqa":     ("<VERIFY>", "test", "question", "answer", "image"),
}


def inspect_suite(hf_path: str, split: str = "test") -> None:
    """Print the first row so you can see the real field names."""
    from datasets import load_dataset
    ds = load_dataset(hf_path, split=split)
    print(f"{hf_path} / {split}: {len(ds)} rows")
    print("features:", ds.features)
    print("row 0:", {k: (type(v).__name__ if k == "image" else v)
                     for k, v in ds[0].items()})


# --- per-suite adapters -------------------------------------------------
# For datasets whose on-disk shape is not (question, answer, image). Each
# returns the same list[dict] as load_suite so callers stay uniform.

def _load_vrsbench(limit: int | None = None) -> list[dict]:
    """VRSBench ships annotations as a flat JSON list, images as a separate zip.

    Answer key is `ground_truth`, not `answer`. `image_id` is a FILENAME
    (e.g. 'P0003_0002.png') that resolves against Images_val.zip, which is
    3.98 GB and is NOT downloaded automatically — point VRSBENCH_IMAGE_DIR at
    an extracted copy.
    """
    import json
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    image_dir = os.environ.get("VRSBENCH_IMAGE_DIR")
    if not image_dir:
        raise NotImplementedError(
            "vrsbench needs its images extracted before it can run.\n"
            "  Annotations: xiang709/VRSBench :: VRSBench_EVAL_vqa.json (37409 rows, ~10 MB)\n"
            "  Images:      xiang709/VRSBench :: Images_val.zip (3.98 GB, NOT auto-downloaded)\n"
            "Then: export VRSBENCH_IMAGE_DIR=/path/to/extracted/Images_val"
        )
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"VRSBENCH_IMAGE_DIR={image_dir!r} is not a directory")

    ann = hf_hub_download("xiang709/VRSBench", "VRSBench_EVAL_vqa.json",
                          repo_type="dataset")
    rows = json.load(open(ann))
    if limit:
        rows = rows[:limit]

    out = []
    for r in rows:
        img = root / r["image_id"]
        if not img.exists():
            raise FileNotFoundError(
                f"VRSBench image {r['image_id']!r} not found under {root}. "
                "Is VRSBENCH_IMAGE_DIR pointing at the extracted Images_val?"
            )
        out.append({"image": str(img),
                    "question": r["question"],
                    "answer": str(r["ground_truth"]).strip()})
    return out


ADAPTERS = {
    "vrsbench": _load_vrsbench,
}


def load_suite(name: str, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    if name in ADAPTERS:
        return ADAPTERS[name](limit=limit)

    if name == "lrsvqa":
        raise NotImplementedError(
            "lrsvqa (ll-13/LRS-VQA) cannot be loaded through `datasets`.\n"
            "Its 7333 QA pairs are in LRS_VQA_merged.jsonl (question is `text`, "
            "answer is `ground_truth`), but every image is a path into "
            "LRS_VQA.7z.001-013 — 57 GB of 7z that `datasets` cannot read. "
            "It needs a manual download + 7z extraction step first. "
            "Decide that explicitly; do not auto-download it here."
        )

    if name not in SUITES:
        raise KeyError(f"Unknown suite {name!r}. Known: {list(SUITES)}")

    path, split, qk, ak, ik = SUITES[name]
    if path.startswith("<VERIFY"):
        raise NotImplementedError(
            f"Suite {name!r} has no verified HF path yet.\n"
            f"Run: python -c \"import sys; sys.path.insert(0,'src'); "
            f"from sih.data.suites import inspect_suite; inspect_suite('<path>')\"\n"
            f"then fill in SUITES[{name!r}] in src/sih/data/suites.py."
        )

    ds = load_dataset(path, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    return [{"image": r[ik], "question": r[qk], "answer": str(r[ak]).strip()}
            for r in ds]
