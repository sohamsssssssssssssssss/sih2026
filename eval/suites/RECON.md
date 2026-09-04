# Suite reconnaissance

Per-suite acquisition facts: what splits exist, whether a suite is safe to
train on, what it is licensed under, and what is currently blocking it.

This is knowledge that costs a day to re-derive and five minutes to read.
Most of it was established by loading each candidate and reading its actual
metadata, not from memory — re-verify before trusting any row that matters.

Licences were read from HuggingFace repo metadata and the Zenodo record, on
**2026-09-02/03**. Licences change. Re-check before anything ships.

## Status table

| suite | wired up | splits | use | licence | status |
|---|---|---|---|---|---|
| `rsvqa` | yes | official train/val/**test** | eval | CC-BY-4.0 | **OK** — the Stage 0 baseline suite |
| `ladder` | yes | generated manifest | eval | LoveDA: academic, **non-commercial** | OK |
| `proxy` | yes | n/a | smoke only | n/a | placeholder samples, not a real suite |
| `vrsbench` | name only | train + val archives | **trainable** | CC-BY-4.0 | BLOCKED: needs a ~4 GB image zip |
| `cdvqa` | name only | unknown | unknown | unknown | **BLOCKED: not on the HF Hub at all** |
| `bentxt` | no | `split` column: train/val/test/**bench** | **eval-only** bench split | **CDLA-Permissive-1.0** | BLOCKED: 118 GB imagery, separate download |
| `hrvqa` | no | validation only (2k mirror) | eval | **CC-BY-NC-4.0** | loads, but non-commercial |
| `lrsvqa` | no | n/a (7z archive) | eval probe | **CC-BY-NC-4.0** | BLOCKED: 57 GB `.7z` |

"wired up" means present in `SUITE_NAMES`; `vrsbench` and `cdvqa` are named
there but currently return placeholder samples, not real data.

## The parts that bite

### rsvqa — the one we actually measure on

Acquired from **Zenodo record 6344334**, the official RSVQA-LR release, with a
hardcoded size + MD5 per file and zip-slip protection on extraction. See
`eval/suites/rsvqa.py`. 10,004 active test questions; `--full` uses all of
them. **CC-BY-4.0** (verified from the Zenodo record), so numbers from it are
safe to publish.

There are unofficial HuggingFace mirrors of RSVQA-LR (e.g. `dmarsili/RSVQA-LR-2k`).
**Do not switch to one.** They are 2,000-row subsets that publish only a
`validation` split — `datasets.get_dataset_split_names` returns exactly
`['validation']` — so they cannot reproduce the baseline, which is measured on
the 10,004-question official *test* split. The mirror is smaller and faster to
load, which is exactly why someone will be tempted.

### vrsbench — the only trainable non-BEN.txt suite here

Annotations are a flat JSON list (`VRSBench_EVAL_vqa.json`, 37,409 rows). Two
shape surprises: the answer key is **`ground_truth`**, not `answer`, and
**`image_id` is a filename** (`P0003_0002.png`), not pixels.

The pixels are in separate archives that are **not** downloaded automatically:
`Images_val.zip` is 3.98 GB and `Images_train.zip` is 8.36 GB. Because it ships
both, it is the one suite here besides BEN.txt that could be trained on rather
than only evaluated.

### lrsvqa — the independent high-resolution probe

7,333 QA pairs in `LRS_VQA_merged.jsonl`. Question is in **`text`**, answer in
`ground_truth`.

Two things to know before wiring it up:

1. Every image is a path into `LRS_VQA.7z.001`–`013` — **57 GB of 7z**, which
   `datasets` cannot read. It needs a manual download and extraction step that
   someone has to decide to fund.
2. The `text` field already ends with `"Answer the question using a single word
   or phrase."`. That will collide with whatever answer-format instruction the
   model wrapper appends, producing a doubled instruction.

Worth the trouble anyway: it is **not** derived from BEN.txt and is
ultra-high-resolution (4000×4000+), which makes it the only independent check
that Stage 1 gains are real rather than the model learning one corpus's
annotation style. **CC-BY-NC-4.0.**

### hrvqa — usable, but non-commercial

2,000 rows, same flat shape as the RSVQA mirrors, images decode to PIL.
Validation-only. **CC-BY-NC-4.0**: fine for the hackathon and the deck, but if
anything downstream ever ships commercially this suite has to come out.

### cdvqa — this is a track blocker, not a missing suite

No mirror on the HuggingFace Hub and no problem-statement-provided source. It
is named in `SUITE_NAMES` but has never resolved to real data. Bi-temporal
change VQA is an entire track; it is blocked on a dataset that does not
currently exist for us.

### bentxt — PS-provided, path unresolved

BigEarthNet.txt (arXiv 2603.29630): 464,044 co-registered Sentinel-1 + Sentinel-2
pairs, **9,553,962** annotations. Now resolved: the annotations are a 467 MB
parquet at HF `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`, public and not gated.
**Licence CDLA-Permissive-1.0, not CC-BY-4.0.**

It ships **no imagery**. The pixels are a separate 118 GB download from Zenodo
10891137 as monolithic `.tar.zst` with no per-patch fetch, which cannot be
subsampled on Kaggle's ~73 GB scratch. See `data/BEN_TXT_FACTS.md` on main for
the full reconnaissance.

Its manually-verified benchmark split is eval-only in the strongest sense:
**never subsampled, never trained on, by anyone.** The rest of the corpus is
the training data. 464k pairs will not fit Kaggle's ~73 GB scratch, so it needs
subsampling to ~50k stratified on annotation type, geography, LULC class and
season — random sampling drops the rare complex LULC classes, which are exactly
the ones the paper reports VLMs failing on.

## Licence summary

Two suites are **CC-BY-NC-4.0** — non-commercial: **`hrvqa`** and **`lrsvqa`**.
`rsvqa` and `vrsbench` are CC-BY-4.0; **`bentxt` is CDLA-Permissive-1.0**
(verified at both txt.bigearth.net and Zenodo 10891137 — an earlier draft of
this table said CC-BY-4.0, which was wrong),
and the `ladder` suite's source imagery (LoveDA) is academic/non-commercial.

Nothing here blocks hackathon use. It matters the moment anything ships.
