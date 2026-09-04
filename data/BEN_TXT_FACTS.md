# BigEarthNet.txt — established facts

Reconnaissance of the PS-provided corpus, **verified 2026-09-04** by querying
the actual 467 MB parquet, not the paper. Every number below was reproduced
from the file; none is quoted from the abstract.

Source of truth: `BigEarthNet.txt.parquet`, md5 `68628955917d75a280e605bf33e9cea5`,
kept at `~/data/bigearthnet_txt/` — deliberately outside the repo. Paper:
arXiv 2603.29630. Distribution: <https://txt.bigearth.net> →
HF `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (public, not gated).

Read this before planning any BEN.txt work. Several of these facts invalidate
assumptions we were carrying.

---

## a. It ships no imagery. This is the longest-lead item in the project.

BEN.txt is **467 MB of text annotations only** — one parquet, keyed to patch
names. There are no pixels in it.

The imagery is a separate download from **Zenodo record 10891137**:

| file | size |
|---|---|
| `BigEarthNet-S1.tar.zst` | 54.44 GB |
| `BigEarthNet-S2.tar.zst` | 63.25 GB |
| `Reference_Maps.tar.zst` | 0.28 GB |
| **total** | **117.98 GB** |

The archives are **monolithic `.tar.zst`. There is no per-patch fetch.** To
obtain a stratified 50 k subsample you must first transfer all 118 GB and
decompress it, then discard ~90%.

**This cannot be done on Kaggle.** Scratch is ~73 GB, which cannot hold even
the compressed pair. Subsampling needs a machine with ~250 GB free, with the
result uploaded to Kaggle afterwards as a Dataset.

Every BEN.txt-dependent task is blocked behind that transfer. Nothing else in
this document is as schedule-critical.

## b. Licence is CDLA-Permissive-1.0, for both text and imagery

Not CC-BY-4.0. Verified at both sources: the HF repo card
(`license: cdla-permissive-1.0`) and Zenodo 10891137
(`{"id": "cdla-permissive-1.0"}`).

CDLA-Permissive-1.0 is permissive and allows commercial use, so nothing is
blocked — but anywhere we state the licence, state it correctly.

## c. Shape

**9,553,962 rows × 13 columns, zero nulls in every column.**
**464,044 unique `patch_id`, 464,044 unique `s1_name`, and 464,044 unique
pairs — a strict 1:1 S2↔S1 mapping.**

```
type            rows        share
binary          3,625,160   37.94%
mcq             3,259,184   34.11%
bounding box    2,205,686   23.09%
captioning        463,932    4.86%
```

`category` has **11** values, not the 8 in the paper's figure. Captioning's
category is the **literal string `'None'`, not a null** — filtering with
`isna()` silently returns nothing.

Annotations per patch: min 1, median 21, mean 20.59, max 67.

## d. OPEN DECISION — caption leakage

Captions state **country, season and climate zone in prose**:

> "This satellite image, captured in Austria during summer, ... within the
> *cold, no dry season, warm summer* climate zone."

Those same three attributes are also MCQ categories, totalling **1,390,573
rows** (`country` 463,274 + `season` 463,662 + `climate zone` 463,637).

Training Stage 1 on captions and then evaluating those MCQ categories scores
the model on facts we taught it. The number would be real and meaningless.

**This is not yet decided.** Two defensible options:

1. **Exclude** the three categories from evaluation entirely. Clean, and
   costs 14.6% of the annotation pool.
2. **Report separately** as known-contaminated, alongside the uncontaminated
   remainder.

Either is fine. Silently reporting the aggregate is not. Whoever owns Stage 1
must choose before any Stage 1 number is quoted.

## e. NO SAR SUPERVISION — this corpus cannot measure fusion gain

Searched all 9,553,962 rows, both the `input` and `output` columns:

```
SAR  Sentinel-1  radar  backscatter  VV  VH  polari…  Sentinel-2  optical  multispectral
  0           0      0            0   0   0        0           0        0              0
```

**Zero occurrences of every term.** The text says "the image" or "satellite
image" and never names a sensor. No column distinguishes S2-only from S1+S2
either; `s1_name` is populated on every row and the mapping is 1:1, so every
annotation applies to the pair.

The annotations were written from land cover, not from SAR-visible structure.
**A fusion model can therefore score identically to an S2-only model here.**
Any fusion gain we report on BEN.txt is unattributable.

A3 needs one of:
- a corpus where SAR carries unique signal (change detection, flood extent,
  night/cloud scenes), or
- an explicit, up-front statement that fusion is demonstrated **architecturally
  and not validated empirically**.

The second is defensible if we say it plainly. It is indefensible if a judge
finds it for us.

## f. The benchmark split is a column value, not a file

`split == 'bench'`. **1,082 patches, 15,029 annotations.** It does not ship
separately, so it is **one missing filter away from being trained on**. It is
our only ISRO-like proxy; contaminating it is unrecoverable.

It is also **shaped differently** from the other splits:

| split | annotations/patch |
|---|---|
| train / validation / test | min 14, max 39 (tight) |
| **bench** | **min 1, median 13, max 67** |

Manual verification kept whatever survived review rather than a fixed quota.
**Aggregate bench metrics per ANNOTATION, never per patch** — per-patch means
weight a 1-annotation patch equally with a 67-annotation one.

Splits are **disjoint by patch** (verified: no `patch_id` appears in two
splits), so there is no cross-split leakage to guard against.

## g. Stratification: one axis is not viable

Measured over unique patches, which is what a 50 k image subsample draws from:

| key | cardinality | imbalance | smallest cell |
|---|---|---|---|
| `season` | 4 | 2.2× | Winter, 72,697 |
| `country` | 10 | 96.4× | Kosovo, 1,571 |
| `climate_zone` | 10 | **27,959×** | **6 patches** |

- **`season` — viable.** Near-balanced, use directly.
- **`country` — viable with a floor.** Kosovo yields ~170 patches
  proportionally, which is thin but workable.
- **`climate_zone` — NOT viable proportionally.** Cells of 6 / 46 / 627 / 639
  patches. A 50 k draw is 10.8% of 464,044, so the smallest cell contributes
  **0.65 patches** — it disappears.

For climate zone, either oversample the four rare zones or collapse them into
a coarser grouping. **Whichever is chosen, document the distortion**, because
the resulting distribution no longer matches the corpus and any per-zone
number inherits that.

`latitude` and `longitude` are present on every row if finer spatial
stratification is wanted than 10 countries.

## h. Stage 1's caption corpus is 463,932 rows, not 9.55M

One caption per patch. Any compute, time, or cost estimate built on the 9.55M
figure is wrong by ~20× and needs redoing.

The 9.55M total is dominated by binary and MCQ, which are Stage 2 material.

## i. Parser specifications

Specs, not code — no loader is written yet, because the imagery is blocked and
an untested loader is worse than none.

**Bounding box** — `output` field. Validated against **200,000 sampled rows,
100.00% conformance, zero exceptions**:

```
'[x1 y1, x2 y2]'      e.g. '[0.64 0.0, 1.0 0.71]'
```
- **Normalised [0,1]** — all coordinates within range, 100%.
- **xyxy, x-first.** `x2 >= x1` and `y2 >= y1` hold in 100% of rows.
- Two decimal places; **space** between x and y, **comma-space** between corners.
- Not JSON. `json.loads` fails on it.
- The referring expression is inline markup in `input`: `<ref>…</ref>`.
- A second sub-type prompts with a point: `<point>(0.82, 0.28)</point>`.
  `bounding box` splits `point` 1,143,883 / `reference` 1,061,803.

**MCQ** — options are **inline in the `input` string**, lettered `a) b) c) d)`:

```
'Which classes share a boundary? a) Broad-leaved forest and Pastures,
 b) Coastal wetlands and Coniferous forest, c) Coniferous forest and Mixed
 forest, d) Arable land and Pastures'                          -> 'd'
```
- **Commas occur inside option text, so splitting on `,` is wrong.**
  Split on the pattern `\b[a-d]\)` instead.
- No `Options:` marker, no newline, no other delimiter.
- Output is a **bare letter**, distribution near-uniform
  (a 816,500 / d 814,648 / c 814,341 / b 813,695) — no position bias.

**Binary** — output is **literally lowercase `yes` / `no`**, only two distinct
values in 3,625,160 rows, split 51/49 (`no` 1,855,449 / `yes` 1,769,711).
Near-balanced, so a degenerate always-yes model scores ~49% here rather than
the ~60% it would score on RSVQA.

---

## Reproducing these numbers

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('~/data/bigearthnet_txt/BigEarthNet.txt.parquet')
print(len(df), df.patch_id.nunique(), df.isna().sum().sum())
print(df.type.value_counts())
"
```
