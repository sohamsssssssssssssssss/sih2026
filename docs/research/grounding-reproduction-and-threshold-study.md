# Grounding DINO on DIOR-RSVG — reproduction, threshold/prompt study, failure analysis

Track: grounding (brief Days 10–13). Experiment IDs `SQ-20260926-200`…`211`
(`eval/registry/`). Harness: `eval/grounding_research.py` (research-only; the
production provider stays CUDA-only and unmodified). Consolidated numbers:
`eval/results/grounding/grounding-study-v1.json`.

## 0. Pre-registered selection rule (committed in `a893ea5`, before any validation run)

Threshold and prompt choices are **selected on validation, never on the
400-expression T-quick test subset** (Protocol 01 §2, §9).

- **Selection set (`val-select`)**: 400 expressions, 20 per DIOR class, drawn
  with `suite.stratified_sample(seed=26167)` from the official `val.txt`,
  restricted to images that do **not** occur anywhere in the official
  `test.txt` (the official splits share images; see §1). It shares zero
  images with T-quick.
- **Candidates**: prompt ∈ {`expression`, `head`, `aerial`} ×
  `box_threshold` ∈ {0, 0.05, 0.10, …, 0.50, 0.60}; `text_threshold` fixed at
  0.25 (it cannot change the selected box — verified separately, §3).
  - `expression`: the raw referring expression (baseline).
  - `head`: the expression truncated before its first relational clause
    (`in/on/at/near/next to/…/is/are/located…`), derived from the expression alone.
  - `aerial`: `"aerial image of " + expression`.
  - `category` (readable class name only) is run as a **diagnostic** and is
    **not eligible**: it uses the annotated class, which a user query does not carry.
- **Objective**: val Pr@0.5 with top-1 selection (highest-confidence returned
  box), a missing box scored as IoU 0 and kept in the denominator.
- **Tie-breaks**: prompt ties → `expression`; threshold ties → the highest
  threshold reaching the maximum (fewest boxes returned).
- **Test use**: the chosen configuration is run **once** on T-quick and
  reported beside the baseline. No further change follows from that number.

## Summary

| Result | Number | Experiment |
|---|---|---|
| Zero-shot T-quick reproduction (CPU, canonical harness) | **65/400 = 0.1625**, mean IoU **0.186203**: exact match; 0 per-example hit flips vs the original T4 run | SQ-20260926-201 |
| Same on Apple MPS | 65/400, mean IoU 0.186203; 0 hit flips vs T4 and CPU | SQ-20260926-202 |
| `text_threshold` effect on the selected box | none (verified: identical boxes, scores, IoUs at 0.25 vs 0.90) | SQ-20260926-205 |
| Val-selected config (expression, box 0.20), run once on T-quick | **75/400 = 0.1875** [0.152, 0.229], mean IoU 0.2203; paired 10 gains / 0 losses (McNemar p = 0.002) | SQ-20260926-209 |
| **T-full** zero-shot, all 7,500 official test expressions, box 0.35 | **1016/7500 = 0.1355** [0.128, 0.143], mean IoU 0.1533 | SQ-20260926-210 |
| T-full at the val-selected box 0.20 (offline replay, parity proven) | 1230/7500 = 0.1640 [0.156, 0.173], mean IoU 0.1902 | SQ-20260926-210 |
| Prompt rewrites (`head`, `aerial`) | both **worse** than the raw expression on val | SQ-20260926-206/207 |
| Adaptation justified? | **Yes**: zero-shot tops out at 0.164 on T-full at any threshold; failures are relational, small-object and dense-scene errors that no threshold reaches | SQ-20260926-211 |

Intervals are 95 % Wilson. T-quick numbers are a regression guard. Per
Protocol 01 §4, per-class numbers come only from T-full. T-quick is
class-balanced, so its baseline (0.1625) overstates the full-distribution
baseline (0.1355).

## 1. Setup and provenance

**Dataset.** DIOR-RSVG from the official Google Drive folder used by
`kaggle/run_grounding_dior_rsvg.py`. Checksums, split sizes and overlaps are
in `eval/results/grounding/dior-rsvg-dataset-audit.json` (SQ-20260926-200):

| Split | Expressions | Images |
|---|---|---|
| train | 26,991 | 14,730 |
| val | 3,829 | 3,465 |
| test | 7,500 | 6,102 |

**The official splits are not image-disjoint**: train∩test = 4,206 images
(69 % of test images), train∩val = 2,449, val∩test = 1,115. Splits are by
expression, so the same scene appears in train and test with different
referred objects. Zero-shot evaluation is unaffected. Selection data
(`val-select`) and any adaptation data must exclude test images (§0, §6).
DIOR images carry no coordinates, so geographic blocking is impossible here;
image identity is the finest leakage control available.

`JPEGImages.zip` (5,386,728,679 bytes, sha256 `1c0a52b1…690e`) is a ZIP64
archive: macOS `unzip` reports it corrupt, but Python `zipfile` reads all
17,402 members. Only the 790 images needed for T-quick and `val-select` are
kept extracted. The T-full run extracted each image transiently and deleted it.

**Model.** `groundingdino_swint_ogc.pth` from HF `ShilongLiu/GroundingDINO`
revision `a94c9b567a2a374598f05c584e96798a170c56fb`, sha256
`3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`;
text tokenizer `bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`;
config `GroundingDINO_SwinT_OGC.py` packaged with `groundingdino-py==0.4.0`.
These are the same package and versions the Kaggle runner pins. The
repository manifest (`configs/model_artifacts.json`) still has
`revision: null, sha256: null` for this provider; the values above are the
first pinned identity.

**Environment.** Python 3.11.15 (uv venv), torch 2.14.0, torchvision 0.29.0,
transformers 4.49.0, groundingdino-py 0.4.0 (no compiled `_C` op; the pure
PyTorch deformable-attention path is used). Apple M3 Pro, 18 GB. The CPU
was shared with four other agents throughout. Total inference: 6,063 s over
10 runs.

**Harness.** `eval/grounding_research.py` loads the checkpoint on CPU or MPS
and injects it into an unmodified `GroundingDINOModel`, pre-seeding the
lazy-load slots. That bypasses only `_load()`'s CUDA gate. `infer()`, its
evidence validation and the cxcywh→xyxy conversion are production code, and
the suite's `normalize_xyxy`, `iou_xyxy` and `_top_prediction` do the scoring.
A forward hook stores the raw decoder outputs (900 query scores + boxes),
computed with exactly the tensor ops `predict()` uses, so thresholds can be
replayed offline.
**Every offline number was checked against its online run: 0 mismatches in
all 10 runs.** Repeated calls on one image were bit-identical on each device,
and the two MPS T-quick runs are bit-identical on 400/400.

## 2. Day 10 — reproduction

The canonical `suite.stratified_sample(seed=26167)` selects exactly the
original 400 test indices, expressions and images.

| Run | Device | Harness | Hits | Pr@0.5 | Mean IoU | No-detection | Runtime |
|---|---|---|---|---|---|---|---|
| Original (`4d43caa`, git_sha `14e8c14`) | Kaggle T4 | suite | 65 | 0.1625 | 0.186203 | 0.275 | — |
| SQ-…-201 | CPU | suite | 65 | 0.1625 | 0.186203 | 0.275 | 616 s |
| SQ-…-202 | MPS | suite | 65 | 0.1625 | 0.186203 | 0.275 | 245 s |
| SQ-…-203 | MPS | research loop | 65 | 0.1625 | 0.186203 | 0.275 | 229 s |

Per-example comparison against the original T4 predictions: **0 hit
flips, 0 detection-presence flips, identical detection counts on all 400**.
Numeric drift: max |ΔIoU| 2.1e-4 (CPU) and 1.2e-4 (MPS); max |Δscore|
7.2e-4 (CPU) and 4.9e-4 (MPS). The mean IoU agrees to 6 decimals. No selected box is
bit-identical across devices, as expected for float kernels, but the drift
changed no hit and no detection outcome. The loop harness used for val and
T-full is bit-identical to `suite.evaluate` (400/400).

**Verdict: reproduced exactly.** The discrepancy investigation found only
sub-1e-3 float drift. MPS parity with both T4 and CPU justified running the
study on MPS (~0.5 s/expression vs ~1.5 s on the shared CPU). T-quick CPU
per-example predictions are committed in
`eval/results/grounding/tquick-cpu-reproduction-predictions.json`.

## 3. Day 11 — threshold and prompt study

### 3.1 Text threshold: invariance verified, with its mechanism

`groundingdino.util.inference.predict()` keeps a query iff
`max_token_score > box_threshold` and reports that max as the confidence.
`text_threshold` is used only afterwards, to build the phrase label from
tokens above it. The provider selects top-1 by confidence, so `text_threshold`
**cannot** change which box is returned, its score, or the number of boxes.
Empirically (SQ-…-205, val-select, 0.25 vs 0.90): identical selected boxes,
scores, IoUs and detection counts on 400/400. Only labels change (0/345
empty at 0.25, 337/345 empty at 0.90). The earlier "text-threshold invariance"
observation is therefore structural, not configuration-specific.

### 3.2 Box threshold: a coverage knob, not an accuracy knob

With top-1 selection, lowering the threshold never changes an existing answer
(the top query stays the top query). It only adds answers where there were
none, so **Pr@0.5 is non-increasing in box_threshold**. Val-select, raw
expression (SQ-…-204, offline replay):

| box | Hits | Pr@0.25 | Pr@0.5 | Pr@0.75 | Mean IoU | Answered | Pr@0.5 among answered | ≥2 boxes | Mean boxes |
|---|---|---|---|---|---|---|---|---|---|
| 0.00–0.20 | 87 | 0.365 | 0.2175 | 0.140 | 0.258 | 1.000 | 0.218 | 0.85–1.00 | 3.2–900 |
| 0.25 | 85 | 0.360 | 0.2125 | 0.135 | 0.253 | 0.975 | 0.218 | 0.63 | 2.0 |
| 0.30 | 82 | 0.345 | 0.2050 | 0.130 | 0.243 | 0.897 | 0.228 | 0.38 | 1.3 |
| **0.35 (baseline)** | 74 | 0.303 | 0.1850 | 0.118 | 0.214 | 0.735 | 0.252 | 0.12 | 0.9 |
| 0.40 | 64 | 0.255 | 0.1600 | 0.103 | 0.182 | 0.573 | 0.280 | 0.04 | 0.6 |
| 0.50 | 36 | 0.148 | 0.0900 | 0.063 | 0.108 | 0.330 | 0.273 | 0.01 | 0.3 |
| 0.60 | 17 | 0.078 | 0.0425 | 0.033 | 0.055 | 0.155 | 0.274 | 0.00 | 0.2 |

The same shape on T-full (all 7,500, offline replay):

| box | Hits | Pr@0.5 | Mean IoU | Answered | Pr@0.5 among answered |
|---|---|---|---|---|---|
| ≤0.20 | 1230 | 0.1640 | 0.190 | 1.000 | 0.164 |
| 0.30 | 1165 | 0.1553 | 0.178 | 0.875 | 0.178 |
| 0.35 | 1016 | 0.1355 | 0.153 | 0.683 | 0.198 |
| 0.45 | 651 | 0.0868 | 0.094 | 0.328 | 0.265 |
| 0.60 | 333 | 0.0444 | 0.046 | 0.119 | 0.373 |

A higher threshold trades coverage for precision among the answers it gives
(0.164 → 0.373). That is a legitimate abstention curve built on a real model
score (input for Day 19). It is **not** calibration: the scores are not
probabilities of a correct box.

### 3.3 Prompt wording (val-select, best threshold per prompt)

| Prompt | Eligible | Best val hits | Best val Pr@0.5 | At 0.35 | No-detection at 0.35 |
|---|---|---|---|---|---|
| `expression` (raw) | yes | **87** | **0.2175** | 0.185 | 0.265 |
| `head` (relational clause removed) | yes | 77 | 0.1925 | 0.1775 | 0.120 |
| `aerial` ("aerial image of …") | yes | 57 | 0.1425 | 0.1325 | 0.093 |
| `category` (annotated class name) | diagnostic | 64 | 0.1600 | 0.1525 | 0.103 |

Negative results: both deployable rewrites are worse. Dropping the
relational clause makes the model fire more often (fewer no-detections) but
on the wrong instance. The prefix inflates scores across many boxes (≥2
boxes for 67 % of expressions at 0.35) without improving top-1. The raw
expression beats the class name alone, so the referring attributes carry
signal even though relations are mostly not resolved.

### 3.4 Selection and the single test run

The pre-registered rule selects **prompt = expression, box_threshold = 0.20,
text_threshold = 0.25**. Run once on T-quick (SQ-…-209):

| T-quick (n=400) | Hits | Pr@0.25 | Pr@0.5 [95 % CI] | Pr@0.75 | Mean IoU | No-detection | Mean boxes |
|---|---|---|---|---|---|---|---|
| Baseline 0.35 | 65 | 0.270 | 0.1625 [0.130, 0.202] | 0.100 | 0.1862 | 0.275 | 0.84 |
| Chosen 0.20 | 75 | 0.3075 | 0.1875 [0.152, 0.229] | 0.110 | 0.2203 | 0.000 | 3.25 |

Paired: 10 gains, 0 losses, exact McNemar p = 0.002. All 10 gains were
no-detections at 0.35, and every answered expression kept its exact box. The
unpaired intervals overlap. On T-full the same change moves 0.1355 → 0.1640
with non-overlapping intervals.
**This is threshold tuning, not fine-tuning.** The cost is more returned
boxes (mean 3.28 vs 0.78 on T-full) and losing the abstention signal at this
threshold. Adopting 0.20 in the provider is a product decision (coverage vs
precision among answered). This study does not change the provider default.

## 4. Day 12 — failure analysis (T-full, zero-shot, box 0.35; SQ-…-211)

### 4.1 Per class (n = 125–1,415 per class; all ≥30, so Protocol 01 §5.3 applies to every class)

| Class | n | Hits | Pr@0.5 [95 % CI] | Mean IoU | No-detection |
|---|---|---|---|---|---|
| stadium | 222 | 97 | 0.437 [0.373, 0.503] | 0.408 | 0.113 |
| airplane | 326 | 142 | 0.436 [0.383, 0.490] | 0.360 | 0.055 |
| ship | 383 | 130 | 0.339 [0.294, 0.388] | 0.325 | 0.120 |
| storagetank | 251 | 64 | 0.255 [0.205, 0.312] | 0.228 | 0.215 |
| groundtrackfield | 465 | 112 | 0.241 [0.204, 0.282] | 0.245 | 0.301 |
| golffield | 222 | 33 | 0.149 [0.108, 0.201] | 0.281 | 0.140 |
| basketballcourt | 249 | 36 | 0.145 [0.106, 0.194] | 0.187 | 0.229 |
| vehicle | 1415 | 200 | 0.141 [0.124, 0.160] | 0.110 | 0.365 |
| chimney | 221 | 31 | 0.140 [0.101, 0.192] | 0.131 | 0.593 |
| dam | 177 | 20 | 0.113 [0.074, 0.168] | 0.166 | 0.412 |
| overpass | 481 | 43 | 0.089 [0.067, 0.118] | 0.130 | 0.347 |
| trainstation | 180 | 14 | 0.078 [0.047, 0.126] | 0.124 | 0.272 |
| expressway_service_area | 289 | 22 | 0.076 [0.051, 0.113] | 0.124 | 0.609 |
| expressway_toll_station | 240 | 13 | 0.054 [0.032, 0.090] | 0.070 | 0.254 |
| bridge | 473 | 25 | 0.053 [0.036, 0.077] | 0.086 | 0.397 |
| harbor | 125 | 6 | 0.048 [0.022, 0.101] | 0.074 | 0.472 |
| airport | 212 | 8 | 0.038 [0.019, 0.073] | 0.192 | 0.052 |
| baseballfield | 668 | 14 | 0.021 [0.013, 0.035] | 0.053 | 0.347 |
| tenniscourt | 307 | 3 | 0.010 [0.003, 0.028] | 0.054 | 0.355 |
| windmill | 594 | 3 | 0.005 [0.002, 0.015] | 0.071 | 0.399 |

Natural-image-like, visually distinctive objects (stadium, airplane, ship)
transfer. Top-down sports fields, windmills and infrastructure do not.
Airport answers almost always (no-detection 0.05) with mean IoU 0.19 but
Pr@0.5 0.04: it finds the region and misjudges the extent.

### 4.2 Structural factors

| Factor | Group | n | Pr@0.5 [95 % CI] | No-detection |
|---|---|---|---|---|
| Object size (gold box, px²) | small <32² | 1,321 | 0.039 [0.029, 0.050] | 0.437 |
| | medium <96² | 2,183 | 0.084 [0.073, 0.097] | 0.334 |
| | large ≥96² | 3,996 | 0.195 [0.183, 0.208] | 0.269 |
| Annotated same-class instances in image | 1 | 3,008 | 0.214 [0.200, 0.229] | 0.240 |
| | 2–3 | 3,357 | 0.086 [0.077, 0.096] | 0.362 |
| | ≥4 | 1,135 | 0.071 [0.058, 0.088] | 0.389 |
| Expression has a relational clause | no | 1,565 | 0.242 [0.221, 0.263] | 0.114 |
| | yes (79 %) | 5,935 | 0.108 [0.100, 0.116] | 0.371 |

These factors overlap (dense scenes carry more relational expressions and
smaller objects), so they are descriptive, not causal effects. "Relational"
means the `head` rule found a relational cue.

### 4.3 Failure types (6,484 misses)

| Type (deterministic rule, `classify_failure`) | Count | Share |
|---|---|---|
| `no_detection`: no box above 0.35 | 2,380 | 36.7 % |
| `background_or_unannotated`: top box overlaps no annotated object ≥0.5 and the target <0.1 | 2,133 | 32.9 % |
| `poor_localization_of_target`: 0.1 ≤ IoU(target) < 0.5 | 1,181 | 18.2 % |
| `same_class_other_instance`: another annotated object of the same class, IoU ≥0.5 | 501 | 7.7 % |
| `other_class_object`: an annotated object of another class, IoU ≥0.5 | 289 | 4.5 % |

Only `no_detection` is reachable by lowering the threshold, and even then only
a minority become hits (T-full: +214 hits from 2,380 no-detections).
DIOR-RSVG XML lists only referred objects, so `same_class_other_instance`
undercounts relational confusion. Some `background_or_unannotated` cases are
real but unannotated objects of the target class.

Representative renders (2 per outcome, the first by `test_index`, not
hand-picked; gold green, top-1 red) are in
`/Users/atharva/satquery-data/grounding/viz/r10-tfull/` (12 PNGs, outside
git). Example: `same_class_other_instance__00006.png`: for "The ship on the
top" the model boxes the lower ship.

## 5. Verdict: is adaptation justified?

**Yes.**

- The best zero-shot configuration found reaches Pr@0.5 0.164 on T-full,
  below Protocol 01's 0.25 acceptance bar. No box threshold can exceed it,
  because thresholds only add answers.
- Prompt rewriting made things worse.
- 63 % of misses are wrong boxes that the model scored confidently
  (background, poor localisation, wrong instance/class). They concentrate in
  relational expressions (79 % of the test set), small objects and dense
  scenes. These are what referring-expression training changes, and what
  inference settings cannot reach.
- Supervised DIOR-RSVG methods report 0.77–0.86 Pr@0.5 (§6.5), so the gap is
  large even allowing for their image-overlap advantage.

The adaptation must use image-disjoint data (§6.2).

## 6. Day 13 — bounded adaptation plan (not executed: no CUDA GPU in this environment)

Threshold and prompt changes are inference settings, **not fine-tuning**. The
only way to change what the model localises is to update weights, which needs
a CUDA GPU (Kaggle T4 or better). This section is the plan; nothing in it has
been run.

### 6.1 Licence and permitted use

| Artefact | Licence | Consequence |
|---|---|---|
| DIOR-RSVG (annotations + images) | CC BY-NC 4.0 per the README ("for Non-commercial use only. Any commercial use should get formal permission first"); the GitHub repo has no LICENSE file ([RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch)) | Research/hackathon use is fine; an adapted checkpoint trained on it inherits the non-commercial restriction for any product use. |
| Grounding DINO code + Swin-T OGC weights | Apache-2.0 ([IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)) | No restriction beyond attribution. |

### 6.2 Data: image-disjoint by construction

The official splits share images (§1): training on official `train.txt` and
testing on `test.txt` puts 4,206 of the 6,102 test images in training. An
adapted model must therefore train on:

- **train-clean**: official train expressions whose image appears in neither
  `test.txt` nor `val.txt` — **15,237 expressions / 8,950 images** (measured;
  43.5 % of official train expressions are dropped). Every class keeps ≥165
  expressions (min class) and ≤2,057 (max class).
- **validation**: `val-select` (§0) for checkpoint/threshold choice; the
  remaining 2,561 val-not-on-test expressions are available for early stopping.
- **test**: T-full (all 7,500 official test expressions) for the headline
  number, T-quick only as a regression guard (Protocol 01 §4).

A committed assertion that train-clean ∩ test images = ∅ and train-clean ∩
val-select images = ∅ must exist before the first training step.

Published supervised DIOR-RSVG results are trained on the official split and
so may benefit from the same image overlap; they are not a clean ceiling for
an image-disjoint adaptation.

### 6.3 Method

Fine-tune the same Swin-T OGC checkpoint as a referring-expression detector
(one positive box per expression, top-1 at inference). MB-ORES reports
building on "a fine-tuned GroundingDINO" in this setting
([arXiv:2503.24219](https://arxiv.org/abs/2503.24219)). The official
IDEA-Research repository ships inference only; candidate training stacks are
MM-Grounding-DINO in MMDetection and the Hugging Face
`GroundingDinoForObjectDetection` port. **Switching implementation for
training is itself a variable**: the adapted model must be re-evaluated
zero-shot in the training stack first, and that zero-shot number must match
this report's T-quick result per example before any fine-tuned number is
compared with it.

Bounded ladder, identical evaluation throughout:

1. Pilot: 100 optimiser steps on train-clean, freeze the text encoder, record
   loss, peak VRAM, step time, checkpoint size.
2. 1 epoch on train-clean → val-select Pr@0.5.
3. Only if (2) improves val-select by more than its 95 % CI: up to 3 epochs,
   early stopping on val.
4. Freeze config → one T-full run → Protocol 01 §5 acceptance
   (≥25 % Pr@0.5 with CI separation; no class with ≥30 test expressions
   regressing by >10 pp; mean IoU not regressing).

### 6.4 Compute estimate (unmeasured; the pilot replaces it)

Inference measured here: ~1.2 s/expression (single image) to ~1.5 s (T-quick
run) on the shared M3 Pro CPU, ~0.5 s on MPS. As a rule of thumb, training
costs about 3× inference per sample (forward + backward) plus optimiser
overhead. On a T4 (fp16, batch 2–4 at 800 px) an order-of-magnitude
estimate is **1–3 h per epoch of 15k expressions**, so the 3-epoch ceiling
fits within one Kaggle weekly quota. Whether Swin-T at batch 2 with frozen
BERT fits in 16 GB is an assumption; the pilot measures peak VRAM.

### 6.5 Remote-sensing grounding alternatives (surveyed, not adopted)

Pr@0.5 values below are as tabulated in the MB-ORES paper
([arXiv:2503.24219](https://arxiv.org/abs/2503.24219)), on the full official
DIOR-RSVG test split with models trained on the official train split.

| Method | Type | Reported DIOR-RSVG Pr@0.5 | Notes / risks for SatQuery |
|---|---|---|---|
| MGVLF (dataset authors, TGRS 2023) | supervised specialist | 76.78 | Trained on official split. |
| LPVA | supervised specialist | 82.27 | Trained on official split. |
| MB-ORES | fine-tuned Grounding DINO + object reasoner | 85.65 | Closest to our stack. |
| GeoGround | RS vision–language model | 77.73 | Unified HBB/OBB/mask grounding ([arXiv:2411.11904](https://arxiv.org/abs/2411.11904)). |
| EarthGPT | RS multimodal LLM | 76.65 | [arXiv:2401.16822](https://arxiv.org/abs/2401.16822). |
| LAE-DINO (AAAI 2025, MIT) | RS open-vocabulary **detector** | not a grounding benchmark | LAE-1M's fine-grained part includes DIOR ([dataset card](https://huggingface.co/datasets/jaychempan/LAE-1M)), so any DIOR-RSVG evaluation of it is contaminated ([repo](https://github.com/jaychempan/LAE-DINO)). |
| RSVG-ZeroOV (AAAI 2026) | training-free zero-shot (VLM + diffusion attention) | not reported on DIOR-RSVG | Zero-shot baselines in its RRSIS-D table sit at 16.8–28.7 % Pr@0.5 ([arXiv:2509.18711](https://arxiv.org/abs/2509.18711)). |

These are reference points, not comparable targets: different test sampling
from T-quick, and training data that shares test images. None of these methods
replaces the provider without its own locked evaluation under this protocol.

## 7. Negative results, deviations and limitations

- **Negative**: `head` and `aerial` prompt rewrites lower val Pr@0.5; the
  class-name diagnostic is below the raw expression; no threshold reaches the
  0.25 acceptance bar.
- **Deviation from the directive**: the reproduction ran on CPU as asked
  (SQ-…-201). The study runs used Apple MPS for speed, justified by exact
  per-example parity with T4 and CPU (SQ-…-202). Neither is the CUDA
  production path.
- **Provenance**: SQ-…-201 predates per-run SHA capture; its record pins
  `9f9e92f` from timestamps (started 36 s after that commit; the only
  untracked file was the dataset audit JSON).
- Per-class and structural numbers come from a single zero-shot model at one
  threshold. The factors are confounded (§4.2).
- `val-select` is 400 expressions; the choice between thresholds 0.20 and 0.25
  (87 vs 85 hits) is within noise. Any threshold ≤0.20 gives identical top-1
  results.
- Scores are not calibrated probabilities; "Pr@0.5 among answered" is
  selective accuracy at a coverage level, not a confidence.

## 8. Reproduction

```bash
# environment (outside git)
uv venv --python 3.11 $SATQUERY_DATA/venvs/grounding
VIRTUAL_ENV=$SATQUERY_DATA/venvs/grounding uv pip install torch torchvision \
  "transformers==4.49.0" huggingface-hub gdown pillow numpy pytest "groundingdino-py==0.4.0"
export HF_HOME=$SATQUERY_DATA/grounding/hf-home HF_HUB_OFFLINE=1   # after caching:
#   ShilongLiu/GroundingDINO@a94c9b56… groundingdino_swint_ogc.pth
#   bert-base-uncased@86b5e093… (config, vocab, tokenizer, model.safetensors)
PY=$SATQUERY_DATA/venvs/grounding/bin/python
DATA=$SATQUERY_DATA/grounding/dior-rsvg          # gdown --folder of the official Drive folder;
                                                 # extract Annotations.zip + split txts to $DATA/extracted
CKPT=$HF_HOME/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth

$PY eval/grounding_research.py audit --data-root $DATA/extracted --files $DATA/*.zip $DATA/extracted/*.txt --out eval/results/grounding/dior-rsvg-dataset-audit.json
# SQ-201: exact reproduction (canonical harness, CPU)
$PY eval/grounding_research.py run --harness suite --selection tquick --data-root $DATA/extracted --checkpoint $CKPT --out-dir runs/r01
# SQ-202..210: same with --device mps; e.g.
$PY eval/grounding_research.py run --device mps --selection val-select --prompt head --data-root $DATA/extracted --checkpoint $CKPT --archive $DATA/JPEGImages.zip --out-dir runs/r06
$PY eval/grounding_research.py run --device mps --selection tfull --data-root $DATA/extracted --checkpoint $CKPT --archive $DATA/JPEGImages.zip --ephemeral-images --out-dir runs/r10
$PY eval/grounding_research.py sweep --run-dir runs/r10
$PY eval/grounding_research.py compare --reference <original-T4-report.json> --candidate runs/r01 --out compare.json
$PY eval/grounding_research.py analyze --run-dir runs/r10 --data-root $DATA/extracted --archive $DATA/JPEGImages.zip --render-dir viz/
```

Exact per-run commands, start/end times, SHAs and package versions are in each
`eval/registry/SQ-20260926-2xx.json`. The suite-harness run needs the T-quick
images already extracted (it does not use `--archive`).
