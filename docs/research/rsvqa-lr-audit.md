# RSVQA-LR audit: reproduction, geographic leakage, subsets, base-result comparability

Registry: `SQ-20260926-101` … `-104` (all PASSED, re-run byte-identically at `1e01729`). Code: `eval/rsvqa_research.py`. CPU only; no model was run.

## 1. Reproduction (SQ-20260926-101)

The 12 official Zenodo 6344334 files were downloaded and MD5-verified by `eval.suites.rsvqa.download_rsvqa_lr`. `scripts/prepare_rsvqa_training_manifest.py` regenerated the manifest with SHA-256 `23a74c573026a26620262eb2440a7effa10168f77010f40dee81a77996fed3e1`, the recorded value. Counts: train 57,223 QA / 572 images; validation 10,005 / 100; test 10,004 / 100; total 77,232. No image ID appears in more than one split.

## 2. Geographic leakage (SQ-20260926-102)

**Coordinate system.** `upperleft_map_x/y` in `LR_split_*_images.json` are **EPSG:3857 (Web Mercator)**, not UTM. Within each source tile they are exactly linear in the `<col>-<row>` pixel offset embedded in `original_name`: a least-squares fit gives **zero residual**, with a per-tile scale of 1.584–1.685 (3857 units per 10 m pixel). That scale equals sec(latitude) at the tile to within 0.3%. A patch therefore spans 2560 m × sec(φ) in 3857 units. Treating the values as UTM (my first attempt) puts the scenes at about 60°N and hides the overlaps below. That result was discarded.

**Split design.** The official split is by Sentinel-2 tile: validation = all of `31UGU` (2016-07-20), test = all of `31UFU` (2016-07-20), train = `31UET`, `31UFT`, `31UFV`, `31UGS`, `31UGT`, `32ULE`. Adjacent MGRS tiles overlap, and `31UFV` and `32ULE` in train were acquired on **the same date** as validation and test.

**Measured overlap** (footprint boxes; overlap fraction relative to the held-out patch):

| held out vs reference | images with any overlap | QA on those images | max overlap | ≥ 50% | reference centre within 5 km |
|---|---:|---:|---:|---:|---:|
| validation vs train | 8 / 100 | 802 / 10,005 (8.0%) | 51.2% | 1 | 18 |
| test vs train | 4 / 100 | 400 / 10,004 (4.0%) | 55.6% | 1 | 12 |
| test vs validation | 3 / 100 | 301 / 10,004 (3.0%) | 36.3% | 0 | 7 |

**Pixel confirmation.** For every overlapping pair, `pixel_agreement()` correlates grey levels over the predicted overlap, searching ±15 px. Every pair shows a peak of **0.58–0.96** at a 0–4 px shift, against window medians of 0.04–0.69. The overlapping patches image the same ground. Same-date pairs (for example validation 509 / train 689 at 0.905, test 257 / train 365 at 0.739 over 36,424 px) are effectively duplicated pixels. Test image 243's strip is only 20 px wide and its window median is 0.69, so that pair is weak evidence on its own.

Train also contains 16 locations imaged at two dates (31UET, 2018-06-30 L2A and 2018-11-15 L1C). That is a within-train duplicate, not leakage. It does mean train mixes 82 L2A patches with L1C, while all held-out data is L1C.

**Handling (official split unchanged):**
- The dev subset used for model selection **excludes** the 8 overlapping validation images.
- The locked test evaluation uses the **whole official split** (primary number). It flags the 4 overlapping test images and also reports a **footprint-disjoint** number without them (400 QA removed) as a sensitivity check.
- The claim "cross-split image overlap: ZERO" is true for image IDs only. It is false for ground footprints.

## 3. Evaluation subsets (SQ-20260926-104)

| subset | file / SHA-256 | selection | counts |
|---|---|---|---|
| validation dev (Days 3–7) | `eval/subsets/rsvqa-lr-validation-dev1000.v1.json` / `e3b43319…4e20c` | seed 26167; the 92 footprint-disjoint images; `rural_urban` (≤ 100) taken whole; remaining 908 proportional to type size (largest remainder); within a type, ascending `sha256(seed‖sample_id)` | comp 368, count 271, presence 269, rural_urban 92 |
| locked test (Day 8 only) | rebuilt on Kaggle, pinned `b21e5046…057b5` | whole official split, overlaps flagged | comp 4,002, count 2,947, presence 2,955, rural_urban 100 |

Model selection, the training-size ladder and the LoRA matrix all use only the validation dev subset. Test is evaluated once, after `run_vlm_program.py lock` writes an immutable lock containing the chosen adapter's SHA-256. Choosing a configuration on test would make the test number an optimistic, tuned estimate.

`type_weighted_strict_accuracy` reweights per-type accuracy to the full split's type mix, so the overall figure is not distorted by `rural_urban` being taken whole. The dev subset has 271 count and 92 rural_urban questions. Per-type Wilson intervals are reported because `eval/eval.py` documents that small open-ended samples are unstable.

## 4. Metrics

Defined in `eval.rsvqa_research.METRICS` and embedded in every report. `strict_accuracy` is primary. `lenient_accuracy` exists only for comparison with the historical number. RSVQA-LR count answers are **raw integers** (1,259 distinct values in train, some in the hundreds), so exact match on counts is severe. `count_bin_accuracy` (0 | 1–10 | 11–100 | 101–1000 | >1000) is reported alongside. Paired base-vs-adapter differences use an exact McNemar test over identical sample IDs.

## 5. Is the historical base result comparable? (SQ-20260926-103)

`results/qwen2.5vl-3b__rsvqa__20260903T175900Z.json` (Tesla T4, git `777707a`, all 10,004 test QA) was re-scored without inference:

| metric (test, n = 10,004) | value |
|---|---:|
| lenient (historical scorer) | 0.5142 (reproduces the recorded 0.514194) |
| strict | 0.5095 [0.4997, 0.5193] |
| invalid output rate | 1.6% |
| strict by type: comp / presence / count / rural_urban | 0.724 / 0.574 / 0.149 / 0.650 |
| count-bin accuracy | 0.317 |

| aspect | historical path (`eval/eval.py` + `QwenVLModel`) | adaptation path (`scripts/evaluate_remote_sensing_adapter.py`) |
|---|---|---|
| prompt text | question + " Answer with a single word or number only. No explanation." | identical (`ANSWER_INSTRUCTION`) |
| image | native 256 px → `smart_resize` 252×252 | `--image-size`; the runner pins 252, verified pixel-identical with qwen-vl-utils 0.0.14 |
| decoding | greedy, fp16, `max_new_tokens=50` | greedy, fp16, `max_new_tokens=32` |
| scorer | lenient `answer_matches` | strict primary; lenient, count-bin and invalid reported |
| samples | whole test split | dev subset for selection; whole test once after lock |
| model revision / packages | unrecorded | pinned `66285546…` with checksum verification; versions recorded |

**Verdict: not comparable for a base-vs-adapter claim.** Scorer choice alone moves the same predictions by 0.47 points overall and 1.3 points on presence. Revision, library versions and `max_new_tokens` are also uncontrolled. Base is therefore re-measured on the adaptation path twice: BASELINE V0 on the dev subset (Day 3), and the full test split in the **same** Day 8 run as the adapter. The historical file stays as a record of a different protocol. No configuration decision uses its test numbers.

One confound remains, and it is documented rather than removed: adapters are trained on a 4-bit NF4 base but evaluated on the fp16 base, as in the pre-existing script.
