# Stage-1 gate evaluation: feeding RSVQA-LR into the 12-band patch embed

Read-only analysis, written during loop 8. No code was changed and nothing was run
against RSVQA-LR; `data/raw/rsvqa_lr/` is not present on this machine, so every
statement below is either a fact read from this repository or is explicitly marked
as unverified.

## The mismatch

Stage 1 trains a Qwen2.5-VL vision tower whose patch embedding was converted from 3
to 12 input channels (`models/qwen_vl/stage1.py::convert_patch_embed`), on
BigEarthNet-S2 patches carrying twelve Sentinel-2 bands. The recorded Stage-0
baseline (`results/qwen2.5vl-3b__rsvqa__20260903T175900Z.json`) never touches that
path: `eval/suites/rsvqa.py` hands the TIFF file paths to `QwenVLModel`, which lets
the stock Qwen processor load them as ordinary images
(`models/qwen_vl/model.py::_generate_answer`).

So the same suite would reach the two models through different input paths unless
the Stage-1 evaluation is made explicit. Comparing a 3-channel-processed Stage-1
model against a 3-channel-processed Stage-0 model is not possible once the patch
embed expects 12 channels: the conv would receive the wrong channel count. The
Stage-1 gate therefore has to build a twelve-band tensor, and the filling rule is
part of the gate.

## What is known about the RSVQA-LR images

* Source: Zenodo record 6344334, `Images_LR.zip`, pinned by size and MD5 in
  `eval/suites/rsvqa.py::FILES`.
* Files are named by database id (`232.tif`), one TIFF per image, 772 images.
* `scripts/dry_run_rsvqa.py` opens a real image and asserts `image.size == (256, 256)`.
* `docs/plan/ANNEX-A-phase0-audit.md` line 198 records the set as "772 `.tif`
  Sentinel-2 256×256 (148 MB)".
* The suite notes that each record keeps an `original_name` holding the long
  Sentinel-2 product name.
* **Not verified locally:** the band count and per-band description of those TIFFs,
  their data type, their DN scale (whether they are S2 L2A BOA x 10000, a rescaled
  8-bit visualisation, or something else), and their geographic extent. The files
  are not on this machine and this loop downloaded nothing.

The user-stated fact "RSVQA-LR images are 3-band" is consistent with everything in
the repository but is not confirmed by any artifact here. It must be confirmed from
`LR_split_test_images.json` plus one opened TIFF before the gate is run.

## Region comparison

BigEarthNet tiles in the frozen split, by tile (measured from
`caption-geo-split.v1.csv.gz`):

| side | tiles | countries |
| --- | --- | --- |
| train | 48 | Finland 21, Austria 7, Lithuania 5, Serbia 4, Ireland 4, Portugal 3, Belgium 2, Switzerland 1, Luxembourg 1 |
| eval | 6 | Serbia 2, Lithuania 2, Austria 1, Ireland 1 (+ Kosovo on one tile) |

The eval tiles share four countries with the train tiles: the split is
MGRS-blocked, not country-blocked, so a country-level domain shift is already
present inside BigEarthNet itself. The RSVQA-LR extent is unknown from this
repository. Any statement that RSVQA-LR is "in-domain" or "out-of-domain" relative
to BigEarthNet would be invented until the extents are read from the split JSON.

## Options

**A. Neutral fill.** Put the available bands in their true Sentinel-2 slots and fill
every missing band with the frozen per-band training mean, so standardization
yields exactly 0 in those slots.
* Assumes: the three channels are identifiable (an RGB rendering maps to
  B04/B03/B02); the DN scale is S2 L2A BOA-like; the patch embed's response to a
  zero (mean-valued) channel is not pathological.
* Risks: during training every one of the twelve channels carried real signal, so
  nine dead channels are out of distribution. The conv output for those channels
  collapses towards the bias term, which shrinks the feature magnitude relative to
  training. This is a distribution shift that is identical for Stage-0 and Stage-1
  only if both are evaluated through the same path.

**B. Zero-fill after normalization.** Fill with literal 0 in standardized space.
Mathematically the same as A when the fill is the training mean; the difference is
only bookkeeping. Filling with 0 in *raw DN* space instead would clip to the
contract floor after standardization, which is a different (worse) value.

**C. Band duplication.** Copy the nearest available band into physically similar
slots (for example B08 into B8A, B07 into B11).
* Assumes: spectral similarity is close enough for the trained filters.
* Risks: fabricates signal the model was trained to read; a duplicated band is
  detectable and would flatter a model that keys on spectral ratios.

**D. Evaluate Stage-1 through the stock 3-channel processor** by padding or
slicing the 12-channel conv back to 3.
* Assumes: discarding nine trained channels is acceptable.
* Risks: it silently discards the adaptation the whole stage exists to install,
  and the baseline number stops being comparable to anything.

## Recommendation

Option A, with these parts frozen before the first gate sample:

1. Band mapping: channel 0 -> B04, 1 -> B03, 2 -> B02, all other slots filled with
   the frozen per-band training mean DN (standardized result exactly 0).
2. Order of operations: resample 256x256 -> 120x120 with rasterio bilinear (the same
   resampling the contract applies), then divide by 10000, clip to [0, 0.9113],
   then standardize with the frozen per-band mean/std, then overwrite the nine
   unfilled slots with 0.
3. Target size 120x120, matching training exactly.
4. The same 12-band path and the same fill must be used for the Stage-0 and the
   Stage-1 model at the gate. The legacy 0.1651 figure (git `777707a`, revision not
   recorded) must never be the comparison point.
5. The DN scale assumption must be checked first: open a handful of the pinned
   test images, record count, dtype, per-band min/median/max, and confirm the values
   are in the thousands rather than 0-255. If they are 8-bit, option A's
   normalization is wrong and the mapping needs an explicit scale factor, which is
   a separate pre-registration.

## What must be pre-registered before any RSVQA gate evaluation

* the chosen option and the exact band mapping table;
* the fill value and the order fill-versus-normalize;
* the resampling method and target size;
* the assumed DN scale, with the measurement that justifies it;
* the primary metric (open_accuracy on the 10,004-question official test split),
  the n, and the comparison rule between Stage-0 and Stage-1 under the identical
  input path;
* the runtime, which is currently UNKNOWN: greedy generation over 10,004 questions
  has never been measured on this input path, so it must be measured, not assumed.

## Known limits of this note

The RSVQA-LR images were not opened in this loop. Everything about their band
count, scale, and extent is unverified here and is listed as a required
pre-registration step rather than assumed.