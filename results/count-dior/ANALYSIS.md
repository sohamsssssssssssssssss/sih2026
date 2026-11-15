# DIOR val detector-count analysis (15 Nov 2026)

Run `python3.11 analyze_count_dior.py` to produce every number below from `dior-count-val-evaluation.json`. Provenance is in [SOURCE.md](SOURCE.md).

## Verdict

**At every box threshold (0.25 to 0.40), Grounding DINO counting does not beat the always-0 baseline on exact match (best 0.364 vs 0.500). Its MAE at best ties the baseline (1.488 vs 1.484).**

- **Cause:** on absent-class pairs, the detector returns at least one box 75 to 96% of the time (84% at 0.35).
- **Present-class pairs:** the detector is informative here (exact 0.492 at 0.35, where always-0 scores 0). At thresholds of 0.35 and above, it under-counts more than it over-counts.

The sample is 1:1 present/absent, so always-0 scores 50% exact by construction. Both the mixed and the split numbers are reported below.

## Setup

- **Sample:** a frozen 1,000 (image, class) pairs from DIOR val: 500 present, 500 absent, 50 per class, seed 17, 942 distinct images.
- **Dataset:** a third-party mirror, pinned by sha256.
- **Model:** Grounding DINO Swin-T with text threshold 0.25 and NMS IoU 0.5. The configured box threshold is 0.35.
- **Scope:** a single seed, on val only.

## Overall and by ground truth

| Box thr | Exact | MAE | Within ±1 | GT=0: false count | GT>0: exact | GT>0: over | GT>0: under | GT>0: predicted 0 |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.194 | 2.080 | 0.554 | 479/500 (96%) | 0.346 | 210 | 117 | 7 |
| 0.30 | 0.268 | 1.698 | 0.708 | 463/500 (93%) | 0.462 | 116 | 153 | 18 |
| **0.35** | **0.324** | **1.528** | **0.804** | **422/500 (84%)** | **0.492** | **57** | **197** | **57** |
| 0.40 | 0.364 | 1.488 | 0.834 | 374/500 (75%) | 0.476 | 23 | 239 | 120 |
| always-0 | 0.500 | 1.484 | 0.785 | 0/500 | 0.000 | 0 | 500 | 500 |
| train-majority | 0.500 | 1.484 | 0.785 | — | — | — | — | — |

The train-majority mode is 0 for every class, so that baseline is identical to always-0.

## Paired bootstrap, detector@0.35 minus always-0

The bootstrap uses 2,000 replicates with seed 17, resampling the 942 image_ids.

| Metric | Difference | CI95 |
|---|---|---|
| Exact match | −0.176 | [−0.224, −0.124] |
| MAE | +0.044 | [−0.034, +0.119] |

The detector's exact match is significantly worse than always-0. On MAE, the two are statistically indistinguishable.

## Per class at box threshold 0.35

Each class has 50 pairs (25 present, 25 absent). "GT=0 FP" is the share of absent pairs with a non-zero count.

| Class | Exact | MAE | Within ±1 | GT=0 FP |
|---|---|---|---|---|
| airplane | 0.780 | 0.280 | 0.940 | 0.36 |
| airport | 0.460 | 0.620 | 0.940 | 1.00 |
| baseballfield | 0.220 | 1.060 | 0.820 | 1.00 |
| basketballcourt | 0.260 | 1.060 | 0.760 | 0.84 |
| bridge | 0.340 | 0.900 | 0.840 | 0.88 |
| chimney | 0.400 | 0.780 | 0.880 | 0.72 |
| dam | 0.440 | 0.660 | 0.900 | 0.92 |
| expressway_service_area | 0.080 | 1.060 | 0.860 | 0.88 |
| expressway_toll_station | 0.460 | 0.740 | 0.840 | 0.96 |
| golffield | 0.420 | 0.680 | 0.900 | 0.92 |
| groundtrackfield | 0.340 | 0.780 | 0.900 | 0.96 |
| harbor | 0.180 | 1.640 | 0.580 | 0.92 |
| overpass | 0.300 | 1.020 | 0.800 | 0.88 |
| ship | 0.200 | 9.320 | 0.740 | 0.96 |
| stadium | 0.180 | 0.980 | 0.860 | 1.00 |
| storagetank | 0.220 | 3.040 | 0.560 | 0.72 |
| tenniscourt | 0.060 | 2.300 | 0.600 | 1.00 |
| trainstation | 0.360 | 0.780 | 0.880 | 1.00 |
| vehicle | 0.480 | 1.500 | 0.740 | 0.16 |
| windmill | 0.300 | 1.360 | 0.740 | 0.80 |

Large MAEs come from dense scenes being under-counted. On present pairs, ship predicts 35 objects against 471 true objects (one image has 154), and storagetank predicts 27 against 144. Only airplane and vehicle have a false-count rate below 0.5 on absent pairs.

## Dropped before sampling

All drops have the reason `nonstandard_size`: the XML declares 800 × 783 to 815, not 800 × 800. There were no missing XML or JPEG files in either split.

| Split | Requested | Kept | Dropped ids |
|---|---|---|---|
| train | 5,861 | 5,854 | 00131, 01072, 01076, 01960, 02419, 04280, 05052 |
| val | 5,860 | 5,853 | 05875, 06402, 08352, 08561, 09528, 09976, 11479 |

## Claim status

**Allowed:** "On a frozen 1,000-pair DIOR val sample (500 present / 500 absent, 50 per class, seed 17, third-party mirror pinned by sha256), Grounding DINO Swin-T at box threshold 0.35: exact 32.4%, MAE 1.53, within ±1 80.4%; always-0 baseline exact 50.0%, MAE 1.48. On present-class pairs: exact 49.2%. Absent-class pairs: false count on 84%."

**Not allowed:**
- Any claim that detector counting beats a trivial baseline.
- Any Cartosat or RISAT claim.
- Any claim about the resolution gate. It was not tested, because DIOR GSD is not in the VOC annotations.

This is a single seed, on val only.
