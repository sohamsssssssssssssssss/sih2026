# India land-cover promotion, attempt 2 (G2' only)

Gate pre-registration `db6e127` (docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md). Attempt 1: [REPORT.md](../REPORT.md), recorded FAIL, unchanged.

| Gate | Attempt 1 (`5e3fcee`) | Attempt 2 |
|---|---|---|
| G1 | PASS (s1 0.71720671, s2 0.68847330) | carried over: PASS |
| G2 s1 | Platt per class, A→B macro ECE 0.12439133 → 0.17925145: FAIL | Temperature: A->B: 0.12439133 → 0.14657851 (T=1.893129); B->A: 0.12186621 → 0.11119078 (T=1.444374): FAIL |
| G2 s2 | Platt per class, A→B macro ECE 0.17777859 → 0.18661885: FAIL | Temperature: A->B: 0.17777859 → 0.13315685 (T=2.408143); B->A: 0.15860850 → 0.19184210 (T=2.719022): FAIL |
| G3 | Ludhiana FAIL, Punjab PASS (input bug since found) | pending corrected scenes |

**G2' verdict: FAIL.** The India model is not shipped with calibrated scores.

## Temperatures

| Mode | T fit A | T fit B | T fit on all 1,800 rows (ships only if G2' passes) |
|---|---:|---:|---:|
| s1 | 1.893129 | 1.444374 | 1.656451 |
| s2 | 2.408143 | 2.719022 | 2.599635 |

T > 1 softens the scores and T < 1 sharpens them. One T per mode cannot change the class ranking.

## Per-class ECE (context only, not gated)

| Mode | Class | Pos A | Pos B | Attempt 1 raw B | Attempt 1 Platt B | Attempt 2 raw B | Attempt 2 temp A→B | Attempt 2 raw A | Attempt 2 temp B→A |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 | trees | 458 | 607 | 0.17148621 | 0.34097547 | 0.17148621 | 0.20850662 | 0.18096456 | 0.15470470 |
| s1 | shrub_grass | 276 | 138 | 0.17671411 | 0.10814143 | 0.17671411 | 0.16200009 | 0.11496514 | 0.07145421 |
| s1 | cropland | 400 | 530 | 0.11249902 | 0.17641676 | 0.11249902 | 0.14918425 | 0.09362906 | 0.07362720 |
| s1 | built | 253 | 279 | 0.13143497 | 0.16600996 | 0.13143497 | 0.10518195 | 0.07704736 | 0.12241328 |
| s1 | water | 216 | 105 | 0.02982235 | 0.10471364 | 0.02982235 | 0.10801966 | 0.14272492 | 0.13375450 |
| s2 | trees | 458 | 607 | 0.27809596 | 0.17525744 | 0.27809596 | 0.15887249 | 0.23322317 | 0.18208964 |
| s2 | shrub_grass | 276 | 138 | 0.15461319 | 0.12996483 | 0.15461319 | 0.15771795 | 0.05069913 | 0.13092841 |
| s2 | cropland | 400 | 530 | 0.29977000 | 0.36042995 | 0.29977000 | 0.24248762 | 0.06728211 | 0.13528110 |
| s2 | built | 253 | 279 | 0.12497778 | 0.20254590 | 0.12497778 | 0.07142251 | 0.12822259 | 0.23961720 |
| s2 | water | 216 | 105 | 0.03143603 | 0.06489612 | 0.03143603 | 0.03528368 | 0.31361548 | 0.27129416 |

Each fold has 900 rows per mode. T is fitted on fold-model OOF predictions and transferred to the final weights, which have no held-out score.

## Verified SHA-256

- Fit NPZ `ben-adapt-predictions-seed-1.npz`: `1c7ba3770e063ff05f3b2d33b11c345657dc9e52c419457d049697d9a97ec15e`
