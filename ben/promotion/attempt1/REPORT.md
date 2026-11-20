# India land-cover promotion

recipe validated on held-out regions; shipped weights have no held-out score

Gate registration `4b4ee12`. Candidate code `4b1d57c`.

| Gate | Evidence | Decision |
|---|---|---|
| G1 s1 | Recipe OOF mAP 0.7172067142770822, floor 0.689087241566672 | PASS |
| G2 s1 | Fold B macro ECE 0.12439133 → 0.17925145 | FAIL |
| G1 s2 | Recipe OOF mAP 0.6884732961958172, floor 0.6588597201349471 | PASS |
| G2 s2 | Fold B macro ECE 0.17777859 → 0.18661885 | FAIL |
| G3 Ludhiana | water 0.85795853, shrub_grass 0.16065476, trees 0.06944927 | FAIL |
| G3 Punjab | cropland 0.99950725, water 0.04120365, trees 0.01771585 | PASS |

**Joint decision: FAIL. Retain the Europe checkpoints.**

G3 uses two one-scene sanity checks. G2 fits each class on 900 seed-1 fold A OOF rows and evaluates on 900 distinct fold B rows. These calibrators transfer from the fold models to the final weights.

| Mode | Class | Raw ECE | Calibrated ECE |
|---|---|---:|---:|
| s1 | trees | 0.17148621 | 0.34097547 |
| s1 | shrub_grass | 0.17671411 | 0.10814143 |
| s1 | cropland | 0.11249902 | 0.17641676 |
| s1 | built | 0.13143497 | 0.16600996 |
| s1 | water | 0.02982235 | 0.10471364 |
| s2 | trees | 0.27809596 | 0.17525744 |
| s2 | shrub_grass | 0.15461319 | 0.12996483 |
| s2 | cropland | 0.29977000 | 0.36042995 |
| s2 | built | 0.12497778 | 0.20254590 |
| s2 | water | 0.03143603 | 0.06489612 |

## Verified SHA-256

- Fit NPZ: `1c7ba3770e063ff05f3b2d33b11c345657dc9e52c419457d049697d9a97ec15e`
- ben-landcover-india-s1.pt: `66e99bb7a572843e269c940e0c3e92647d513ca6adb11d3dfb846e399af5a72d`
- ben-landcover-india-s2.pt: `26e0cdfb7dd40abc93d025b4bd3db2ad6fcf0678e6b7385d218fee3f0675bdf9`
