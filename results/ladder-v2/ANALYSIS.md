# LoveDA resolution ladder v2

[SIH26167 Ladder Qwen Full](https://www.kaggle.com/code/sohamssssss/sih26167-ladder-qwen-full) produced the 2,000 predictions on a Tesla T4 on 2 September 2026 using `growinfame/loveda`. Its downloaded output includes the cloned repository ref `450cf04df4353e6b8c35655ae631420048c25594`. The notebook uses the generator's default seed 26167 and its original `results_ladder.json` scored 0.4040. All prediction payloads and tile IDs match the [committed v2 file](../../outputs/results_ladder_qwen_full_v2.json) in order. Commit `daab09b` applied the shared matcher and recorded 0.4885 without another inference run. The [later stratified report](../qwen2.5vl-3b__ladder__rescored__20260904.json) reused those predictions. The original log recorded 1,550.9 s wall time, with roughly 1,169 s for evaluation and reporting.

The accuracy column uses the stored correctness flags. The majority baseline predicts the most frequent gold answer in each task. Building yes-rate counts normalized `yes`, `yes, 1` and `1`. The percentile intervals resample the exact same 200 tile indices across all rungs and tasks in each of 2,000 bootstrap draws (NumPy seed 17). Differences are native minus the indicated rung. Combined accuracy equally weights both questions per tile.

| GSD (m) | Task | Accuracy [95% CI] | Yes-rate | Majority | Native minus rung [95% CI] |
|---:|---|---:|---:|---|---:|
| 0.3 | Dominant | 0.335 [0.270, 0.400] | — | 0.450 (agriculture) | — |
| 0.3 | Building | 0.790 [0.730, 0.845] | 0.640 | 0.700 (yes) | — |
| 0.3 | Combined | 0.5625 [0.520, 0.603] | — | — | — |
| 1 | Dominant | 0.290 [0.225, 0.350] | — | 0.450 (agriculture) | +0.045 [+0.005, +0.085] |
| 1 | Building | 0.720 [0.660, 0.785] | 0.500 | 0.700 (yes) | +0.070 [-0.005, +0.145] |
| 1 | Combined | 0.505 [0.460, 0.552] | — | — | +0.058 [+0.015, +0.098] |
| 2 | Dominant | 0.275 [0.210, 0.335] | — | 0.450 (agriculture) | +0.060 [+0.025, +0.100] |
| 2 | Building | 0.575 [0.510, 0.640] | 0.315 | 0.700 (yes) | +0.215 [+0.130, +0.300] |
| 2 | Combined | 0.425 [0.380, 0.468] | — | — | +0.138 [+0.092, +0.185] |
| 5 | Dominant | 0.245 [0.185, 0.305] | — | 0.450 (agriculture) | +0.090 [+0.050, +0.135] |
| 5 | Building | 0.685 [0.620, 0.745] | 0.970 | 0.700 (yes) | +0.105 [+0.030, +0.180] |
| 5 | Combined | 0.465 [0.425, 0.505] | — | — | +0.098 [+0.058, +0.138] |
| 10 | Dominant | 0.270 [0.205, 0.330] | — | 0.450 (agriculture) | +0.065 [+0.030, +0.105] |
| 10 | Building | 0.700 [0.635, 0.760] | 1.000 | 0.700 (yes) | +0.090 [+0.015, +0.170] |
| 10 | Combined | 0.485 [0.445, 0.527] | — | — | +0.077 [+0.040, +0.115] |

| GSD (m) | Dominant class answer distribution (count out of 200) |
|---:|---|
| 0.3 | forest 102, field 39, fields 25, water 9, agriculture 6, residential 6, plants 4, building 2, buildings 2, roads 2, desert 1, 2 1, grassland 1 |
| 1 | forest 130, fields 25, urban 22, residential 5, field 5, agriculture 4, buildings 3, plants 3, desert 1, grassland 1, farms 1 |
| 2 | forest 163, urban 28, field 3, residential 2, agriculture 1, water 1, grass 1, desert 1 |
| 5 | forest 167, urban 20, grassland 8, desert 4, grass 1 |
| 10 | forest 191, urban 9 |

## Matching audit

Among 1023 stored wrong answers, 91 are plausible matching misses under a narrow review convention: agriculture ← field (41), agriculture ← fields (48), building ← buildings (1), agriculture ← farms (1). All are dominant-class answers. Wrong `Yes, 1` responses all have gold `no`. A looser convention adds 13 related-label cases: barren ← desert (2), building ← residential (1), building ← urban (10). A scene type or specific subclass does not uniquely identify the mask class, so those are listed separately. Stored scores remain unchanged.

## Reading the results

- The native rung gives the highest pooled score. Its paired advantage is positive against all coarser rungs.
- Building accuracy recovers beyond 2 m as predictions collapse toward yes.
- The 5 m building score is just below the always-yes baseline, and the 10 m score equals it.
- Forest dominates the open answers at every rung and greatly exceeds its frequency as the gold majority class.
- The fixed LoveDA cohort is a resolution proxy. These scores do not establish performance on Indian imagery or actual satellite sensors.
