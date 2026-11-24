# India land-cover promotion: decision

**Decision: NOT SHIPPED.** `/ask` keeps the Europe (BigEarthNet) checkpoints. No further calibration attempts. Decided 20 Nov 2026.

## What was tested

| Gate | Attempt 1 (gate `4b4ee12`, result `5e3fcee`) | Attempt 2 (gate `db6e127`, result `ce8449e`) |
|---|---|---|
| G1 recipe accuracy (out-of-fold India mAP) | PASS: s1 0.7172 ≥ 0.6891, s2 0.6885 ≥ 0.6589 | carried over: PASS |
| G2 calibration (Platt, one sigmoid per class, fit fold A, test fold B) | **FAIL**: macro ECE rose, s1 0.1244 → 0.1793, s2 0.1778 → 0.1866 | — |
| G2' calibration (one temperature per mode, both directions) | — | **FAIL**: s1 A→B 0.1244 → 0.1466 (worse), B→A 0.1219 → 0.1112; s2 A→B 0.1778 → 0.1332, B→A 0.1586 → 0.1918 (worse) |
| G3 one-scene sanity (target class in top two) | **FAIL** Ludhiana (water 0.858 first); PASS Punjab (cropland 0.9995) | not run |

### G3 on corrected inputs (record only; promotion already failed on G2)

Integration fixed the duplicate −1,000 DN Sentinel-2 BOA offset in `84d7f57` and regenerated the Ludhiana cache in `404039f`. Both final checkpoint SHA-256 hashes matched. The rerun used the `/ask` scene loader and model preprocessing. The same top-two checks also passed with the frozen attempt-1 calibrator.

| Scene | Raw top three | Original threshold |
|---|---|---|
| Ludhiana v2 | Built areas 0.999985, trees 0.501719, cropland 0.197492 | Built areas in top two · **PASS** |
| Punjab | Cropland 0.997638, trees 0.013360, built areas 0.004779 | Cropland in top two · **PASS** |

## Why it is not shipped

The model ranks classes well on held-out Indian regions (G1), but its scores cannot be turned into trustworthy probabilities with the data we have.

- **Calibration does not transfer between Indian regions.** Fold A (adapt Punjab, Kerala, Assam; test Delhi, Sundarbans, Rajasthan) and fold B (the reverse) have opposite biases. Average raw scores minus observed class frequencies were +7.94 and −2.64 percentage points for S1, and +14.82 and −3.02 for S2, in A and B respectively. A correction fitted on one fold makes the other worse: per-class Platt failed one way, and a single temperature failed in one direction for each mode.
- **Class counts differ a lot between folds.** Positives per 900 rows, fold A vs B: water 216 vs 105, shrub/grass 276 vs 138, trees 458 vs 607, cropland 400 vs 530. A calibrator learns the base rates of the regions it was fitted on.
- **Only six regions exist**, so there is no third set of regions to settle which fold is typical.

The corrected scene resolves the Ludhiana sanity check. Both calibration failures stand because they used held-out recipe predictions independently of the app.

## What would change this

- More Indian regions with labels, so calibration can be fitted and tested on regions that neither fold has seen.
- A per-region calibration study that shows whether miscalibration follows region, class prevalence or season, before any new gate is written.
- Any new attempt needs a new gate registered before results, as attempts 1 and 2 had.

Sources: [attempt 1 report](REPORT.md), [attempt 2 report](attempt2/REPORT.md), [attempt 1 gate](../../docs/gates/LANDCOVER-INDIA-PROMOTION.md), [attempt 2 gate](../../docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md), attempt-1 code in [ben/promotion/attempt1/](../../ben/promotion/attempt1/).

## Attempt 3 decision (24 Nov 2026)

**Decision: NOT PROMOTED.** Ask continues to use the Europe BigEarthNet checkpoints.

Attempt 3 passed G1 on held-out annual composites and passed G3 on both sanity scenes. G2 and G4 failed their joint requirements. Calibration lowered macro ECE in 2 of 10 comparisons, against a requirement of 10 of 10. On paired single-date inputs, India minus Europe was −0.0406 mAP for S1 (95% CI [−0.0678, −0.0170]) and +0.1471 for S2 (95% CI [+0.1262, +0.1680]). Both modes needed a positive delta with the entire interval above zero.

The scoring thresholds stayed fixed. The pre-data amendment that registered G4 was already part of the frozen protocol. No amendments were applied during scoring.

- **May say:** India S2 beat Europe by +0.147 mAP on the registered single-date pairs. Calibration did not transfer across the held-out groups.
- **May not say:** The India model is better overall, shipped, or calibrated.

Sources: [attempt 3 report](attempt3/REPORT.md#attempt-3-scored-results-24-nov-2026) and [preregistered gates](../../docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT3.md).
