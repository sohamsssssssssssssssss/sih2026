# India land-cover promotion: decision

**Decision: NOT SHIPPED.** `/ask` keeps the Europe (BigEarthNet) checkpoints. No further calibration attempts. Decided 20 Nov 2026.

## What was tested

| Gate | Attempt 1 (gate `4b4ee12`, result `5e3fcee`) | Attempt 2 (gate `db6e127`, result `ce8449e`) |
|---|---|---|
| G1 recipe accuracy (out-of-fold India mAP) | PASS: s1 0.7172 ≥ 0.6891, s2 0.6885 ≥ 0.6589 | carried over: PASS |
| G2 calibration (Platt, one sigmoid per class, fit fold A, test fold B) | **FAIL**: macro ECE rose, s1 0.1244 → 0.1793, s2 0.1778 → 0.1866 | — |
| G2' calibration (one temperature per mode, both directions) | — | **FAIL**: s1 A→B 0.1244 → 0.1466 (worse), B→A 0.1219 → 0.1112; s2 A→B 0.1778 → 0.1332, B→A 0.1586 → 0.1918 (worse) |
| G3 one-scene sanity (target class in top two) | **FAIL** Ludhiana (water 0.858 first); PASS Punjab (cropland 0.9995) | not run |
| G3 on corrected inputs (record only; promotion already failed on G2) | Ludhiana v2: built 0.99998, trees 0.502, cropland 0.197: PASS. Punjab: cropland 0.998, trees 0.013, built 0.005: PASS. Raw scores, [g3-corrected.json](g3-corrected.json) | |

## The input-bug finding

The attempt-1 Ludhiana failure was not the model. The app applied the Sentinel-2 BOA offset (−1000) twice when fetching scenes, so the model saw darkened reflectances. Integration fixed this in `84d7f57` and regenerated the cached scenes in `404039f`. With the corrected scene, the India model ranks built first on Ludhiana (0.99998). Punjab comes from an uploaded fixture, not the fetch path, so it was unaffected. G3 now passes, but promotion needs G1, G2 and G3 together, so this does not change the decision.

## Why it is not shipped

The model ranks classes well on held-out Indian regions (G1), but its scores cannot be turned into trustworthy probabilities with the data we have.

- **Calibration does not transfer between Indian regions.** Fold A (adapt Punjab, Kerala, Assam; test Delhi, Sundarbans, Rajasthan) and fold B (the reverse) are miscalibrated in opposite directions. A correction fitted on one fold makes the other worse: per-class Platt failed one way, and a single temperature failed in one direction for each mode.
- **Class counts differ a lot between folds.** Positives per 900 rows, fold A vs B: water 216 vs 105, shrub/grass 276 vs 138, trees 458 vs 607, cropland 400 vs 530. A calibrator learns the base rates of the regions it was fitted on.
- **Only six regions exist**, so there is no third set of regions to settle which fold is typical.

Shipping uncalibrated scores from a new model gains nothing over the Europe model, which also returns uncalibrated scores with `confidence: null`, and it adds an unvalidated change. The gates were fixed in advance and stopped the promotion as designed.

## What would change this

- More Indian regions with labels, so calibration can be fitted and tested on regions that neither fold has seen.
- A per-region calibration study that shows whether miscalibration follows region, class prevalence or season, before any new gate is written.
- Any new attempt needs a new gate registered before results, as attempts 1 and 2 had.

Sources: [attempt 1 report](REPORT.md), [attempt 2 report](attempt2/REPORT.md), [attempt 1 gate](../../docs/gates/LANDCOVER-INDIA-PROMOTION.md), [attempt 2 gate](../../docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md), attempt-1 code in [ben/promotion/attempt1/](../../ben/promotion/attempt1/).
