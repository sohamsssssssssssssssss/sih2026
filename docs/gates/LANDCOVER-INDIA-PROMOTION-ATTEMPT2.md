# India land-cover promotion, attempt 2 (amendment)

Written on 20 Nov 2026, **after** seeing the attempt-1 result. Attempt 1 failed G2 and G3 ([report](../../results/landcover-india-promotion/REPORT.md), commit `5e3fcee`; gate [LANDCOVER-INDIA-PROMOTION.md](LANDCOVER-INDIA-PROMOTION.md), commit `4b4ee12`). The attempt-1 result stays recorded as **FAIL**. Neither attempt-1 file is edited.

## Why an amendment

Attempt 1 fitted a separate Platt sigmoid per class (two parameters per class, ten per mode) on fold A out-of-fold rows. On fold B, macro ECE rose in both modes (s1 0.12439133 → 0.17925145, s2 0.17777859 → 0.18661885). We read this as the per-class calibrators overfitting the fit regions. Attempt 2 pre-registers a simpler calibrator with one parameter per mode.

Attempt 1's G3 failure on Ludhiana has been traced to an input preprocessing bug in the app (S2 offset applied twice). That fix is being made in integration and is outside this amendment.

## G1: carried over

Unchanged and already PASS (s1 0.7172067142770822 ≥ 0.689087241566672; s2 0.6884732961958172 ≥ 0.6588597201349471). Not rerun.

## G2': temperature scaling

- **Data**: the same file as attempt 1, `adapt-run-3/seed-1/ben-adapt-predictions-seed-1.npz`, sha256 `1c7ba3770e063ff05f3b2d33b11c345657dc9e52c419457d049697d9a97ec15e`. Rows: arm `full`, per mode `s1` and `s2`, fold A (900 rows) and fold B (900 rows), five classes (trees, shrub_grass, cropland, built, water). `final-run-1/ben-adapt-predictions-seed-1.npz` (sha256 `b141bf5e…`) is not used.
- **Logits**: z = logit(clip(p, 0.000001, 0.999999)) from the stored probabilities, the same clip as attempt 1.
- **Calibrator**: one scalar T > 0 per mode, shared by all five classes. Calibrated probability = sigmoid(z / T).
- **Fit**: T minimises the mean binary cross-entropy (NLL) over all fit rows × five classes. Optimiser: `scipy.optimize.minimize_scalar`, method `bounded`, on log T in [log 0.05, log 20]. If the optimum lands on a bound, it is reported and still used.
- **Directions**: fit on fold A → evaluate on fold B, and fit on fold B → evaluate on fold A.
- **ECE**: as in attempt 1. Per class: 10 equal-width bins on [0, 1], left-inclusive, with 1 in the last bin; sum over bins of |Σ(p − y)| / n. Macro ECE is the unweighted mean over the five classes.
- **PASS rule**: calibrated macro ECE is strictly lower than raw macro ECE on the evaluation fold, in **both** directions for **both** modes (4 of 4 comparisons). Anything else is FAIL.
- **Shipped T**: if G2' passes, the shipped T per mode is fitted the same way on all 1,800 OOF rows of that mode.
- **Context only, not gated**: per-class ECE and positive counts for attempt 1 (Platt, A → B) and attempt 2 (both directions).
- **Known limit, same as attempt 1**: T is fitted on fold-model OOF predictions and transferred to the final weights, which have no held-out score.

## G3: same thresholds as attempt 1

Built must rank in the top two calibrated scores on Ludhiana, and cropland in the top two on Punjab. G3 is rerun on the corrected (v2) cached scenes once the offset fix lands in integration. **It is not run in this attempt.**

## Fixed in advance

No method, data, optimiser bounds, binning, threshold or decision rule may change after results are seen. If G2' fails, the India model is not shipped with calibrated scores. Promotion still requires G1, G2' and G3 together.
