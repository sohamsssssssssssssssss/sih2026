# India land-cover promotion, attempt 3 (new data)

Written on 20 Nov 2026, **after** attempts 1 and 2 failed, and before any attempt-3 data is built or any model is trained. Attempts 1 and 2 stay recorded as **FAIL** ([decision](../../results/landcover-india-promotion/DECISION.md); gates [1](LANDCOVER-INDIA-PROMOTION.md) and [2](LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md)). Neither earlier gate or report is edited.

Attempt 3 uses **new data**: ten more Indian regions. The hypothesis is that a calibrator fitted on several varied Indian regions transfers to regions it has not seen. With only two folds of three regions, attempts 1 and 2 could not test this.

## 1. Data

**Existing pack, unchanged.** WorldCover pack MANIFEST sha256 `5cc82c713ad76a3286afd4ad3eb66f33da3fb7ff365ab23a6371dc3ba999c40d`. Rows for the six India regions (Punjab, Delhi, Kerala, Sundarbans, Assam, Rajasthan) and the four Europe controls keep their exact patches.

**New regions.** The ten regions below are added using the same method as `~/ben-work/wc/build_wc.py` (sha256 `9f19069b…`):
- ESA WorldCover 2021 v200 map as labels.
- The WorldCover 2021 S1 VV/VH and S2 RGBNIR annual composites as inputs.
- Each source reprojected to local UTM at 10 m (map nearest; S1/S2 bilinear).
- A 0.5° box around the centre, cut into 120 × 120 patches. Patches with any nodata are rejected.
- 300 patches per region drawn with `default_rng(17)`.
- Coarse labels present at fraction ≥ 0.05.

If a region yields fewer than 300 valid patches, it is replaced by nothing: the run stops and is recorded as not run.

| Region | Centre (lat, lon) | Box (W, S, E, N) |
|---|---|---|
| hyderabad | 17.5, 78.5 | 78.25, 17.25, 78.75, 17.75 |
| marathwada | 19.5, 75.5 | 75.25, 19.25, 75.75, 19.75 |
| kodagu | 12.5, 75.5 | 75.25, 12.25, 75.75, 12.75 |
| garhwal | 30.5, 78.5 | 78.25, 30.25, 78.75, 30.75 |
| gangetic | 25.5, 85.5 | 85.25, 25.25, 85.75, 25.75 |
| kutch | 23.5, 69.5 | 69.25, 23.25, 69.75, 23.75 |
| bastar | 19.5, 81.5 | 81.25, 19.25, 81.75, 19.75 |
| chilika | 19.5, 85.5 | 85.25, 19.25, 85.75, 19.75 |
| mizoram | 23.5, 92.5 | 92.25, 23.25, 92.75, 23.75 |
| cauvery | 10.5, 79.5 | 79.25, 10.25, 79.75, 10.75 |

None of these boxes overlaps an existing region or contains the G3 Ludhiana scene (30.90 N, 75.86 E).

## 2. Groups

There are K = 5 region groups. Attempts 1 and 2 keep their original folds:
- **A** = Delhi, Sundarbans, Rajasthan (attempt-1 fold A test regions).
- **B** = Punjab, Kerala, Assam (attempt-1 fold B test regions).
- **C, D, E** = the ten new regions, split into groups of 4, 3 and 3.

**Assignment rule.** This rule uses WorldCover labels only. It runs after the pack is built and before any training or prediction.
1. Enumerate every split of the ten new regions into unlabelled sets of sizes 4, 3 and 3.
2. Keep only splits where every group A–E has at least 50 positives for each of the five classes (trees, shrub_grass, cropland, built, water).
3. Choose the split that minimises the largest absolute gap between any group's class prevalence and the pooled 16-region prevalence, over all groups and classes. Break ties by the smallest sum of squared gaps, then by sorted region names.
4. Name the sets C, D and E in alphabetical order of their first region.

If no split passes step 2, attempt 3 stops before training and is recorded as **not run**.

The assignment, per-group positive counts and new MANIFEST sha256 are committed before training starts.

## 3. Training (unchanged recipe)

This is the attempt-1 G1 full-arm recipe, run per mode (s1, s2):
- Start from the Europe BigEarthNet checkpoint.
- Train seed 1.
- Per-region 80/20 train/validation split with `lc.SEED = 17`.
- Train-only normalisation and augmentation.
- AdamW (LR 0.0001, weight decay 0.0001), cosine schedule, batch size 128, 15 epochs.
- Restore the first epoch with the maximum validation common-class macro AP.

**Leave-one-group-out.** For each held-out group g, train on the train subsets of the other four groups (validation on their validation subsets). Predict every patch of g exactly once. Europe regions are not used.

## 4. Gates (all fixed in advance)

**G1, accuracy.** Pool the held-out predictions of all five groups, so every India patch is counted once. Score with the T13b five-common-class macro AP (classes with at least 10 positives). Pass if s1 ≥ 0.689087241566672 **and** s2 ≥ 0.6588597201349471. These are the attempt-1 floors, unchanged. Per-group mAP is reported but not gated.

**G2, calibration transfer.**
- **Method:** per-class logit offset. Each class c gets one intercept b_c, and calibrated p = sigmoid(logit(clip(p, 0.000001, 0.999999)) + b_c). That is five parameters per mode, with the slope fixed at 1.
- **Fit:** b_c minimises mean binary cross-entropy, using `scipy.optimize.minimize_scalar`, method `bounded`, on [−10, 10].
- **Why this method:** attempt 2 showed the folds' scores were biased in opposite directions (mean raw score minus observed frequency +7.94 / −2.64 points for s1, +14.82 / −3.02 for s2), with very different class base rates. A per-class offset is the smallest correction that targets a base-rate bias. A single temperature cannot shift a bias, and per-class Platt doubles the parameters that overfit in attempt 1. With five groups, the offset is fitted on four varied groups, which is the hypothesis under test.
- **Procedure:** for each held-out group g and each mode, fit the offsets on the pooled held-out predictions of the other four groups. Each of those predictions came from a model that did not see its own patches. Then evaluate on g's held-out predictions.
- **ECE:** as in attempts 1 and 2. Per class, 10 equal-width bins on [0, 1], left-inclusive, with 1 in the last bin; sum over bins of |Σ(p − y)| / n. Macro ECE is the unweighted mean over the five classes.
- **Pass:** calibrated macro ECE is strictly lower than raw macro ECE for **every** held-out group in **both** modes (10 of 10 comparisons). Anything else is FAIL.

**G3, sanity.** This is the same rule as attempts 1 and 2. Run the final s2 weights with the final calibrator on `data/fresh-backups/ludhiana-s2-v2.tif` and `tests/fixtures/punjab-s2.tif`, from integration at or after `404039f`, through the `/ask` preprocessing. Built must rank in the top two calibrated scores on Ludhiana, and cropland in the top two on Punjab. Known limit: the Punjab fixture comes from the Punjab training region, so it checks plumbing, not generalisation.

**Context only, not gated.**
- Per-group mAP, per-class ECE and positive counts.
- The same offset calibrator fitted A→B and B→A on the attempt-1 file `adapt-run-3/seed-1/ben-adapt-predictions-seed-1.npz` (sha256 `1c7ba377…`), for comparison with attempts 1 and 2.

## 5. Final weights

The final s1 and s2 weights are trained on all five groups' train subsets with the same recipe. The final offsets are fitted on all held-out predictions of all five groups. **No held-out score is claimed for the final weights or calibrator.** Promotion requires G1, G2 and G3 together. Otherwise nothing ships, and `/ask` keeps the Europe checkpoints.

## 6. Fixed in advance

No region, box, seed, split, group rule, recipe, calibrator, bound, binning, threshold or decision rule may change after any attempt-3 data, label count or result is seen. The only exception is the label-only group assignment in section 2, which is computed by the stated rule. A build or training failure is recorded as not run, never retried with changed settings. If G2 fails, the India model is not shipped with calibrated scores, and no attempt 4 follows on this data.

## Amendment 1 (written before any attempt-3 data was built): G4 single-date check

**Purpose.** Test whether the leave-one-group-out weights improve on the shipped Europe weights on single-date inputs like those used by Ask.

- **Data.** For every region in the held-out group of each fold, use the pack's 120 × 120, 10 m UTM patch grids and their unchanged WorldCover 2021 map labels. Use the read-only integration checkout's app fetch and preprocessing code. For Sentinel-2, use Earth Search L2A acquisitions from 1 January through 30 April 2021 with scene-level cloud cover strictly below 10%, including the app's BOA-offset handling. For Sentinel-1, use Planetary Computer dual-polarisation RTC acquisitions from the same dates and apply the app's `10 * log10` conversion once. Align imagery to the exact pack grid and record the integration commit, item IDs, acquisition dates and file SHA-256s. Target at least 100 usable patches per region and record skipped patches and their reasons.
- **Comparison.** In each mode use identical held-out patches and labels for both models. Compare the India weights trained with that group held out to the shipped Europe checkpoint in `~/ben-work/transfer-run-1/ben-landcover-{s1,s2}-checkpoint.pt`. Preprocess each model's inputs with its own frozen training statistics via the app's `make_batch` path.
- **Decision.** Pool the five held-out groups once per mode. Use the G1 common-class macro mAP definition on this set. Compute India minus Europe with a paired bootstrap of patches using 2,000 replicates and seed 17. Use a two-sided 95% percentile interval. Pass only if the point difference is positive and the interval's lower endpoint is strictly above zero in both s1 and s2. Report per-group scores and differences for context.
- **Promotion.** G4 is required in addition to G1–G3. A missing or failed G4 cannot support promotion. The data and decision rules above are fixed before generating any attempt-3 pack or label counts.
