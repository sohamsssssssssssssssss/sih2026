# India land-cover promotion gate

Frozen before the candidate run. Reference recipe a5531188731d8cbefdd21a9a493775e04e7502c9, docs/hypotheses/T13b.md and results/t13b/REPORT.md on integration/spine-v1. WorldCover supplies map labels. The joint decision covers both single-sensor modes, s1 and s2.

recipe validated on held-out regions; shipped weights have no held-out score.

## G1 — recipe accuracy

From the original BEN-19 transfer checkpoint, retrain only the full arm with train seed 1. Use the verified T13b WorldCover manifest and every original India patch. Fold A adapts on Punjab, Kerala and Assam and tests on Delhi, Sundarbans and Rajasthan. Fold B reverses these groups. Use the original per-region 80/20 train/validation split, split seed lc.SEED=17, train-only normalization, augmentation, AdamW (LR 0.0001, weight decay 0.0001), cosine schedule, batch size 128 and 15 epochs. Restore the first epoch attaining maximum validation common-class macro AP (eligible with at least one validation positive). Run all epochs.

Pool the raw held-out full-arm predictions across folds, exactly once per India patch. Compute T13b five-common-class macro AP with at least 10 positives per eligible class. Both inclusive floors must pass: s1 >= 0.689087241566672 and s2 >= 0.6588597201349471. Each is the smallest corresponding full-arm pooled AP across T13b seeds 1–3 minus 0.01.

Train one fresh full-arm candidate per mode from its Europe checkpoint, combining the two folds' adaptation training subsets and, separately, their validation subsets across all six regions. Keep the same normalization and epoch-selection rules. Save weights, preprocessing, selected epoch, provenance and validation mAP. Record test mAP as null. Fold results belong to the recipe.

## G2 — calibration transfer

Use only seed 1 full-arm out-of-fold predictions and labels from adapt-run-3, separately for each mode. For each of the five classes fit a Platt sigmoid to the logit of the raw probability (clip to [0.000001, 0.999999]); use sklearn LogisticRegression with C=1, lbfgs, intercept, max_iter=1000. Fit on fold A OOF rows only. Freeze these five calibrators per mode and report raw and calibrated ECE on the disjoint fold B OOF rows. Use 10 equal-width bins on [0,1] (left-inclusive with 1 in the last bin), sum count-weighted absolute differences between mean probability and positive frequency, then average equally across all five classes. Report class values and macro values. Require strictly lower macro ECE in each mode. A missing class outcome in fit data or a failed fit fails the gate. Apply these fold-model calibrators unchanged to the final retrained models; this measures OOF calibration transfer across regions and gives no final-weight calibration guarantee.

## G3 — final-weight sanity checks

Run final s2 weights with their saved preprocessing and frozen G2 calibrators on data/fresh-backups/ludhiana-s2.tif (S2B_43REQ_20261028_0_L2A) and tests/fixtures/punjab-s2.tif from integration. Require built in the top two calibrated scores on Ludhiana and cropland in the top two on Punjab. Rank descending, breaking ties by T13b common-class order. Each is a one-scene sanity check with no population-level evidential claim.

Promotion requires G1, G2 and G3 together. Missing evidence or any failure ships nothing; the Europe checkpoints stay in service. No thresholds, seeds, splits, fixtures, methods or decision rules may change after observing results.
