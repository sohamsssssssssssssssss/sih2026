# Offline DIOR presence gate

Verdict: **does not beat always-0** under the required report-half exact CI rule.

Source SHA-256: `f77636624bd4824404b327f1168e40a3a450f74e7ca59a107fb3fb6375387df2`. Only this committed evaluation file was used. Python stdlib + numpy; no downloads, GPU or model run.

Structure: top-level metadata, metrics, baselines and threshold_sweep. Each threshold contains overall/per_class summaries and 1,000 records. Records store image_id, class, image, expected, hashes, prompt, prediction and evidence. Box evidence stores confidence and coordinates; detection_count evidence stores thresholds. A/B/C are possible. Scores are retained post-NMS box scores, not raw model proposals; scores below 0.25 are unavailable.

Split: 500 pairs per half, each 250 present / 250 absent; 471 select images / 471 report images, no image overlap. Image groups stratified by (present, absent) pair counts; near-half quotas chosen for exact 250/250 balance, then closest image half and quota balance; lexicographic tie break; shuffle sorted image IDs within strata with random.Random(17).

Selection: maximum select exact; ties use A then B/C enumeration order, ascending thresholds and k/score. All candidates were specified before report scoring. B uses retained scores at its count threshold; C recounts retained boxes from threshold 0.25 (no new NMS). Recount equality checks: {'0.25': True, '0.3': True, '0.35': True, '0.4': True}.

Chosen gate: **B count=0.3 score=0.700**. Select exact: 0.498.

## Report half

| Method | Exact | MAE | Within-1 | Absent false count | Present exact |
|---|---:|---:|---:|---:|---:|
| B count=0.3 score=0.700 | 0.474 | 1.336 | 0.802 | 0.144 | 0.092 |
| always_zero | 0.500 | 1.332 | 0.788 | 0.000 | 0.000 |
| detector@0.35 | 0.342 | 1.358 | 0.820 | 0.824 | 0.508 |
| oracle_presence@0.35 | 0.754 | 0.846 | 0.898 | 0.000 | 0.508 |

Paired image bootstrap: 2000 reps, seed 17, percentile 95% CIs. Gate minus always-0 exact: -0.0260, CI [-0.058, 0.004048994437312792]; MAE: +0.0040, CI [-0.03968253968253968, 0.04618939467547607]. Negative MAE difference is improvement.

Oracle true presence @0.35 is a ceiling for removing absent-class errors while preserving detector counts; it does not fix missed/incorrect counts on present pairs and is not an implementable gate. Other count-threshold oracle metrics are recorded in JSON.

## Presence versus counting errors

Disjoint error counts (output presence, not latent gate decision):
- B count=0.3 score=0.700: absent false positives 36; present output-zero misses 222; wrong nonzero count on present pairs 5.
- detector@0.35: absent false positives 206; present output-zero misses 27; wrong nonzero count on present pairs 96.

## Limits

A fresh frozen sample is required before any claim. This split is exploratory reuse of already inspected evidence; bootstrap intervals condition on the selected gate and this split. Fresh-sample performance, deployment accuracy and localization correctness: not measured. Confidence gates use stored retained boxes only. No per-class tuning or model changes.

## Full 1,000 sweep — in-sample, optimistic

| Gate | Exact | MAE | Within-1 | Absent false count | Present exact | Select exact |
|---|---:|---:|---:|---:|---:|---:|
| always_zero (baseline) | 0.500 | 1.484 | 0.785 | 0.000 | 0.000 | 0.500 |
| detector@0.35 (baseline) | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| oracle_presence@0.35 (baseline) | 0.746 | 0.998 | 0.892 | 0.000 | 0.492 | 0.738 |
| A count=0.25 gate=0.25 k=1 | 0.194 | 2.080 | 0.554 | 0.958 | 0.346 | 0.192 |
| A count=0.25 gate=0.25 k=2 | 0.259 | 2.078 | 0.518 | 0.578 | 0.096 | 0.250 |
| A count=0.25 gate=0.3 k=1 | 0.207 | 2.063 | 0.563 | 0.926 | 0.340 | 0.202 |
| A count=0.25 gate=0.3 k=2 | 0.356 | 1.898 | 0.612 | 0.352 | 0.064 | 0.350 |
| A count=0.25 gate=0.35 k=1 | 0.242 | 1.979 | 0.596 | 0.844 | 0.328 | 0.230 |
| A count=0.25 gate=0.35 k=2 | 0.438 | 1.656 | 0.712 | 0.176 | 0.052 | 0.428 |
| A count=0.25 gate=0.4 k=1 | 0.273 | 1.935 | 0.620 | 0.748 | 0.294 | 0.262 |
| A count=0.25 gate=0.4 k=2 | 0.485 | 1.534 | 0.768 | 0.072 | 0.042 | 0.478 |
| A count=0.3 gate=0.3 k=1 | 0.268 | 1.698 | 0.708 | 0.926 | 0.462 | 0.252 |
| A count=0.3 gate=0.3 k=2 | 0.375 | 1.693 | 0.653 | 0.352 | 0.102 | 0.368 |
| A count=0.3 gate=0.35 k=1 | 0.294 | 1.678 | 0.711 | 0.844 | 0.432 | 0.274 |
| A count=0.3 gate=0.35 k=2 | 0.450 | 1.558 | 0.730 | 0.176 | 0.076 | 0.440 |
| A count=0.3 gate=0.4 k=1 | 0.321 | 1.679 | 0.712 | 0.748 | 0.390 | 0.300 |
| A count=0.3 gate=0.4 k=2 | 0.488 | 1.492 | 0.776 | 0.072 | 0.048 | 0.484 |
| A count=0.35 gate=0.35 k=1 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| A count=0.35 gate=0.35 k=2 | 0.450 | 1.518 | 0.745 | 0.176 | 0.076 | 0.436 |
| A count=0.35 gate=0.4 k=1 | 0.344 | 1.553 | 0.787 | 0.748 | 0.436 | 0.320 |
| A count=0.35 gate=0.4 k=2 | 0.489 | 1.473 | 0.782 | 0.072 | 0.050 | 0.484 |
| A count=0.4 gate=0.4 k=1 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| A count=0.4 gate=0.4 k=2 | 0.490 | 1.471 | 0.780 | 0.072 | 0.052 | 0.484 |
| B count=0.25 score=0.300 | 0.207 | 2.063 | 0.563 | 0.926 | 0.340 | 0.202 |
| C count=0.25 score=0.300 | 0.207 | 2.063 | 0.563 | 0.926 | 0.340 | 0.202 |
| B count=0.25 score=0.325 | 0.220 | 2.029 | 0.572 | 0.896 | 0.336 | 0.210 |
| C count=0.25 score=0.325 | 0.220 | 2.029 | 0.572 | 0.896 | 0.336 | 0.210 |
| B count=0.25 score=0.350 | 0.242 | 1.979 | 0.596 | 0.844 | 0.328 | 0.230 |
| C count=0.25 score=0.350 | 0.242 | 1.979 | 0.596 | 0.844 | 0.328 | 0.230 |
| B count=0.25 score=0.375 | 0.256 | 1.947 | 0.606 | 0.800 | 0.312 | 0.244 |
| C count=0.25 score=0.375 | 0.256 | 1.947 | 0.606 | 0.800 | 0.312 | 0.244 |
| B count=0.25 score=0.400 | 0.273 | 1.935 | 0.620 | 0.748 | 0.294 | 0.262 |
| C count=0.25 score=0.400 | 0.273 | 1.935 | 0.620 | 0.748 | 0.294 | 0.262 |
| B count=0.25 score=0.425 | 0.306 | 1.845 | 0.653 | 0.664 | 0.276 | 0.292 |
| C count=0.25 score=0.425 | 0.306 | 1.845 | 0.653 | 0.664 | 0.276 | 0.292 |
| B count=0.25 score=0.450 | 0.339 | 1.749 | 0.687 | 0.582 | 0.260 | 0.318 |
| C count=0.25 score=0.450 | 0.339 | 1.749 | 0.687 | 0.582 | 0.260 | 0.318 |
| B count=0.25 score=0.475 | 0.355 | 1.712 | 0.701 | 0.534 | 0.244 | 0.330 |
| C count=0.25 score=0.475 | 0.355 | 1.712 | 0.701 | 0.534 | 0.244 | 0.330 |
| B count=0.25 score=0.500 | 0.383 | 1.663 | 0.719 | 0.466 | 0.232 | 0.360 |
| C count=0.25 score=0.500 | 0.383 | 1.663 | 0.719 | 0.466 | 0.232 | 0.360 |
| B count=0.25 score=0.525 | 0.407 | 1.608 | 0.739 | 0.396 | 0.210 | 0.392 |
| C count=0.25 score=0.525 | 0.407 | 1.608 | 0.739 | 0.396 | 0.210 | 0.392 |
| B count=0.25 score=0.550 | 0.428 | 1.552 | 0.759 | 0.340 | 0.196 | 0.424 |
| C count=0.25 score=0.550 | 0.428 | 1.552 | 0.759 | 0.340 | 0.196 | 0.424 |
| B count=0.25 score=0.575 | 0.446 | 1.519 | 0.772 | 0.288 | 0.180 | 0.452 |
| C count=0.25 score=0.575 | 0.446 | 1.519 | 0.772 | 0.288 | 0.180 | 0.452 |
| B count=0.25 score=0.600 | 0.453 | 1.513 | 0.778 | 0.254 | 0.160 | 0.454 |
| C count=0.25 score=0.600 | 0.453 | 1.513 | 0.778 | 0.254 | 0.160 | 0.454 |
| B count=0.25 score=0.625 | 0.469 | 1.505 | 0.783 | 0.204 | 0.142 | 0.470 |
| C count=0.25 score=0.625 | 0.469 | 1.505 | 0.783 | 0.204 | 0.142 | 0.470 |
| B count=0.25 score=0.650 | 0.477 | 1.494 | 0.787 | 0.176 | 0.130 | 0.480 |
| C count=0.25 score=0.650 | 0.477 | 1.494 | 0.787 | 0.176 | 0.130 | 0.480 |
| B count=0.25 score=0.675 | 0.484 | 1.482 | 0.793 | 0.140 | 0.108 | 0.482 |
| C count=0.25 score=0.675 | 0.484 | 1.482 | 0.793 | 0.140 | 0.108 | 0.482 |
| B count=0.25 score=0.700 | 0.480 | 1.485 | 0.796 | 0.124 | 0.084 | 0.490 |
| C count=0.25 score=0.700 | 0.480 | 1.485 | 0.796 | 0.124 | 0.084 | 0.490 |
| B count=0.3 score=0.300 | 0.268 | 1.698 | 0.708 | 0.926 | 0.462 | 0.252 |
| C count=0.3 score=0.300 | 0.268 | 1.698 | 0.708 | 0.926 | 0.462 | 0.252 |
| B count=0.3 score=0.325 | 0.279 | 1.691 | 0.707 | 0.896 | 0.454 | 0.262 |
| C count=0.3 score=0.325 | 0.279 | 1.691 | 0.707 | 0.896 | 0.454 | 0.262 |
| B count=0.3 score=0.350 | 0.294 | 1.678 | 0.711 | 0.844 | 0.432 | 0.274 |
| C count=0.3 score=0.350 | 0.294 | 1.678 | 0.711 | 0.844 | 0.432 | 0.274 |
| B count=0.3 score=0.375 | 0.307 | 1.666 | 0.712 | 0.800 | 0.414 | 0.284 |
| C count=0.3 score=0.375 | 0.307 | 1.666 | 0.712 | 0.800 | 0.414 | 0.284 |
| B count=0.3 score=0.400 | 0.321 | 1.679 | 0.712 | 0.748 | 0.390 | 0.300 |
| C count=0.3 score=0.400 | 0.321 | 1.679 | 0.712 | 0.748 | 0.390 | 0.300 |
| B count=0.3 score=0.425 | 0.351 | 1.629 | 0.738 | 0.664 | 0.366 | 0.328 |
| C count=0.3 score=0.425 | 0.351 | 1.629 | 0.738 | 0.664 | 0.366 | 0.328 |
| B count=0.3 score=0.450 | 0.382 | 1.569 | 0.761 | 0.582 | 0.346 | 0.354 |
| C count=0.3 score=0.450 | 0.382 | 1.569 | 0.761 | 0.582 | 0.346 | 0.354 |
| B count=0.3 score=0.475 | 0.392 | 1.564 | 0.765 | 0.534 | 0.318 | 0.362 |
| C count=0.3 score=0.475 | 0.392 | 1.564 | 0.765 | 0.534 | 0.318 | 0.362 |
| B count=0.3 score=0.500 | 0.413 | 1.544 | 0.774 | 0.466 | 0.292 | 0.382 |
| C count=0.3 score=0.500 | 0.413 | 1.544 | 0.774 | 0.466 | 0.292 | 0.382 |
| B count=0.3 score=0.525 | 0.432 | 1.511 | 0.786 | 0.396 | 0.260 | 0.414 |
| C count=0.3 score=0.525 | 0.432 | 1.511 | 0.786 | 0.396 | 0.260 | 0.414 |
| B count=0.3 score=0.550 | 0.449 | 1.485 | 0.792 | 0.340 | 0.238 | 0.440 |
| C count=0.3 score=0.550 | 0.449 | 1.485 | 0.792 | 0.340 | 0.238 | 0.440 |
| B count=0.3 score=0.575 | 0.466 | 1.471 | 0.793 | 0.288 | 0.220 | 0.466 |
| C count=0.3 score=0.575 | 0.466 | 1.471 | 0.793 | 0.288 | 0.220 | 0.466 |
| B count=0.3 score=0.600 | 0.468 | 1.473 | 0.796 | 0.254 | 0.190 | 0.462 |
| C count=0.3 score=0.600 | 0.468 | 1.473 | 0.796 | 0.254 | 0.190 | 0.462 |
| B count=0.3 score=0.625 | 0.479 | 1.474 | 0.797 | 0.204 | 0.162 | 0.478 |
| C count=0.3 score=0.625 | 0.479 | 1.474 | 0.797 | 0.204 | 0.162 | 0.478 |
| B count=0.3 score=0.650 | 0.486 | 1.468 | 0.798 | 0.176 | 0.148 | 0.488 |
| C count=0.3 score=0.650 | 0.486 | 1.468 | 0.798 | 0.176 | 0.148 | 0.488 |
| B count=0.3 score=0.675 | 0.491 | 1.462 | 0.800 | 0.140 | 0.122 | 0.490 |
| C count=0.3 score=0.675 | 0.491 | 1.462 | 0.800 | 0.140 | 0.122 | 0.490 |
| B count=0.3 score=0.700 | 0.486 | 1.472 | 0.801 | 0.124 | 0.096 | 0.498 |
| C count=0.3 score=0.700 | 0.486 | 1.472 | 0.801 | 0.124 | 0.096 | 0.498 |
| B count=0.35 score=0.300 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| C count=0.35 score=0.300 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| B count=0.35 score=0.325 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| C count=0.35 score=0.325 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| B count=0.35 score=0.350 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| C count=0.35 score=0.350 | 0.324 | 1.528 | 0.804 | 0.844 | 0.492 | 0.306 |
| B count=0.35 score=0.375 | 0.331 | 1.534 | 0.797 | 0.800 | 0.462 | 0.306 |
| C count=0.35 score=0.375 | 0.331 | 1.534 | 0.797 | 0.800 | 0.462 | 0.306 |
| B count=0.35 score=0.400 | 0.344 | 1.553 | 0.787 | 0.748 | 0.436 | 0.320 |
| C count=0.35 score=0.400 | 0.344 | 1.553 | 0.787 | 0.748 | 0.436 | 0.320 |
| B count=0.35 score=0.425 | 0.368 | 1.529 | 0.800 | 0.664 | 0.400 | 0.344 |
| C count=0.35 score=0.425 | 0.368 | 1.529 | 0.800 | 0.664 | 0.400 | 0.344 |
| B count=0.35 score=0.450 | 0.397 | 1.490 | 0.815 | 0.582 | 0.376 | 0.368 |
| C count=0.35 score=0.450 | 0.397 | 1.490 | 0.815 | 0.582 | 0.376 | 0.368 |
| B count=0.35 score=0.475 | 0.408 | 1.496 | 0.808 | 0.534 | 0.350 | 0.380 |
| C count=0.35 score=0.475 | 0.408 | 1.496 | 0.808 | 0.534 | 0.350 | 0.380 |
| B count=0.35 score=0.500 | 0.428 | 1.486 | 0.810 | 0.466 | 0.322 | 0.398 |
| C count=0.35 score=0.500 | 0.428 | 1.486 | 0.810 | 0.466 | 0.322 | 0.398 |
| B count=0.35 score=0.525 | 0.444 | 1.467 | 0.814 | 0.396 | 0.284 | 0.424 |
| C count=0.35 score=0.525 | 0.444 | 1.467 | 0.814 | 0.396 | 0.284 | 0.424 |
| B count=0.35 score=0.550 | 0.457 | 1.456 | 0.813 | 0.340 | 0.254 | 0.448 |
| C count=0.35 score=0.550 | 0.457 | 1.456 | 0.813 | 0.340 | 0.254 | 0.448 |
| B count=0.35 score=0.575 | 0.471 | 1.448 | 0.812 | 0.288 | 0.230 | 0.470 |
| C count=0.35 score=0.575 | 0.471 | 1.448 | 0.812 | 0.288 | 0.230 | 0.470 |
| B count=0.35 score=0.600 | 0.473 | 1.453 | 0.812 | 0.254 | 0.200 | 0.466 |
| C count=0.35 score=0.600 | 0.473 | 1.453 | 0.812 | 0.254 | 0.200 | 0.466 |
| B count=0.35 score=0.625 | 0.482 | 1.459 | 0.808 | 0.204 | 0.168 | 0.480 |
| C count=0.35 score=0.625 | 0.482 | 1.459 | 0.808 | 0.204 | 0.168 | 0.480 |
| B count=0.35 score=0.650 | 0.487 | 1.457 | 0.807 | 0.176 | 0.150 | 0.488 |
| C count=0.35 score=0.650 | 0.487 | 1.457 | 0.807 | 0.176 | 0.150 | 0.488 |
| B count=0.35 score=0.675 | 0.491 | 1.454 | 0.807 | 0.140 | 0.122 | 0.490 |
| C count=0.35 score=0.675 | 0.491 | 1.454 | 0.807 | 0.140 | 0.122 | 0.490 |
| B count=0.35 score=0.700 | 0.486 | 1.468 | 0.805 | 0.124 | 0.096 | 0.498 |
| C count=0.35 score=0.700 | 0.486 | 1.468 | 0.805 | 0.124 | 0.096 | 0.498 |
| B count=0.4 score=0.300 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| C count=0.4 score=0.300 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| B count=0.4 score=0.325 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| C count=0.4 score=0.325 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| B count=0.4 score=0.350 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| C count=0.4 score=0.350 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| B count=0.4 score=0.375 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| C count=0.4 score=0.375 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| B count=0.4 score=0.400 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| C count=0.4 score=0.400 | 0.364 | 1.488 | 0.834 | 0.748 | 0.476 | 0.342 |
| B count=0.4 score=0.425 | 0.385 | 1.477 | 0.832 | 0.664 | 0.434 | 0.356 |
| C count=0.4 score=0.425 | 0.385 | 1.477 | 0.832 | 0.664 | 0.434 | 0.356 |
| B count=0.4 score=0.450 | 0.409 | 1.455 | 0.834 | 0.582 | 0.400 | 0.374 |
| C count=0.4 score=0.450 | 0.409 | 1.455 | 0.834 | 0.582 | 0.400 | 0.374 |
| B count=0.4 score=0.475 | 0.417 | 1.468 | 0.824 | 0.534 | 0.368 | 0.386 |
| C count=0.4 score=0.475 | 0.417 | 1.468 | 0.824 | 0.534 | 0.368 | 0.386 |
| B count=0.4 score=0.500 | 0.435 | 1.464 | 0.822 | 0.466 | 0.336 | 0.402 |
| C count=0.4 score=0.500 | 0.435 | 1.464 | 0.822 | 0.466 | 0.336 | 0.402 |
| B count=0.4 score=0.525 | 0.447 | 1.454 | 0.822 | 0.396 | 0.290 | 0.426 |
| C count=0.4 score=0.525 | 0.447 | 1.454 | 0.822 | 0.396 | 0.290 | 0.426 |
| B count=0.4 score=0.550 | 0.461 | 1.447 | 0.818 | 0.340 | 0.262 | 0.450 |
| C count=0.4 score=0.550 | 0.461 | 1.447 | 0.818 | 0.340 | 0.262 | 0.450 |
| B count=0.4 score=0.575 | 0.473 | 1.444 | 0.815 | 0.288 | 0.234 | 0.470 |
| C count=0.4 score=0.575 | 0.473 | 1.444 | 0.815 | 0.288 | 0.234 | 0.470 |
| B count=0.4 score=0.600 | 0.475 | 1.450 | 0.814 | 0.254 | 0.204 | 0.466 |
| C count=0.4 score=0.600 | 0.475 | 1.450 | 0.814 | 0.254 | 0.204 | 0.466 |
| B count=0.4 score=0.625 | 0.482 | 1.458 | 0.809 | 0.204 | 0.168 | 0.478 |
| C count=0.4 score=0.625 | 0.482 | 1.458 | 0.809 | 0.204 | 0.168 | 0.478 |
| B count=0.4 score=0.650 | 0.487 | 1.457 | 0.807 | 0.176 | 0.150 | 0.486 |
| C count=0.4 score=0.650 | 0.487 | 1.457 | 0.807 | 0.176 | 0.150 | 0.486 |
| B count=0.4 score=0.675 | 0.490 | 1.456 | 0.806 | 0.140 | 0.120 | 0.488 |
| C count=0.4 score=0.675 | 0.490 | 1.456 | 0.806 | 0.140 | 0.120 | 0.488 |
| B count=0.4 score=0.700 | 0.485 | 1.470 | 0.804 | 0.124 | 0.094 | 0.496 |
| C count=0.4 score=0.700 | 0.485 | 1.470 | 0.804 | 0.124 | 0.094 | 0.496 |
