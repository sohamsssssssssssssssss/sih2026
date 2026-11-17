# T18 CDVQA image-differencing baseline

This report uses the three binary question types in the [preregistration](../../docs/hypotheses/T18.md). Change destination, largest and smallest changed class, and both semantic change-ratio types are out of scope. Binary CDVQA questions also name classes. The pixel-only arm uses one image-pair feature vector per type, so it cannot tell which named class a question refers to.

Intervals are percentile CI95 from 2,000 image-pair bootstrap replicates with seed 17. Accuracy is question-weighted within each type. Test1 and test2 share image pairs and should not be treated as independent replications. The oracle-mask arm uses supplied semantic masks alongside the RGB features.

Identical and swapped agreement use the original labels, matching the G3 convention. The identical-no column instead measures the correct presence answer for the artificial identical pair. For directional questions, swap flip rate uses only originally correct answers and scores no counterfactual labels. A class with no net area change can legitimately receive no in both orders.

## test1

Full-split results and G3 scores use different rows. Adapter comparisons are confined to the matched subtable with the exact frozen IDs, pair IDs, question text, and labels from commit `89fc5bf`.

### Full official split

| Type | Arm | Real accuracy | Identical agreement | Swapped agreement | Identical no | Swap change or correct-answer flip |
|---|---|---|---|---|---|---|
| change_or_not (n=13882, 968 pairs, 7797 yes) | majority | 0.562 [0.553, 0.571] | 0.562 [0.553, 0.571] | 0.562 [0.553, 0.571] | 0.000% [0.000, 0.000] | 0.000% [0.000, 0.000] |
| change_or_not (n=13882, 968 pairs, 7797 yes) | question-only | 0.828 [0.822, 0.834] | 0.828 [0.822, 0.834] | 0.828 [0.822, 0.834] | 44.410% [44.043, 44.779] | 0.000% [0.000, 0.000] |
| change_or_not (n=13882, 968 pairs, 7797 yes) | diff-baseline | 0.560 [0.551, 0.570] | 0.438 [0.429, 0.447] | 0.557 [0.548, 0.567] | 100.000% [100.000, 100.000] | 5.540% [4.201, 7.024] |
| change_or_not (n=13882, 968 pairs, 7797 yes) | oracle-mask | 0.568 [0.558, 0.577] | 0.574 [0.565, 0.583] | 0.566 [0.557, 0.575] | 16.136% [13.870, 18.544] | 14.422% [12.300, 16.646] |
| increase_or_not (n=4600, 968 pairs, 1535 yes) | majority | 0.666 [0.656, 0.676] | 0.666 [0.656, 0.676] | 0.666 [0.656, 0.676] | — | Insufficient support (n=3065); correct yes=0 |
| increase_or_not (n=4600, 968 pairs, 1535 yes) | question-only | 0.737 [0.725, 0.749] | 0.737 [0.725, 0.749] | 0.737 [0.725, 0.749] | — | 0.000% [0.000, 0.000] (n=3391); correct yes=548 |
| increase_or_not (n=4600, 968 pairs, 1535 yes) | diff-baseline | 0.666 [0.656, 0.676] | 0.666 [0.656, 0.676] | 0.666 [0.656, 0.676] | — | Insufficient support (n=3065); correct yes=0 |
| increase_or_not (n=4600, 968 pairs, 1535 yes) | oracle-mask | 0.664 [0.654, 0.675] | 0.658 [0.647, 0.669] | 0.661 [0.650, 0.671] | — | 2.062% [1.201, 3.013] (n=3056); correct yes=26 |
| decrease_or_not (n=4658, 968 pairs, 1444 yes) | majority | 0.690 [0.680, 0.700] | 0.690 [0.680, 0.700] | 0.690 [0.680, 0.700] | — | Insufficient support (n=3214); correct yes=0 |
| decrease_or_not (n=4658, 968 pairs, 1444 yes) | question-only | 0.737 [0.724, 0.750] | 0.737 [0.724, 0.750] | 0.737 [0.724, 0.750] | — | 0.000% [0.000, 0.000] (n=3431); correct yes=782 |
| decrease_or_not (n=4658, 968 pairs, 1444 yes) | diff-baseline | 0.689 [0.679, 0.700] | 0.690 [0.680, 0.700] | 0.690 [0.680, 0.700] | — | Insufficient support (n=3211); correct yes=1 |
| decrease_or_not (n=4658, 968 pairs, 1444 yes) | oracle-mask | 0.687 [0.677, 0.697] | 0.684 [0.674, 0.695] | 0.686 [0.675, 0.696] | — | Insufficient support (n=3200); correct yes=6 |

### Exact G3 question subset

| Type | Arm | Real accuracy | Identical agreement | Swapped agreement | Identical no | Swap change or correct-answer flip |
|---|---|---|---|---|---|---|
| change_or_not (n=50, 42 pairs, 23 yes) | majority | 0.460 [0.327, 0.596] | 0.460 [0.327, 0.596] | 0.460 [0.327, 0.596] | 0.000% [0.000, 0.000] | 0.000% [0.000, 0.000] |
| change_or_not (n=50, 42 pairs, 23 yes) | question-only | 0.780 [0.660, 0.893] | 0.780 [0.660, 0.893] | 0.780 [0.660, 0.893] | 44.000% [31.250, 56.863] | 0.000% [0.000, 0.000] |
| change_or_not (n=50, 42 pairs, 23 yes) | diff-baseline | 0.440 [0.304, 0.582] | 0.540 [0.404, 0.673] | 0.460 [0.327, 0.596] | 100.000% [100.000, 100.000] | 6.000% [0.000, 13.639] |
| change_or_not (n=50, 42 pairs, 23 yes) | oracle-mask | 0.500 [0.365, 0.638] | 0.620 [0.490, 0.750] | 0.480 [0.340, 0.617] | 24.000% [11.316, 38.469] | 26.000% [13.725, 40.435] |
| change_or_not (n=50, 42 pairs, 23 yes) | G3 adapter | 0.780 [0.667, 0.887] | 0.760 [0.640, 0.870] | 0.780 [0.646, 0.889] | 58.000% [45.098, 70.588] | 12.000% [4.082, 21.569] |
| increase_or_not (n=50, 40 pairs, 21 yes) | majority | 0.580 [0.423, 0.729] | 0.580 [0.423, 0.729] | 0.580 [0.423, 0.729] | — | Insufficient support (n=29); correct yes=0 |
| increase_or_not (n=50, 40 pairs, 21 yes) | question-only | 0.740 [0.608, 0.857] | 0.740 [0.608, 0.857] | 0.740 [0.608, 0.857] | — | 0.000% [0.000, 0.000] (n=37); correct yes=10 |
| increase_or_not (n=50, 40 pairs, 21 yes) | diff-baseline | 0.580 [0.423, 0.729] | 0.580 [0.423, 0.729] | 0.580 [0.423, 0.729] | — | Insufficient support (n=29); correct yes=0 |
| increase_or_not (n=50, 40 pairs, 21 yes) | oracle-mask | 0.660 [0.520, 0.800] | 0.640 [0.490, 0.780] | 0.580 [0.423, 0.729] | — | Insufficient support (n=33); correct yes=4 |
| increase_or_not (n=50, 40 pairs, 21 yes) | G3 adapter | 0.720 [0.615, 0.824] | 0.560 [0.420, 0.704] | 0.580 [0.447, 0.712] | — | 44.444% [28.571, 60.526] (n=36); correct yes=14 |
| decrease_or_not (n=50, 43 pairs, 16 yes) | majority | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 43 pairs, 16 yes) | question-only | 0.820 [0.700, 0.925] | 0.820 [0.700, 0.925] | 0.820 [0.700, 0.925] | — | Insufficient support (n=41); correct yes=8 |
| decrease_or_not (n=50, 43 pairs, 16 yes) | diff-baseline | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 43 pairs, 16 yes) | oracle-mask | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | 0.680 [0.558, 0.800] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 43 pairs, 16 yes) | G3 adapter | 0.840 [0.739, 0.938] | 0.720 [0.596, 0.837] | 0.740 [0.617, 0.854] | — | Insufficient support (n=42); correct yes=8 |

- **H1 FAIL.** Presence accuracy gain versus train majority: -0.002 [-0.005, 0.001].
- **H2 PASS.** Presence no on identical pairs (≥95%): 100.000% [100.000, 100.000].
- **H3 presence FAIL.** Presence swap changes (<5%): 5.540% [4.201, 7.024].
- **H3 increase INCONCLUSIVE.** Increase swap flips (>80%): Insufficient support (n=3065).
- **H3 decrease INCONCLUSIVE.** Decrease swap flips (>80%): Insufficient support (n=3211).

## test2

Full-split results and G3 scores use different rows. Adapter comparisons are confined to the matched subtable with the exact frozen IDs, pair IDs, question text, and labels from commit `89fc5bf`.

### Full official split

| Type | Arm | Real accuracy | Identical agreement | Swapped agreement | Identical no | Swap change or correct-answer flip |
|---|---|---|---|---|---|---|
| change_or_not (n=5126, 967 pairs, 2844 yes) | majority | 0.555 [0.541, 0.569] | 0.555 [0.541, 0.569] | 0.555 [0.541, 0.569] | 0.000% [0.000, 0.000] | 0.000% [0.000, 0.000] |
| change_or_not (n=5126, 967 pairs, 2844 yes) | question-only | 0.823 [0.813, 0.834] | 0.823 [0.813, 0.834] | 0.823 [0.813, 0.834] | 44.577% [43.359, 45.732] | 0.000% [0.000, 0.000] |
| change_or_not (n=5126, 967 pairs, 2844 yes) | diff-baseline | 0.550 [0.536, 0.564] | 0.445 [0.431, 0.459] | 0.551 [0.537, 0.565] | 100.000% [100.000, 100.000] | 5.892% [4.347, 7.558] |
| change_or_not (n=5126, 967 pairs, 2844 yes) | oracle-mask | 0.568 [0.554, 0.582] | 0.574 [0.560, 0.588] | 0.560 [0.546, 0.573] | 16.270% [13.903, 18.913] | 14.826% [12.394, 17.286] |
| increase_or_not (n=1805, 849 pairs, 579 yes) | majority | 0.679 [0.661, 0.698] | 0.679 [0.661, 0.698] | 0.679 [0.661, 0.698] | — | Insufficient support (n=1226); correct yes=0 |
| increase_or_not (n=1805, 849 pairs, 579 yes) | question-only | 0.740 [0.721, 0.760] | 0.740 [0.721, 0.760] | 0.740 [0.721, 0.760] | — | 0.000% [0.000, 0.000] (n=1335); correct yes=200 |
| increase_or_not (n=1805, 849 pairs, 579 yes) | diff-baseline | 0.679 [0.661, 0.698] | 0.679 [0.661, 0.698] | 0.678 [0.660, 0.697] | — | Insufficient support (n=1226); correct yes=0 |
| increase_or_not (n=1805, 849 pairs, 579 yes) | oracle-mask | 0.674 [0.655, 0.693] | 0.672 [0.653, 0.691] | 0.673 [0.654, 0.692] | — | Insufficient support (n=1216); correct yes=8 |
| decrease_or_not (n=1754, 864 pairs, 545 yes) | majority | 0.689 [0.670, 0.708] | 0.689 [0.670, 0.708] | 0.689 [0.670, 0.708] | — | Insufficient support (n=1209); correct yes=0 |
| decrease_or_not (n=1754, 864 pairs, 545 yes) | question-only | 0.738 [0.718, 0.757] | 0.738 [0.718, 0.757] | 0.738 [0.718, 0.757] | — | 0.000% [0.000, 0.000] (n=1295); correct yes=299 |
| decrease_or_not (n=1754, 864 pairs, 545 yes) | diff-baseline | 0.689 [0.670, 0.708] | 0.689 [0.670, 0.708] | 0.689 [0.670, 0.708] | — | Insufficient support (n=1208); correct yes=1 |
| decrease_or_not (n=1754, 864 pairs, 545 yes) | oracle-mask | 0.686 [0.666, 0.705] | 0.684 [0.664, 0.703] | 0.682 [0.662, 0.701] | — | Insufficient support (n=1203); correct yes=5 |

### Exact G3 question subset

| Type | Arm | Real accuracy | Identical agreement | Swapped agreement | Identical no | Swap change or correct-answer flip |
|---|---|---|---|---|---|---|
| change_or_not (n=50, 39 pairs, 27 yes) | majority | 0.540 [0.395, 0.673] | 0.540 [0.395, 0.673] | 0.540 [0.395, 0.673] | 0.000% [0.000, 0.000] | 0.000% [0.000, 0.000] |
| change_or_not (n=50, 39 pairs, 27 yes) | question-only | 0.820 [0.712, 0.918] | 0.820 [0.712, 0.918] | 0.820 [0.712, 0.918] | 52.000% [40.814, 64.000] | 0.000% [0.000, 0.000] |
| change_or_not (n=50, 39 pairs, 27 yes) | diff-baseline | 0.480 [0.340, 0.625] | 0.460 [0.327, 0.605] | 0.540 [0.395, 0.673] | 100.000% [100.000, 100.000] | 18.000% [4.255, 32.653] |
| change_or_not (n=50, 39 pairs, 27 yes) | oracle-mask | 0.520 [0.375, 0.667] | 0.600 [0.465, 0.731] | 0.580 [0.444, 0.714] | 30.000% [14.582, 46.438] | 18.000% [5.553, 32.692] |
| change_or_not (n=50, 39 pairs, 27 yes) | G3 adapter | 0.860 [0.761, 0.941] | 0.800 [0.702, 0.898] | 0.860 [0.745, 0.958] | 62.000% [50.979, 73.913] | 8.000% [2.083, 14.815] |
| increase_or_not (n=50, 39 pairs, 17 yes) | majority | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | — | Insufficient support (n=33); correct yes=0 |
| increase_or_not (n=50, 39 pairs, 17 yes) | question-only | 0.720 [0.600, 0.848] | 0.720 [0.600, 0.848] | 0.720 [0.600, 0.848] | — | Insufficient support (n=36); correct yes=7 |
| increase_or_not (n=50, 39 pairs, 17 yes) | diff-baseline | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | — | Insufficient support (n=33); correct yes=0 |
| increase_or_not (n=50, 39 pairs, 17 yes) | oracle-mask | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | 0.660 [0.549, 0.767] | — | Insufficient support (n=33); correct yes=0 |
| increase_or_not (n=50, 39 pairs, 17 yes) | G3 adapter | 0.680 [0.542, 0.809] | 0.660 [0.549, 0.767] | 0.620 [0.500, 0.740] | — | Insufficient support (n=34); correct yes=7 |
| decrease_or_not (n=50, 41 pairs, 16 yes) | majority | 0.680 [0.551, 0.809] | 0.680 [0.551, 0.809] | 0.680 [0.551, 0.809] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 41 pairs, 16 yes) | question-only | 0.740 [0.612, 0.857] | 0.740 [0.612, 0.857] | 0.740 [0.612, 0.857] | — | Insufficient support (n=37); correct yes=8 |
| decrease_or_not (n=50, 41 pairs, 16 yes) | diff-baseline | 0.680 [0.551, 0.809] | 0.680 [0.551, 0.809] | 0.680 [0.551, 0.809] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 41 pairs, 16 yes) | oracle-mask | 0.680 [0.551, 0.809] | 0.680 [0.551, 0.809] | 0.700 [0.574, 0.833] | — | Insufficient support (n=34); correct yes=0 |
| decrease_or_not (n=50, 41 pairs, 16 yes) | G3 adapter | 0.760 [0.642, 0.880] | 0.700 [0.574, 0.833] | 0.680 [0.547, 0.810] | — | Insufficient support (n=38); correct yes=7 |

- **H1 FAIL.** Presence accuracy gain versus train majority: -0.005 [-0.011, 0.000].
- **H2 PASS.** Presence no on identical pairs (≥95%): 100.000% [100.000, 100.000].
- **H3 presence FAIL.** Presence swap changes (<5%): 5.892% [4.347, 7.558].
- **H3 increase INCONCLUSIVE.** Increase swap flips (>80%): Insufficient support (n=1226).
- **H3 decrease INCONCLUSIVE.** Decrease swap flips (>80%): Insufficient support (n=1208).

## Preregistered verdicts

**H1 FAIL. H2 PASS. H3 FAIL.** H3 requires both directional controls and presence swap stability on both splits.

The presence criteria yield **FAIL** (H1 FAIL, H2 PASS, presence swap stability FAIL). On real CDVQA presence questions the accuracies were test1: diff-baseline 0.560, majority 0.562, question-only 0.828 and test2: diff-baseline 0.550, majority 0.555, question-only 0.823. These criteria measure CDVQA presence performance and image controls. Independent validation on Indian imagery is outside this evaluation.

The selected regularization, exact counts, unrounded metrics, G3 hashes and QA hashes are in `summary.json`.
