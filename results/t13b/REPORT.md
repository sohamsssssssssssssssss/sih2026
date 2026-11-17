# T13b analysis, registered CI method

Full fine-tuning beats the stored baseline (P1 **PASS**) and shuffle (P2 **PASS**) across all three modes. The overall gate is **PASS**. The fusion shuffle-fixed diagnostic falls below chance, and full fine-tuning still regresses in some regions. The fold-A optical shuffle gain on the seed mean is +0.037185. Rule (d) supports an epoch-selection artefact, and rule (e) supports BN re-estimation. Compared with the substitute-rule report at `9672b78`, the rerun confirms P1 and confirms the rule (e) result. All 21 cell means match the earlier report. Its half-gap recovery and shuffled-control findings also hold: full fine-tuning clears the half-gap threshold only for S1, and the S1 and S2 shuffled controls sit more than 0.05 from chance. The saved predictions resolve the previously inconclusive P2, gate and rule (d).

**CI method used: REGISTERED** (docs/hypotheses/T13b.md:31): patch-level bootstrap, 1000 reps, seed 17, on seed-averaged per-patch predictions read from the three saved NPZ archives. Differences use paired resamples. The seed-level substitute rule was not used.

Verdicts: P1 **PASS**, P2 **PASS**, P3 (non-diagnostic cells) **PASS**, overall gate **PASS**. Not a final-model claim. Real-data index/permutation/pixel-hash audits: **PASS**.

## OOF India-pooled cells (seed mean, min–max, exact chance)

| Mode | Arm | Seed 1 | Seed 2 | Seed 3 | Mean | Min–max | Exact chance | P3 | Seed-averaged CI95 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| s1 | none | 0.516018 | 0.516018 | 0.516018 | 0.516018 | 0.516018–0.516018 | 0.36244444444444446 | PASS | [0.502831, 0.532436] |
| s1 | renorm | 0.505243 | 0.505243 | 0.505243 | 0.505243 | 0.505243–0.505243 | 0.36244444444444446 | PASS | [0.491907, 0.521256] |
| s1 | head | 0.540109 | 0.544968 | 0.545499 | 0.543525 | 0.540109–0.545499 | 0.36244444444444446 | PASS | [0.529230, 0.562236] |
| s1 | full | 0.717207 | 0.699087 | 0.717931 | 0.711408 | 0.699087–0.717931 | 0.36244444444444446 | PASS | [0.698242, 0.730833] |
| s1 | shuffle | 0.538944 | 0.504472 | 0.574302 | 0.539239 | 0.504472–0.574302 | 0.36244444444444446 | PASS | [0.537447, 0.572799] |
| s1 | shuffle-fixed | 0.475415 | 0.446677 | 0.505551 | 0.475881 | 0.446677–0.505551 | 0.36244444444444446 | PASS | [0.479632, 0.513818] |
| s1 | bn-only | 0.678590 | 0.676092 | 0.676194 | 0.676959 | 0.676092–0.678590 | 0.36244444444444446 | PASS | [0.663123, 0.694188] |
| s2 | none | 0.575885 | 0.575885 | 0.575885 | 0.575885 | 0.575885–0.575885 | 0.36244444444444446 | PASS | [0.560810, 0.592751] |
| s2 | renorm | 0.471049 | 0.471049 | 0.471049 | 0.471049 | 0.471049–0.471049 | 0.36244444444444446 | PASS | [0.455888, 0.487507] |
| s2 | head | 0.642226 | 0.654522 | 0.653356 | 0.650035 | 0.642226–0.654522 | 0.36244444444444446 | PASS | [0.636473, 0.665363] |
| s2 | full | 0.688473 | 0.676393 | 0.668860 | 0.677909 | 0.668860–0.688473 | 0.36244444444444446 | PASS | [0.670334, 0.698532] |
| s2 | shuffle | 0.524229 | 0.526369 | 0.586072 | 0.545557 | 0.524229–0.586072 | 0.36244444444444446 | PASS | [0.564462, 0.596176] |
| s2 | shuffle-fixed | 0.385678 | 0.417623 | 0.402879 | 0.402060 | 0.385678–0.417623 | 0.36244444444444446 | PASS | [0.396137, 0.417961] |
| s2 | bn-only | 0.595780 | 0.596273 | 0.597655 | 0.596569 | 0.595780–0.597655 | 0.36244444444444446 | PASS | [0.582817, 0.611179] |
| s1s2 | none | 0.667333 | 0.667333 | 0.667333 | 0.667333 | 0.667333–0.667333 | 0.36244444444444446 | PASS | [0.653408, 0.682608] |
| s1s2 | renorm | 0.504708 | 0.504708 | 0.504708 | 0.504708 | 0.504708–0.504708 | 0.36244444444444446 | PASS | [0.493053, 0.517963] |
| s1s2 | head | 0.640912 | 0.658096 | 0.641555 | 0.646854 | 0.640912–0.658096 | 0.36244444444444446 | PASS | [0.631631, 0.665517] |
| s1s2 | full | 0.698369 | 0.718712 | 0.697301 | 0.704794 | 0.697301–0.718712 | 0.36244444444444446 | PASS | [0.699284, 0.727088] |
| s1s2 | shuffle | 0.490396 | 0.334116 | 0.373945 | 0.399486 | 0.334116–0.490396 | 0.36244444444444446 | PASS | [0.419731, 0.444862] |
| s1s2 | shuffle-fixed | 0.405764 | 0.349766 | 0.321906 | 0.359145 | 0.321906–0.405764 | 0.36244444444444446 | FAIL | [0.331389, 0.360433] |
| s1s2 | bn-only | 0.610023 | 0.607308 | 0.613282 | 0.610204 | 0.607308–0.613282 | 0.36244444444444446 | PASS | [0.596067, 0.628744] |

P3 uses unrounded values. shuffle-fixed and bn-only are diagnostics, excluded from the gate.

## Paired contrasts, registered CI

| Mode | Contrast | Seed 1 | Seed 2 | Seed 3 | Seed mean | Seed-averaged point | CI95 (paired) | Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| s1 | full−none (P1) | 0.201188 | 0.183069 | 0.201913 | 0.195390 | 0.197036 | [0.178779, 0.215607] | PASS |
| s1 | full−shuffle (P2) | 0.178263 | 0.194615 | 0.143628 | 0.172169 | 0.160004 | [0.142892, 0.175891] | PASS |
| s1 | full−renorm (matched, not gated) | 0.211964 | 0.193844 | 0.212688 | 0.206165 | 0.207811 | [0.192496, 0.222491] | PASS |
| s2 | full−none (P1) | 0.112589 | 0.100508 | 0.092975 | 0.102024 | 0.108257 | [0.090485, 0.124109] | PASS |
| s2 | full−shuffle (P2) | 0.164244 | 0.150024 | 0.082788 | 0.132352 | 0.104402 | [0.085131, 0.122753] | PASS |
| s2 | full−renorm (matched, not gated) | 0.217424 | 0.205343 | 0.197810 | 0.206859 | 0.213092 | [0.194673, 0.230042] | PASS |
| s1s2 | full−none (P1) | 0.031036 | 0.051380 | 0.029969 | 0.037461 | 0.044533 | [0.031054, 0.058809] | PASS |
| s1s2 | full−shuffle (P2) | 0.207972 | 0.384596 | 0.323356 | 0.305308 | 0.280763 | [0.265337, 0.295054] | PASS |
| s1s2 | full−renorm (matched, not gated) | 0.193661 | 0.214005 | 0.192594 | 0.200086 | 0.207158 | [0.193714, 0.220422] | PASS |

P1/P2 rule: mean over seeds > 0 AND CI95 lower bound > 0 (PASS), otherwise FAIL. No INCONCLUSIVE verdict arises under this method.

## Diagnostic arms (not gated)

| Contrast | Seed 1 | Seed 2 | Seed 3 | Seed mean | Seed-averaged point | CI95 (paired) | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| (d) shuffle-fixed−none, fold A, s2 | -0.209239 | -0.149272 | -0.167190 | -0.175234 | -0.165220 | [-0.189519, -0.140348] | FAIL |
| (e) bn-only−shuffle-fixed, fold A, s2 | 0.321214 | 0.261342 | 0.278232 | 0.286929 | 0.276934 | [0.256587, 0.296798] | PASS |

(d) artefact claimed: **YES** (requires seed-17 shuffle > none and shuffle-fixed−none CI lower bound <= 0). (e) mean rule: bn-only 0.6880084555066773 vs shuffle-fixed 0.4010789691722225: **PASS**. Attribution: **BN re-estimation supported by rule (e)**. The (e) rule is the registered mean inequality; its CI is reported beside it, not as a gate.

## Fold-A optical anomaly, arm means

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean | Exact fold prevalence | Seed-averaged CI95 |
|---|---:|---:|---:|---:|---|---|
| none | 0.576312 | 0.576312 | 0.576312 | 0.576312 | 0.35622222222222216 | [0.552903, 0.601576] |
| shuffle | 0.582602 | 0.597907 | 0.659984 | 0.613498 | 0.35622222222222216 | [0.628258, 0.669001] |
| shuffle-fixed | 0.367073 | 0.427041 | 0.409123 | 0.401079 | 0.35622222222222216 | [0.396281, 0.429049] |
| bn-only | 0.688287 | 0.688383 | 0.687355 | 0.688008 | 0.35622222222222216 | [0.674118, 0.705296] |
| full | 0.733637 | 0.726336 | 0.718647 | 0.726207 | 0.35622222222222216 | [0.717455, 0.756362] |

Seed 17 re-read: none=0.5763124803967011, shuffle=0.6452245460444148, difference=0.06891206564771368.

## Selected epochs

| Mode | Fold | Seed | Full | Shuffle | Shuffle-fixed | BN-only |
|---|---|---:|---:|---:|---:|---:|
| s1 | A | 1 | 15 | 1 | 15 | 15 |
| s1 | B | 1 | 12 | 4 | 15 | 15 |
| s2 | A | 1 | 14 | 1 | 15 | 15 |
| s2 | B | 1 | 14 | 4 | 15 | 15 |
| s1s2 | A | 1 | 15 | 2 | 15 | 15 |
| s1s2 | B | 1 | 14 | 5 | 15 | 15 |
| s1 | A | 2 | 14 | 2 | 15 | 15 |
| s1 | B | 2 | 12 | 12 | 15 | 15 |
| s2 | A | 2 | 12 | 2 | 15 | 15 |
| s2 | B | 2 | 15 | 7 | 15 | 15 |
| s1s2 | A | 2 | 15 | 7 | 15 | 15 |
| s1s2 | B | 2 | 14 | 14 | 15 | 15 |
| s1 | A | 3 | 14 | 1 | 15 | 15 |
| s1 | B | 3 | 12 | 2 | 15 | 15 |
| s2 | A | 3 | 11 | 1 | 15 | 15 |
| s2 | B | 3 | 13 | 4 | 15 | 15 |
| s1s2 | A | 3 | 14 | 10 | 15 | 15 |
| s1s2 | B | 3 | 14 | 1 | 15 | 15 |

## Region regressions (seed means, not gated)

| Mode | Region | None | Full | Full−none |
|---|---|---:|---:|---:|
| s1 | none | — | — | — |
| s2 | kerala | 0.643653 | 0.631634 | -0.012018 |
| s1s2 | rajasthan | 0.477219 | 0.442397 | -0.034822 |
| s1s2 | punjab | 0.708656 | 0.663820 | -0.044836 |

## Seed-17 stability, every OOF cell

| Mode | Arm | Seed 17 | Seeds 1–3 min–max | In range? |
|---|---|---:|---|---|
| s1 | none | 0.516018 | 0.516018–0.516018 | IN |
| s1 | renorm | 0.505243 | 0.505243–0.505243 | IN |
| s1 | head | 0.544784 | 0.540109–0.545499 | IN |
| s1 | full | 0.725981 | 0.699087–0.717931 | **OUTSIDE** |
| s1 | shuffle | 0.445725 | 0.504472–0.574302 | **OUTSIDE** |
| s1 | shuffle-fixed | not saved | 0.446677–0.505551 | N/A: absent in seed 17 |
| s1 | bn-only | not saved | 0.676092–0.678590 | N/A: absent in seed 17 |
| s2 | none | 0.575885 | 0.575885–0.575885 | IN |
| s2 | renorm | 0.471049 | 0.471049–0.471049 | IN |
| s2 | head | 0.674506 | 0.642226–0.654522 | **OUTSIDE** |
| s2 | full | 0.720657 | 0.668860–0.688473 | **OUTSIDE** |
| s2 | shuffle | 0.559940 | 0.524229–0.586072 | IN |
| s2 | shuffle-fixed | not saved | 0.385678–0.417623 | N/A: absent in seed 17 |
| s2 | bn-only | not saved | 0.595780–0.597655 | N/A: absent in seed 17 |
| s1s2 | none | 0.667333 | 0.667333–0.667333 | IN |
| s1s2 | renorm | 0.504708 | 0.504708–0.504708 | IN |
| s1s2 | head | 0.653197 | 0.640912–0.658096 | IN |
| s1s2 | full | 0.690754 | 0.697301–0.718712 | **OUTSIDE** |
| s1s2 | shuffle | 0.386363 | 0.334116–0.490396 | IN |
| s1s2 | shuffle-fixed | not saved | 0.321906–0.405764 | N/A: absent in seed 17 |
| s1s2 | bn-only | not saved | 0.607308–0.613282 | N/A: absent in seed 17 |

## Gate and remaining criteria

Overall gate: **PASS** (P1=PASS, P2=PASS, P3=PASS). Step 3 (final models): NOT RUN. No final checkpoint or deployment claim.

## Provenance

```json
{
  "manifest_sha256": "7a214a412b20d73cd356e6bd356555209621baca23ffe25f23a8f0a6250f52a6",
  "reference_json_sha256": "f5eb2569d343f3eab62bc65b19e9b3e6e773ea881d75454caef00000070fd447",
  "code_commit": "0b581f71f3cd82dcbb3cc4b3aaab05ac75c32d31",
  "seed_json_sha256": {
    "1": "ca9336591d71031a28ae6723ef021780fa7059c8bd2710f4842cd5f57cfcf71e",
    "2": "7b4ebb55a98e684482e40edb28d8664a0a9e161f22c78cdfcd8928fcbc7912c9",
    "3": "1556fd211bf27debec5b20e76c18ba4816740ce06669d68a2657fdb8c33cbbef"
  }
}
```

## Preregistered rules and observed results

> Folds are fixed: fold A (adapt punjab, kerala, assam; test delhi, sundarbans, rajasthan) and fold B (swapped). The split comes from `fold_indices`, which keeps `lc.SEED = 17`. The brief asks for the same folds, so the split does not change with the seed.

Verified identical fold memberships for all three rerun seeds and the reference.

> Training seeds are 1, 2 and 3. A training seed controls `seed_everything` (python, numpy, torch), the batch-order RNG and torch generator used in training, and the label permutation in `shuffled_labels`. It is passed explicitly as `--train-seed`. Nothing else changes.

The three seed identities and their recorded configuration match the registered seeds.

> Seed 17 is the original T13 run. It is reported as a reference but is not in the seed mean.

The reference hash was verified and the score appears only in stability and reference comparisons.

> The "none" arm uses the stored checkpoint predictions and does not train, so it has no seed spread by construction.

None has zero spread in every OOF mode.

> Known limit: fixed folds mean seed spread does not capture split variability. Only six regions exist.

The intervals are conditional on the fixed folds.

> **Permutation scope.** Test: for every fold and mode, train labels are a permutation of the original train labels, val labels are a permutation of the original val labels, and test labels are byte-identical to the originals. A failure means the shuffle touched test data. (Existing test `test_shuffle_is_confined_to_train_and_val` covers this on the fixture; the check is repeated on the real fold indices.)

18/18 checks passed. Saved permutations preserve test labels, and the exported labels match the WorldCover pack.

> **Leakage.** Test: train, val and test index sets are pairwise disjoint per fold, and no test region appears in train or val. Also a sha256 of each patch's pixels: no exact duplicate between a test patch and any train/val patch.

18/18 checks passed for indices, regions, and exact pixel hashes.

> **Seed persistence.** The gap (shuffle minus none) must hold on the seed mean (seeds 1, 2, 3), not only on seed 17.

**PASS**. Mean shuffle−none was +0.037185.

> **Candidate mechanism, to test not assume.** The shuffle arm also shuffles validation labels, so its epoch is chosen on shuffled validation. Record the selected epochs for shuffle and full. If they differ, the arms are not identically trained. The diagnostic arm below tests this directly.

The selected-epochs table gives the full and shuffle epochs for all 18 seed, mode and fold combinations.

> **Seed mean and spread.** Report the mean over seeds 1, 2, 3, plus the min and max per cell.

The OOF table gives the three seed scores, their arithmetic mean and min–max for every cell.

> **CI95.** Patch-level bootstrap, 1000 reps, seed 17, on the seed-averaged predictions. Differences use a paired bootstrap on the seed-averaged predictions of both arms. This interval does not include seed variance (see (a)).

The OOF table gives intervals for individual arms. The contrast tables give paired intervals using the same resampled patches for both arms.

> **P1 (full beats none).** Per mode: mean over seeds of (full - none) > 0 AND the CI95 lower bound > 0. All three modes must pass.

**PASS**. The contrast table gives each mode’s seed mean, paired CI95 and verdict.

> **P2 (full beats shuffle).** Per mode: mean over seeds of (full - shuffle) > 0 AND CI95 lower bound > 0. All three modes must pass.

**PASS**. The contrast table gives each mode’s seed mean, paired CI95 and verdict.

> **P3 (chance).** A cell whose seed-mean mAP is below the pooled prevalence baseline for India out-of-fold patches, as computed in the run's JSON, is a reported failure. The threshold is the exact prevalence value from the run, not a rounded number.

The OOF table reports every cell against its exact prevalence. Gate-eligible cells **PASS**.

> **P4 (regressions).** Any region where the seed-mean full mAP is below its seed-mean none mAP is listed by name. Reported, not a gate.

All regressions are listed by region and mode above. They do not enter the gate.

> **Gate.** PASS only if P1 and P2 pass in all three modes and no P3 failure exists. A tie or a CI that touches zero is a FAIL.

**PASS** with P1 PASS, P2 PASS, and gate-eligible P3 PASS.

> **Step 3 (final models).** No pass/fail and no test metrics. Output is checkpoint, data hash, code commit, seed, epochs, region list.

Final models fall outside this saved-output analysis.

> shuffle-fixed uses exactly the shuffle arm's training labels (train permuted within train, val permuted within val, as in `shuffled_labels`), the same optimiser, schedule and epoch count as full, and returns the **final-epoch** weights. It does no validation-based epoch selection.

The saved shuffle permutations match across the two arms in all fold and mode checks. The epoch table records the fixed final epoch.

> It is reported next to shuffle in every table, with the same CI95 method.

Both shuffle arms appear with their registered intervals in the OOF and anomaly tables.

> It is excluded from P1, P2 and the gate.

The gate uses full against none and shuffle, with P3 applied to the non-diagnostic cells.

> Hypothesis H-diag: if shuffle-fixed scores close to prevalence (not above none), the optical fold-A shuffle result in T13 came from validation-based epoch selection under shuffled labels, not from the image signal. If shuffle-fixed also beats none, the shuffled-label signal is not explained by epoch selection and the (b) checks decide.

Fold-A optical shuffle-fixed mean 0.401079, prevalence 0.35622222222222216, and none mean 0.576312. The quoted decision rule below determines the verdict.

> Decision rule for the diagnosis, fixed now: "epoch-selection artefact" is claimed only if shuffle beats none in T13 AND shuffle-fixed does not beat none (CI lower bound <= 0). Otherwise the cause is recorded as unexplained or as named by the (b) checks.

Epoch-selection artefact **supported** under (d). Seed-17 shuffle−none was +0.068912. The shuffle-fixed−none paired CI95 was [-0.189519, -0.140348].

> Procedure: start from the original checkpoint, use the **renorm** statistics as input normalisation, and run the adaptation-train images through the network in train mode under no_grad, using the same epochs, seeded batch order and augmentation as the other trained arms. Only BatchNorm running statistics change. No labels, no optimiser, no weight updates. Final-epoch model, no validation-based selection.

The run metadata and selected epochs identify the BN-only arm for all three seeds and folds.

> Tests required: convolution and classifier weights are unchanged; BatchNorm running statistics change.

The existing synthetic BN test verifies unchanged convolution and classifier weights and changed BN running statistics. The saved outputs contain predictions for the real-data analysis.

> Reported next to shuffle-fixed in every table, with the same CI95 method.

Both diagnostic arms have seed-averaged CI95s in the OOF and fold-A tables.

> Excluded from P1, P2 and the gate.

The BN-only comparison does not enter the gate.

> Decision rule, fixed now: compare fold-A pooled mAP for s2, seed mean over seeds 1, 2, 3. If **bn-only >= shuffle-fixed**, BatchNorm re-estimation, not label noise, explains the fold-A anomaly. If bn-only < shuffle-fixed, the BatchNorm candidate is not supported as the whole explanation, and the cause stays as recorded under (b)/(d).

**PASS** under (e). BN-only was 0.688008 versus shuffle-fixed 0.401079.

> The registered reference stays **none** (stored BEN normalisation, original checkpoint). It is not changed.

P1 uses none. The matched comparison uses renorm.

> Every table reports two comparisons side by side:
> - **full − none** (registered, used by P1)
> - **full − renorm** (matched normalisation: the same renorm statistics, with no training other than the full arm's)

The table below places the registered and matched comparisons side by side.

> The report states the confound: none and renorm differ in normalisation, so full − none includes the cost of renormalisation. The renorm arm alone is worse than none in T13 (s2 out-of-fold 0.471 vs 0.576), so full − none understates the training effect in s2 and overstates it if renorm is the better reference.

Full−none includes the change in normalization. For optical input, full−renorm is larger than full−none in this rerun.

## Registered and matched comparisons by mode

| Mode | Full−none mean [paired CI95] | Full−renorm mean [paired CI95] |
|---|---|---|
| s1 | +0.195390 [0.178779, 0.215607] | +0.206165 [0.192496, 0.222491] |
| s2 | +0.102024 [0.090485, 0.124109] | +0.206859 [0.194673, 0.230042] |
| s1s2 | +0.037461 [0.031054, 0.058809] | +0.200086 [0.193714, 0.220422] |

Seed 17 is a reference and is excluded from seed means. Fixed folds cover six regions. Patch bootstrap intervals omit seed and region variance. WorldCover provides map labels.
