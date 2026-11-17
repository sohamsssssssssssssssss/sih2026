CI verdicts use the conservative per-seed substitute; see DECISIONS.md.

Full fine-tuning beats none in every seed and mode under the available paired CIs (P1 PASS), and all full-minus-renorm intervals are positive. P2 and the overall gate are INCONCLUSIVE because full-minus-shuffle paired CIs were not saved; the larger half-Europe/India-gap recovery claim does not hold for optical or fusion, and shuffled controls are not uniformly at chance. The fold-A optical shuffle lift persists on seed means (0.613498 versus none 0.576312). BN re-estimation is supported by rule (e): bn-only 0.688008 >= shuffle-fixed 0.401079. The mean-only (d) direction supports an epoch-selection effect, but its registered CI predicate is not evaluable. No new training/inference run or final-model claim is made.

# T13b saved-output analysis

Deviation: [DECISIONS.md](../../docs/plan/DECISIONS.md), commit 1966156. Original [T13b preregistration](../../docs/hypotheses/T13b.md) is unchanged (4a9162c, 2662aab, as present at a0d7002). All means below are the plain arithmetic mean of seed-level mAPs; no ensemble mAP or CI is imputed. No CI endpoints are averaged. Per-seed intervals are the saved 1000-replicate patch bootstrap, seed 17. They ignore training-seed, region and split variance. Fixed folds cover six Indian regions; WorldCover labels are map products, not independent ground truth.

## OOF India-pooled cells

| Mode | Arm | Seed 1 mAP [CI95] | Seed 2 mAP [CI95] | Seed 3 mAP [CI95] | Mean | Min–max | Exact chance | P3 |
|---|---|---|---|---|---:|---|---|---|
| s1 | none | 0.516018 [0.502831, 0.532436] | 0.516018 [0.502831, 0.532436] | 0.516018 [0.502831, 0.532436] | 0.516018 | 0.516018–0.516018 | 0.36244444444444446 | PASS |
| s1 | renorm | 0.505243 [0.491907, 0.521256] | 0.505243 [0.491907, 0.521256] | 0.505243 [0.491907, 0.521256] | 0.505243 | 0.505243–0.505243 | 0.36244444444444446 | PASS |
| s1 | head | 0.540109 [0.524908, 0.557579] | 0.544968 [0.529895, 0.562885] | 0.545499 [0.530301, 0.563821] | 0.543525 | 0.540109–0.545499 | 0.36244444444444446 | PASS |
| s1 | full | 0.717207 [0.702029, 0.735120] | 0.699087 [0.684608, 0.716577] | 0.717931 [0.703008, 0.735501] | 0.711408 | 0.699087–0.717931 | 0.36244444444444446 | PASS |
| s1 | shuffle | 0.538944 [0.523906, 0.557235] | 0.504472 [0.489218, 0.524933] | 0.574302 [0.557064, 0.592169] | 0.539239 | 0.504472–0.574302 | 0.36244444444444446 | PASS |
| s1 | shuffle-fixed | 0.475415 [0.461820, 0.493840] | 0.446677 [0.433515, 0.463953] | 0.505551 [0.490651, 0.524410] | 0.475881 | 0.446677–0.505551 | 0.36244444444444446 | PASS |
| s1 | bn-only | 0.678590 [0.664735, 0.695554] | 0.676092 [0.662192, 0.693189] | 0.676194 [0.662235, 0.693484] | 0.676959 | 0.676092–0.678590 | 0.36244444444444446 | PASS |
| s2 | none | 0.575885 [0.560810, 0.592751] | 0.575885 [0.560810, 0.592751] | 0.575885 [0.560810, 0.592751] | 0.575885 | 0.575885–0.575885 | 0.36244444444444446 | PASS |
| s2 | renorm | 0.471049 [0.455888, 0.487507] | 0.471049 [0.455888, 0.487507] | 0.471049 [0.455888, 0.487507] | 0.471049 | 0.471049–0.471049 | 0.36244444444444446 | PASS |
| s2 | head | 0.642226 [0.627629, 0.657928] | 0.654522 [0.640843, 0.669779] | 0.653356 [0.639580, 0.669046] | 0.650035 | 0.642226–0.654522 | 0.36244444444444446 | PASS |
| s2 | full | 0.688473 [0.673646, 0.703764] | 0.676393 [0.662640, 0.690990] | 0.668860 [0.654883, 0.684137] | 0.677909 | 0.668860–0.688473 | 0.36244444444444446 | PASS |
| s2 | shuffle | 0.524229 [0.510269, 0.540304] | 0.526369 [0.510271, 0.545199] | 0.586072 [0.572685, 0.601851] | 0.545557 | 0.524229–0.586072 | 0.36244444444444446 | PASS |
| s2 | shuffle-fixed | 0.385678 [0.375185, 0.397675] | 0.417623 [0.407591, 0.431817] | 0.402879 [0.393050, 0.415740] | 0.402060 | 0.385678–0.417623 | 0.36244444444444446 | PASS |
| s2 | bn-only | 0.595780 [0.582080, 0.610500] | 0.596273 [0.582255, 0.610919] | 0.597655 [0.583732, 0.612269] | 0.596569 | 0.595780–0.597655 | 0.36244444444444446 | PASS |
| s1s2 | none | 0.667333 [0.653408, 0.682608] | 0.667333 [0.653408, 0.682608] | 0.667333 [0.653408, 0.682608] | 0.667333 | 0.667333–0.667333 | 0.36244444444444446 | PASS |
| s1s2 | renorm | 0.504708 [0.493053, 0.517963] | 0.504708 [0.493053, 0.517963] | 0.504708 [0.493053, 0.517963] | 0.504708 | 0.504708–0.504708 | 0.36244444444444446 | PASS |
| s1s2 | head | 0.640912 [0.625310, 0.658279] | 0.658096 [0.642099, 0.675297] | 0.641555 [0.625480, 0.658949] | 0.646854 | 0.640912–0.658096 | 0.36244444444444446 | PASS |
| s1s2 | full | 0.698369 [0.686390, 0.712565] | 0.718712 [0.705296, 0.734677] | 0.697301 [0.685632, 0.711710] | 0.704794 | 0.697301–0.718712 | 0.36244444444444446 | PASS |
| s1s2 | shuffle | 0.490396 [0.477697, 0.504960] | 0.334116 [0.324589, 0.349129] | 0.373945 [0.361771, 0.389889] | 0.399486 | 0.334116–0.490396 | 0.36244444444444446 | PASS |
| s1s2 | shuffle-fixed | 0.405764 [0.390633, 0.422435] | 0.349766 [0.339779, 0.365400] | 0.321906 [0.311447, 0.336365] | 0.359145 | 0.321906–0.405764 | 0.36244444444444446 | FAIL |
| s1s2 | bn-only | 0.610023 [0.594540, 0.627352] | 0.607308 [0.591679, 0.625289] | 0.613282 [0.598206, 0.630448] | 0.610204 | 0.607308–0.613282 | 0.36244444444444446 | PASS |

P3 is tested on unrounded values. shuffle-fixed and bn-only are reported but excluded from the gate by (d)/(e); the below-chance s1s2 shuffle-fixed cell is therefore a reported diagnostic failure, not a gate failure.

## Registered and matched comparisons side by side

| Mode | Seed | full−none (registered), paired CI95 | full−renorm (matched), paired CI95 |
|---|---:|---|---|
| s1 | 1 | 0.201188 [0.182288, 0.220379] | 0.211964 [0.196645, 0.226556] |
| s1 | 2 | 0.183069 [0.164265, 0.201282] | 0.193844 [0.179312, 0.207060] |
| s1 | 3 | 0.201913 [0.183559, 0.220332] | 0.212688 [0.197658, 0.227553] |
| s1 | plain mean / substitute verdict | 0.195390 / PASS | 0.206165 / PASS |
| s2 | 1 | 0.112589 [0.094959, 0.129577] | 0.217424 [0.197905, 0.235693] |
| s2 | 2 | 0.100508 [0.083681, 0.115751] | 0.205343 [0.187269, 0.223024] |
| s2 | 3 | 0.092975 [0.076156, 0.108586] | 0.197810 [0.179371, 0.215971] |
| s2 | plain mean / substitute verdict | 0.102024 / PASS | 0.206859 / PASS |
| s1s2 | 1 | 0.031036 [0.018198, 0.044593] | 0.193661 [0.181053, 0.205857] |
| s1s2 | 2 | 0.051380 [0.036186, 0.065931] | 0.214005 [0.199712, 0.227946] |
| s1s2 | 3 | 0.029969 [0.016272, 0.043917] | 0.192594 [0.179456, 0.206132] |
| s1s2 | plain mean / substitute verdict | 0.037461 / PASS | 0.200086 / PASS |

None uses stored BEN normalization; renorm/full share adaptation-train normalization. full−none includes renormalization cost and is not the same estimand as full−renorm. No seed-mean CI is available.

## Registered rules, quoted, and verdicts

### CI95

> **CI95.** Patch-level bootstrap, 1000 reps, seed 17, on the seed-averaged predictions. Differences use a paired bootstrap on the seed-averaged predictions of both arms. This interval does not include seed variance (see (a)).

DEVIATION: use saved per-seed paired CIs and unanimity: all PASS -> PASS; all FAIL -> FAIL; any mixture or unavailable required CI -> INCONCLUSIVE. No registered seed-averaged bootstrap can be reconstructed.

### P1

> **P1 (full beats none).** Per mode: mean over seeds of (full - none) > 0 AND the CI95 lower bound > 0. All three modes must pass.

| Mode | Seed | Difference | Paired CI95 | Per-seed verdict |
|---|---:|---:|---|---|
| s1 | 1 | 0.201188 | [0.182288, 0.220379] | PASS |
| s1 | 2 | 0.183069 | [0.164265, 0.201282] | PASS |
| s1 | 3 | 0.201913 | [0.183559, 0.220332] | PASS |
| s1 | mean / substitute | 0.195390 | not averaged | PASS |
| s2 | 1 | 0.112589 | [0.094959, 0.129577] | PASS |
| s2 | 2 | 0.100508 | [0.083681, 0.115751] | PASS |
| s2 | 3 | 0.092975 | [0.076156, 0.108586] | PASS |
| s2 | mean / substitute | 0.102024 | not averaged | PASS |
| s1s2 | 1 | 0.031036 | [0.018198, 0.044593] | PASS |
| s1s2 | 2 | 0.051380 | [0.036186, 0.065931] | PASS |
| s1s2 | 3 | 0.029969 | [0.016272, 0.043917] | PASS |
| s1s2 | mean / substitute | 0.037461 | not averaged | PASS |

Overall P1: **PASS**. Missing paired full−shuffle intervals are not replaced by differences between marginal CI endpoints.

### P2

> **P2 (full beats shuffle).** Per mode: mean over seeds of (full - shuffle) > 0 AND CI95 lower bound > 0. All three modes must pass.

| Mode | Seed | Difference | Paired CI95 | Per-seed verdict |
|---|---:|---:|---|---|
| s1 | 1 | 0.178263 | not saved | INCONCLUSIVE |
| s1 | 2 | 0.194615 | not saved | INCONCLUSIVE |
| s1 | 3 | 0.143628 | not saved | INCONCLUSIVE |
| s1 | mean / substitute | 0.172169 | not averaged | INCONCLUSIVE |
| s2 | 1 | 0.164244 | not saved | INCONCLUSIVE |
| s2 | 2 | 0.150024 | not saved | INCONCLUSIVE |
| s2 | 3 | 0.082788 | not saved | INCONCLUSIVE |
| s2 | mean / substitute | 0.132352 | not averaged | INCONCLUSIVE |
| s1s2 | 1 | 0.207972 | not saved | INCONCLUSIVE |
| s1s2 | 2 | 0.384596 | not saved | INCONCLUSIVE |
| s1s2 | 3 | 0.323356 | not saved | INCONCLUSIVE |
| s1s2 | mean / substitute | 0.305308 | not averaged | INCONCLUSIVE |

Overall P2: **INCONCLUSIVE**. Missing paired full−shuffle intervals are not replaced by differences between marginal CI endpoints.

### P3

> **P3 (chance).** A cell whose seed-mean mAP is below the pooled prevalence baseline for India out-of-fold patches, as computed in the run's JSON, is a reported failure. The threshold is the exact prevalence value from the run, not a rounded number.

Gate-eligible cells: **PASS**. All cells and their exact run prevalence, including diagnostic failures, are listed above.

### P4

> **P4 (regressions).** Any region where the seed-mean full mAP is below its seed-mean none mAP is listed by name. Reported, not a gate.

Reported, not gated. Every region/mode regression is named below.

### Gate

> **Gate.** PASS only if P1 and P2 pass in all three modes and no P3 failure exists. A tie or a CI that touches zero is a FAIL.

**INCONCLUSIVE**: P1=PASS, P2=INCONCLUSIVE, P3 (non-diagnostic arms)=PASS. This is not a PASS and does not authorize final models.

### Step 3

> **Step 3 (final models).** No pass/fail and no test metrics. Output is checkpoint, data hash, code commit, seed, epochs, region list.

NOT RUN. No final checkpoint, final-model accuracy, or deployment claim is produced.

## Fold-A optical anomaly and required checks

> 1. **Permutation scope.** Test: for every fold and mode, train labels are a permutation of the original train labels, val labels are a permutation of the original val labels, and test labels are byte-identical to the originals. A failure means the shuffle touched test data. (Existing test `test_shuffle_is_confined_to_train_and_val` covers this on the fixture; the check is repeated on the real fold indices.)

> 2. **Leakage.** Test: train, val and test index sets are pairwise disjoint per fold, and no test region appears in train or val. Also a sha256 of each patch's pixels: no exact duplicate between a test patch and any train/val patch.

> 3. **Seed persistence.** The gap (shuffle minus none) must hold on the seed mean (seeds 1, 2, 3), not only on seed 17.

> 4. **Candidate mechanism, to test not assume.** The shuffle arm also shuffles validation labels, so its epoch is chosen on shuffled validation. Record the selected epochs for shuffle and full. If they differ, the arms are not identically trained. The diagnostic arm below tests this directly.

Real-data index/permutation/pixel-hash audits: **PASS**. All 18 seed/mode/fold checks are in summary.json. Test-label preservation is reconstructed from saved permutations plus verified source, not from saved runtime label snapshots. No model was executed for these audits.

| Arm | Seed 1 [CI95] | Seed 2 [CI95] | Seed 3 [CI95] | Mean | Exact fold prevalence |
|---|---|---|---|---:|---|
| none | 0.576312 [0.552903, 0.601576] | 0.576312 [0.552903, 0.601576] | 0.576312 [0.552903, 0.601576] | 0.576312 | 0.35622222222222216 |
| shuffle | 0.582602 [0.564983, 0.603668] | 0.597907 [0.575938, 0.620829] | 0.659984 [0.643164, 0.680495] | 0.613498 | 0.35622222222222216 |
| shuffle-fixed | 0.367073 [0.350803, 0.387876] | 0.427041 [0.409576, 0.447521] | 0.409123 [0.392264, 0.427972] | 0.401079 | 0.35622222222222216 |
| bn-only | 0.688287 [0.674439, 0.705604] | 0.688383 [0.674445, 0.705651] | 0.687355 [0.673486, 0.704605] | 0.688008 | 0.35622222222222216 |
| full | 0.733637 [0.715059, 0.755826] | 0.726336 [0.709003, 0.747068] | 0.718647 [0.701598, 0.738583] | 0.726207 | 0.35622222222222216 |

Seed 17 re-read: none=0.5763124803967011, shuffle=0.6452245460444148, difference=0.06891206564771368. Seed-mean shuffle−none=0.037185054964395836; persistence check **PASS**.

### (d)

> Decision rule for the diagnosis, fixed now: "epoch-selection artefact" is claimed only if shuffle beats none in T13 AND shuffle-fixed does not beat none (CI lower bound <= 0). Otherwise the cause is recorded as unexplained or as named by the (b) checks.

Requested mean-based reading: **PASS** (seed-17 shuffle > none; shuffle-fixed seed mean <= none). This is a directional epoch-selection diagnostic, not a claim that an unavailable CI passed. The original CI predicate remains **INCONCLUSIVE**.

| Seed | shuffle-fixed−none | Paired CI95 | CI predicate |
|---:|---:|---|---|
| 1 | -0.209239 | not saved | INCONCLUSIVE |
| 2 | -0.149272 | not saved | INCONCLUSIVE |
| 3 | -0.167190 | not saved | INCONCLUSIVE |
### (e)

> Decision rule, fixed now: compare fold-A pooled mAP for s2, seed mean over seeds 1, 2, 3. If **bn-only >= shuffle-fixed**, BatchNorm re-estimation, not label noise, explains the fold-A anomaly. If bn-only < shuffle-fixed, the BatchNorm candidate is not supported as the whole explanation, and the cause stays as recorded under (b)/(d).


Mean-based verdict: **PASS**. bn-only=0.6880084555066773, shuffle-fixed=0.4010789691722225. Attribution: **BN re-estimation supported by rule (e)**. This is the preregistered mean inequality, not an independent causal estimate. The existing synthetic BN-invariance test checks unchanged weights/changed running stats; real before/after model states were not saved.


### Selected epochs (validation selection differs from fixed-final diagnostics)

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

## Region regressions (seed means)

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

Inclusivity is tested at full precision; display rounding never determines a verdict. Diagnostic arms have no seed-17 reference.

## T13 rules still in force (point rules on seed means)

> H1: The best adaptation arm recovers at least half the original Europe-minus-India
> mAP gap for each mode. Choose the best of renorm/head/full separately within each
> fold using adaptation-validation mAP only (ties: renorm, head, full). Pool its
> out-of-fold predictions for the final score. For each mode, let I0 be original
> India mAP, E0 original Europe mAP, and Ia selected-arm out-of-fold India mAP on
> the same five-class comparison set. Pass iff E0 > I0 and
> Ia - I0 >= 0.5 * (E0 - I0). Overall H1 passes only if all three modes pass.
> A nonpositive gap or unavailable comparison metric is reported as not evaluable,
> not a pass. Compute E0 and I0 from the original checkpoint on this pack; do not
> use rounded context numbers as thresholds. Include uncertainty without claiming
> that a point-threshold pass proves statistical significance.

> H2: Arm ranking is renorm < head < full. Pass per mode only if this strict ordering
> holds for out-of-fold India-pooled mAP, using each arm's validation-selected
> epochs. Ties fail. Overall H2 passes only if all modes pass. None is reported as
> an anchor, not included in this ordering.

> H3: Shuffled-label full controls stay at prevalence level. Operational tolerance:
> absolute difference between out-of-fold control mAP and the prevalence baseline
> must be <= 0.05 per mode; overall pass requires all modes. Report the difference
> and CI even if this fails. A pretrained checkpoint can retain predictive signal
> under shuffled fine-tuning, so this is a hypothesis, not a guaranteed sanity
> check. Shuffled-score chance columns are distinct from shuffled-label training.

The later T13 correction forbids arm selection: all renorm/head/full arms are reported independently. No best arm is selected.

| Mode | Arm | Mean gain | Half-gap threshold | H1 |
|---|---|---:|---:|---|
| s1 | renorm | -0.010775 | 0.085838 | FAIL |
| s1 | head | 0.027507 | 0.085838 | FAIL |
| s1 | full | 0.195390 | 0.085838 | PASS |
| s2 | renorm | -0.104835 | 0.169400 | FAIL |
| s2 | head | 0.074150 | 0.169400 | FAIL |
| s2 | full | 0.102024 | 0.169400 | FAIL |
| s1s2 | renorm | -0.162625 | 0.119379 | FAIL |
| s1s2 | head | -0.020478 | 0.119379 | FAIL |
| s1s2 | full | 0.037461 | 0.119379 | FAIL |

| Mode | H2 strict renorm<head<full | Shuffle mean−exact prevalence | H3 absolute gap <=0.05 |
|---|---|---:|---|
| s1 | PASS | 0.176795 | FAIL |
| s2 | PASS | 0.183112 | FAIL |
| s1s2 | PASS | 0.037041 | PASS |

## Inconclusive criteria — stop after this report

- P2/s1: paired full-minus-shuffle CI not saved
- P2/s2: paired full-minus-shuffle CI not saved
- P2/s1s2: paired full-minus-shuffle CI not saved
- (d): registered shuffle-fixed-minus-none CI predicate not saved; mean direction reported separately
- Overall T13b gate: P2 unresolved

No rerun is initiated. Soham decides whether to authorize one. Future NPZ export is independent code-only instrumentation (8e87936); its synthetic archive test passed, and no experiment was executed.

## Provenance

```json
{
  "manifest_sha256": "7dece9ad5296f3da61fa96be86a3923b8164ded6d052b250ee22d3d4b6ba0e51",
  "reference_json_sha256": "f5eb2569d343f3eab62bc65b19e9b3e6e773ea881d75454caef00000070fd447",
  "code_commit": "a0d7002cb9a43162e297f98a795694982791acb5",
  "seed_json_sha256": {
    "1": "827e98194eb3e7acd77fbb77ac8c54c84856c6b60d274f5c3cf1305917802bcf",
    "2": "5ae1dadbb20ef51d8d165681544a798c47cdb784de9d052351d1bd8ab5092389",
    "3": "c4e92c69fccc119a4cf6b8b2a871b95d398d52035fffb4a084758934e1b89478"
  }
}
```

All input JSON source hashes match a0d7002. Input seed logs end in DONE and SHA256SUMS.txt is verified; seed 17 matches the preregistered reference hash. Small summary.json contains all unrounded numbers and audit checks; no data or weights are committed.
