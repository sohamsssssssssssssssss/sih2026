# RESULTS: every MEASURED number
Each row was checked on 16 Nov 2026 with `git show <commit>:<path>`. A row marked VERIFY has no results file in git yet, so it must not be quoted until one is found.

| # | What | Value | n | Controls / baseline | Results file | Commit |
|---|---|---|---|---|---|---|
| R1 | BEN land cover, test macro mAP, s1 | 0.5598 [0.5512, 0.5785] | 6,000 | prior and shuffled in file | `results/ben-landcover/analysis.json` | 99f27ea (`feature/ben-landcover`) |
| R2 | BEN land cover, test macro mAP, s2 | 0.6254 [0.6121, 0.6483] | 6,000 | same | same | 99f27ea |
| R3 | BEN land cover, test macro mAP, s1s2 | 0.6283 [0.6193, 0.6492] | 6,000 | same | same | 99f27ea |
| R4 | B13 RGB adapter, RSVQA-LR val, macro acc | 0.6957 vs base 0.4683 | 389 Q / 89 images | adapter_constant (constant image) 0.5814 | `results/rgb-pilot-b13/SOURCE.md` (+ `rgb-b13-evaluation-01/result.json`) | 20596be |
| R5 | B13 counting accuracy | 0.24 real image, 0.24 constant image | 100 Q | the count score doesn't change when the image is removed | same | 20596be |
| R6 | CDVQA G3 test1, macro acc | adapter 0.605, base 0.395, majority 0.440 | 400 Q / 99 pairs | identical 0.557, swapped 0.530 | `results/cdvqa-g3/ANALYSIS.md` | 89fc5bf (`training/cdvqa-g0`) |
| R7 | CDVQA G3 test1, paired diffs (macro) | real−base +0.210 [+0.150, +0.267]; real−identical +0.048 [+0.002, +0.092]; real−swapped +0.075 [+0.033, +0.119] | 99 pairs | — | same | 89fc5bf |
| R8 | CDVQA G3 test2, macro acc | adapter 0.608, base 0.448, majority 0.453 | 400 Q / 100 pairs | identical 0.535, swapped 0.547 | same | 89fc5bf |
| R9 | CDVQA G3 test2, paired diffs (macro) | real−base +0.160 [+0.103, +0.225]; real−identical +0.073 [+0.031, +0.113]; real−swapped +0.060 [+0.018, +0.102] | 100 pairs | — | same | 89fc5bf |
| R10 | DIOR val detector counting, exact match @0.35 | 0.324 vs always-0 0.500 (best of all thresholds 0.364) | 500 | always-0, train-majority | `results/count-dior/ANALYSIS.md` | 1837259 (`feature/count-grounding`) |
| R11 | Qwen2.5-VL-3B zero-shot, RSVQA-LR | 0.565 acc | 200 | — | **VERIFY**: `results_v2.json` isn't in git history; commit bbbde2c only changes eval code | — |
| R12 | Phase 0 test count | 254 tests | — | — | **VERIFY**: the number is only in prose (`MASTER-PLAN-v5.md`, ANNEX-C), with no saved test output | 14e8c14a |

## Claim limits (copied from the SOURCE files)
- CDVQA G3 (R6–R9): NOT allowed: "uses temporal order", "detects change without text priors", any comparison to the paper's 73–75 %, anything about Cartosat/RISAT. `cdvqa_validated` stays false.
- DIOR counting (R10): the detector does not beat always-0 at any threshold from 0.25 to 0.40.
- §2.1's "CDVQA G1 base zero-shot 0.395 vs majority 0.44" equals G3 test1 base_real/majority (R6).
