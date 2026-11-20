# TRACKER
Status: done | in_progress | pending | open | unknown. Gates are copied from MASTER_PLAN.md Part 9.

| Id | Title | Status | Branch | Gate | Evidence |
|---|---|---|---|---|---|
| T01 | Push every local branch (no merges) | done | — | `git branch -r` lists all branches | All work branches on origin at local sha (16 Nov); CONTEXT.md §7 |
| T02 | BEN transfer Kaggle run; download outputs | done | feature/ben-landcover @ 4707d4c | `CELL DONE`; retrain within ~0.01–0.02 of committed | Retrain matched exactly; RESULTS.md §1a |
| T03 | Record CDVQA G3 result | done | training/cdvqa-g0 @ 89fc5bf | Numbers + controls in RESULTS.md | RESULTS.md §2, `results/cdvqa-g3/` |
| T04 | Create `docs/plan/` files | done | task/T04-plan-files @ 9bde585 | Files exist | Already merged in origin/main (39b58e7); T05 inventory confirms ancestry |
| T05 | Integration branch `integration/spine-v1` | done | integration/spine-v1 | Pytest 1,401 passed / 47 skipped / 0 failed. Vitest 50, tsc 0 diagnostics, Playwright 9 across live and offline. | [20 Nov verification](INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture); T09 merge 54ae066 and scoped T13 merge d6cb91f; no main merge |
| T06 | Shared contracts + T06b fixes | done | task/T06-schema @ 8042dae; T06b @ 23dbbbf | Contract tests pass | Integrated via T09 a45d1f9; current schema v1.3.0 |
| T07 | Tool registry (fail-closed) + adapters | done | task/T07-registry @ 7bfb179 | Registry status mapping and tests | Integrated by merge 54ae066; experimental status is preserved |
| T07b | Land-cover + flood-otsu adapters | done | task/T07b-tools @ b806750 | Experimental only | flood-otsu India-test IoU 0.631332 < 0.65: FAIL; experimental_unvalidated, never validated (report 7ffe130) |
| T08 | Router + answerability gate | done | task/T08-router @ 06adbc9 | Forty routing cases + refusal properties | Integrated via T09 a45d1f9; deterministic routing and hash-chained decisions |
| T09 | FastAPI `/ask` end to end with trace | done | task/T09-ask-api @ a45d1f9 | Real CPU API tests pass; trace verifies | Integrated by 54ae066; full-suite limitations in INTEGRATION_LOG.md |
| T10 | Streamlit on the API | pending | — | Old UI works on new backend | — |
| T11 | Analyse transfer results; 3 shift numbers | pending | — | Chart + RESULTS.md entries | Shift numbers already generated in RESULTS.md §1b; chart not made |
| T12 | VLM serving on Mac | pending | — | Decision in DECISIONS.md with numbers | — |
| T13 | India adaptation code + hypotheses | done | task/T13-india-adapt @ a0d7002; integration merge d6cb91f | Code compatibility: 49 targeted tests pass | Code/hypotheses merged; T13b registered results merged 3cec3da (results/t13b/REPORT.md, P1/P2 PASS). G1 passed (S1 0.717 / floor 0.689, S2 0.688 / floor 0.659). Platt and temperature scaling failed to transfer across region folds. G3 passed on corrected inputs, record only. India promotion closed, not shipped. [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md) |
| T14 | Flood pack builder from Sen1Floods11 | done | task/T14-flood-pack @ df4a236 | Verified builder/manifest tests | Already merged in origin/main (b97e70b); pack stays external, licence unconfirmed |
| T15 | Flood model vs Otsu per event | open | task/T15-flood-model (excluded) | Tiny-overfit/model gate unresolved | Not merged; no full-model validation claimed |
| T16 | Indian flood before/after scenes | pending | — | Scenes in golden set with masks + areas | — |
| T17 | Detector counting eval + thresholds | pending | — | Counting MAE + refusal rule MEASURED | DIOR baseline evidence 1837259 |
| T18 | Change baseline + CDVQA adapter decision | done | task/T18-change-baseline @ 1d0cbaf | Change tool registered with honest status | Change QA disabled by decision 252ea5e; DECISIONS.md 18 Nov; core/tool_status.json `disabled` |
| T19 | Verifier checks | pending | — | Agree + disagree test per check | — |
| T20 | Calibration + conformal abstention | pending | — | Selective acc + coverage on held-out India | — |
| T21 | Build + freeze SatQuery-Bench-India | pending | — | Bench hash recorded | — |
| T22 | Full bench + ablations + controls | pending | — | Results table in RESULTS.md | — |
| T23 | 12 golden scenes; offline golden test | pending | — | Passes twice, networking off | — |
| T24 | Next.js + MapLibre shell | done | integration/spine-v1 @ f7232d0 | Talks to `/ask` | Live S1/S2 Ask screenshots in docs/screens/p17/ |
| T25 | Overlays | pending | — | Golden scenes render | — |
| T26 | Evidence panel | pending | — | Every number traces to a file | — |
| T27 | Capabilities + known-failures pages | pending | — | Built from `/capabilities` + RESULTS.md | — |
| T28 | Offline bundle | pending | — | Works with Wi-Fi off | — |
| T29 | Rehearsal × 3 | pending | — | No failures in last 2 runs | — |
| T30 | Finale kit | pending | — | Checklist signed off | — |
| T31 | Q&A sheet | pending | — | Soham answers all without notes | — |
| T32 | Freeze, tag `finale-rc` | pending | — | Tag exists | — |

### Current Ask demo evidence

The Europe-trained BigEarthNet model serves uncalibrated scores. After the BOA offset fix (`84d7f57`), built areas ranked first in live Ludhiana (0.955) and Delhi (0.949), and cropland ranked first on the Punjab fixture (0.378). The dated results describe those scenes. The affected backups and cloudy-Mumbai fixture were regenerated in `404039f` and `980cab7`. [API verification](INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups)

Live acquisition has a 20-second overall deadline (`5a778ba`) and a 60-second S2 per-read timeout (`ecb651e`). At the deadline Ask identifies the matching cached scene and shows its acquisition date, or refuses if there is no matching cache. [Deadline verification](INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture)
