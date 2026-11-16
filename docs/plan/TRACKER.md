# TRACKER
Status: done | in_progress | pending | unknown. Gates are copied from MASTER_PLAN.md Part 9.

| Id | Title | Status | Branch | Gate | Evidence |
|---|---|---|---|---|---|
| T01 | Push every local branch (no merges) | done | — | `git branch -r` lists all branches | All work branches on origin at local sha (16 Nov); CONTEXT.md §7 |
| T02 | BEN transfer Kaggle run; download outputs | done | feature/ben-landcover @ 4707d4c | `CELL DONE`; retrain within ~0.01–0.02 of committed | Retrain matched exactly; RESULTS.md §1a |
| T03 | Record CDVQA G3 result | done | training/cdvqa-g0 @ 89fc5bf | Numbers + controls in RESULTS.md | RESULTS.md §2, `results/cdvqa-g3/` |
| T04 | Create `docs/plan/` files | in_progress | task/T04-plan-files | Files exist, CONTEXT matches repo | this branch |
| T05 | Integration branch `integration/h0` | pending | — | Full suite passes; merge to main with approval | — |
| T06 | `core/schema.py` + contract tests | done | task/T06-schema @ 8042dae | Tests pass | 57 core tests pass; full suite has the same failures and errors as before T06 (DECISIONS.md i) |
| T07 | Tool registry (fail-closed) + wrap tools | pending | — | `/capabilities` lists tools with status | — |
| T08 | Router + answerability gate | pending | — | 40 routing cases incl. 10 refusals | — |
| T09 | FastAPI `/ask` end to end with trace | pending | — | One golden scene via API, trace verifies | — |
| T10 | Streamlit on the API | pending | — | Old UI works on new backend | — |
| T11 | Analyse transfer results; 3 shift numbers | pending | — | Chart + RESULTS.md entries | Shift numbers already generated in RESULTS.md §1b; chart not made |
| T12 | VLM serving on Mac | pending | — | Decision in DECISIONS.md with numbers | — |
| T13 | India adaptation, 3 arms × 2 folds | in_progress | task/T13-india-adapt @ 8453495 | Before/after chart from MEASURED numbers | Hypotheses 3d1f3cf; Kaggle run VERIFY |
| T14 | Flood pack from Sen1Floods11 | in_progress | task/T14-flood-pack @ df4a236 | Manifest + pack tests pass | df4a236 records gate; awaiting review |
| T15 | Flood model vs Otsu per event | pending | task/T15-flood-model @ 57a919d (local) | Results per event | Hypotheses pre-registered 57a919d |
| T16 | Indian flood before/after scenes | pending | — | Scenes in golden set with masks + areas | — |
| T17 | Detector counting eval + thresholds | pending | — | Counting MAE + refusal rule MEASURED | DIOR baseline evidence 1837259 |
| T18 | Change baseline + CDVQA adapter decision | pending | — | Change tool registered with honest status | G3 controls in RESULTS.md §2 |
| T19 | Verifier checks | pending | — | Agree + disagree test per check | — |
| T20 | Calibration + conformal abstention | pending | — | Selective acc + coverage on held-out India | — |
| T21 | Build + freeze SatQuery-Bench-India | pending | — | Bench hash recorded | — |
| T22 | Full bench + ablations + controls | pending | — | Results table in RESULTS.md | — |
| T23 | 12 golden scenes; offline golden test | pending | — | Passes twice, networking off | — |
| T24 | Next.js + MapLibre shell | pending | — | Talks to `/ask` | — |
| T25 | Overlays | pending | — | Golden scenes render | — |
| T26 | Evidence panel | pending | — | Every number traces to a file | — |
| T27 | Capabilities + known-failures pages | pending | — | Built from `/capabilities` + RESULTS.md | — |
| T28 | Offline bundle | pending | — | Works with Wi-Fi off | — |
| T29 | Rehearsal × 3 | pending | — | No failures in last 2 runs | — |
| T30 | Finale kit | pending | — | Checklist signed off | — |
| T31 | Q&A sheet | pending | — | Soham answers all without notes | — |
| T32 | Freeze, tag `finale-rc` | pending | — | Tag exists | — |
