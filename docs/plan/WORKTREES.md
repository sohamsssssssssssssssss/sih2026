# Branches and worktrees

Branch roles and worktree locations come from the 20 Nov 2026 audit. Counts reflect the audit snapshot. A worktree shown as `-suffix` is under `~/sih2026-suffix`.

- **IN-INTEG** covers the release branches and work already represented in integration.
- **RECORD** preserves training or evaluation history. Do not delete these branches.
- **DEAD** marks superseded or archived work with no release action.

| Branch and worktree | Role | Commits outside integration at audit time | Release disposition |
|---|---|---|---|
| `integration/spine-v1` (`~/sih2026-integ`) | IN-INTEG | Integration source | Released to `main` in PR #24. |
| `main` (local branch, no worktree) | IN-INTEG | Local main had 0 unique commits. On GitHub, `0fd9d6c` was empty and `cc42590` merged an earlier integration tree. | GitHub main carries the release merge from PR #24. |
| task/T04-plan-files (`-docs`) | IN-INTEG | 0 | None |
| task/T06-schema (`-spine`) | IN-INTEG | 0 | None |
| task/T07-registry (`-registry`) | IN-INTEG | 0 | None |
| task/T07b-tools (`-tools`) | IN-INTEG | 0 | None |
| task/T08-router (`-router`) | IN-INTEG | 0 | None |
| task/T09-ask-api (`-api`) | IN-INTEG | 0 | None |
| task/T13-india-adapt (`-adapt`) | IN-INTEG | 0 | None |
| task/T14-flood-pack (`-flood`) | IN-INTEG | 0 | None |
| app/b13-adapter (`-app-b13`, untracked IDE files) | IN-INTEG | 0 | None |
| feature/count-grounding (`-count`) | IN-INTEG | 0 | None |
| keep/count-fix-99e7e60 | IN-INTEG | 0 | None (the revoked GPU-cell change) |
| `task/T18-change-baseline` (`-changebase`) | RECORD | 5, including T18 preregistration and results | Do not delete. Only `86cdf97` and `1d0cbaf` were cherry-picked. The source branch also contains the earlier change-VQA history. |
| `feature/ben-landcover` (`-ben`) | RECORD | 9, including the India promotion gates and decision | Do not delete. Only `results/landcover-india-promotion/` and `docs/gates/` were copied, from `4695cd9`. |
| task/T15-flood-model (`-floodmodel`) | RECORD | 10 (`57a919d`…`f72523f`, T15 preregs, diagnostics, park decision) | Do not delete. U-Net parked. |
| analysis/count-presence-gate (`-gate`) | RECORD | 1: `de6157e` presence-gate analysis | Do not delete. Preserves the frozen DIOR validation analysis. |
| training/cdvqa-g0 (`~/sih2026`, dirty) | RECORD | 9 (`da6f50b`…`89fc5bf`, CDVQA G0–G3 training/eval) | Do not delete. Training and evaluation history. |
| training/rgb-pilot-b12-b13 | RECORD | 2: `da6f50b`, `20596be` | Do not delete. Training and evaluation history. |
| cleanup/runner-reconcile (`-runner-reconcile`) | RECORD | 6 (the 5 training commits above plus `aa77270` runner docs) | Do not delete. Contains the same training commits. |
| keep/count-fix-547467b | RECORD | 1: `547467b` count GPU-cell repo search | Do not delete. Preserves the revoked change. |
| feature/change-vqa (`-change`) | DEAD (superseded) | 3: `da7c636`, `695f58f` flag-gated CDVQA route, `8bd52a5` | Do not merge. Change QA was disabled by decision (`252ea5e`). Keep on origin. |
| origin/feat/video-integration | DEAD (superseded) | 36 (35 not patch-equivalent), Sep–Oct landing work | Integration carries the current cinematic landing. |
| origin/archive/resume-satquery, origin/archive/phase-a-manifest-sha | DEAD (archived) | 18 and 1 (pre-migration) | No action |
| origin/research/integration-v1-migration | DEAD (superseded) | 10 (pre-migration research notes) | No action |
| origin/feat/prototype-{assets,change,sar,system}, origin/feat/phase0-frontend-integration, origin/feat/frontend-migration | DEAD (superseded) | 5 each (frontend-migration 6) | No action |
| origin/dev, origin/frontend-gui, origin/streak-fix | DEAD (superseded) | 3, 1, 1 (Sep–Oct) | No action |
| origin/claude/relaxed-clarke-7c6e7y, origin/feat/{capability-orchestration, controlled-image-ingestion, gui-real-inference, phase2-flood-backend, phase3-question-understanding, request-planner, structured-execution-plan}, origin/fix/{download-resume-and-dep-pins, gui-golden-fresh-clone, model-execution-hardening, trace-restart-persistence, week0-fabrication-and-ci}, origin/lead/ladder-demo, origin/migration/post-phase0-satquery-core, origin/part{1,7,8,9,10,11}-* | IN-INTEG | 0 each | None |
