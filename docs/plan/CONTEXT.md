# CONTEXT: state of the repo
Last updated 16 Nov 2026 (T04). Every line below comes from a git command or a file listing, not from memory.

## main
`main` = `origin/main` = 0fc4c40 (15 Oct 2026, "feat(sar): integrate verified Kosi 2024 change artifacts"). Nothing merged since then.

## Branches with work not on main (all on GitHub at the same sha, checked 16 Nov)
| Branch | Local | Remote | Ahead of main | Last commit |
|---|---|---|---|---|
| `analysis/count-presence-gate` | de6157e | de6157e | 10 | count: offline presence-gate analysis on frozen DIOR val |
| `app/b13-adapter` | bcd0e23 | bcd0e23 | 2 | fix(vqa): route amount counts and record pinned Mac parity |
| `cleanup/runner-reconcile` | aa77270 | aa77270 | 6 | docs: reconcile RGB runners and document stage1 fixture recovery |
| `feature/ben-landcover` | 4707d4c | 4707d4c | 12 | ben: India transfer test (Kaggle cell, not yet run) |
| `feature/change-vqa` | 8bd52a5 | 8bd52a5 | 7 | chore(kaggle): DIOR-VOC fetch cell |
| `feature/count-grounding` | 1837259 | 1837259 | 9 | count: DIOR val detector-count evidence (below always-0 baseline) |
| `training/cdvqa-g0` | 89fc5bf | 89fc5bf | 9 | results: CDVQA G3 eval (test1/test2) + analysis |
| `training/rgb-pilot-b12-b13` | 20596be | 20596be | 2 | results: B13 RGB-VQA adapter evaluation (750 steps) |
| `task/T13-india-adapt` | 8a3ec42 | not pushed | (on top of 4707d4c) | docs: record approved T13 coarse-logit adaptation design |
| `task/T04-plan-files` | this commit | not pushed | 1 | these plan files |

Local-only on purpose: `keep/count-fix-547467b`, `keep/count-fix-99e7e60`.

## Worktrees
| Path | Branch |
|---|---|
| `/Users/soham/sih2026` | training/cdvqa-g0 |
| `/Users/soham/sih2026-adapt` | task/T13-india-adapt |
| `/Users/soham/sih2026-app-b13` | app/b13-adapter |
| `/Users/soham/sih2026-ben` | feature/ben-landcover |
| `/Users/soham/sih2026-change` | feature/change-vqa |
| `/Users/soham/sih2026-count` | feature/count-grounding |
| `/Users/soham/sih2026-gate` | analysis/count-presence-gate |
| `/Users/soham/sih2026-plan` | task/T04-plan-files |
| `/Users/soham/sih2026-runner-reconcile` | cleanup/runner-reconcile |

## Plan files on main before this branch
`docs/plan/MASTER-PLAN-v5.md` (13 Sep) plus ANNEX-A to E and PROTOCOL-01. v5 is superseded by `MASTER_PLAN.md` (see DECISIONS.md). Nothing was deleted.

## Data outside git (never commit)
| What | Path | Identity |
|---|---|---|
| BigEarthNet v2 (read-only) | `~/Datasets/BigEarthNet` | — |
| BEN pilot pack, 20,000 patches | `~/ben-work/pack` | MANIFEST sha256 `dc9c43ad…` |
| WorldCover transfer pack, 3,000 patches (10 regions × 300) | `~/ben-work/wc/pack` | MANIFEST sha256 `5cc82c713ad76a3286afd4ad3eb66f33da3fb7ff365ab23a6371dc3ba999c40d` |
| WorldCover pack zip (566 MB) | `~/ben-work/wc/ben-wc-transfer-pack.zip` | sha256 `5cc5f49c76c78cfacbb42cf38066c77a31af208b2d911bfe0c425ea79198befd` |
| Transfer code zip | `~/ben-work/satquery-ben-transfer-code.zip` | sha256 `f6f5351163ac99e26f3a0a29cf3681c07c1ce16bbdce3ead1b9029b02682ed8d` |
| Transfer build log | `~/ben-work/wc/LOOP_LOG.md` | — |

## Pending
- **T02 BEN → WorldCover transfer**: code at 4707d4c (`ben/run_transfer_gpu.txt`). Not run yet. Kaggle steps are in `~/ben-work/wc/LOOP_LOG.md`: attach exactly 3 inputs, and do not attach the old code zip.
- **T03 CDVQA G3**: results are committed at 89fc5bf (`results/cdvqa-g3/`) and recorded in RESULTS.md. Controls (identical, swapped) were run.
- **Corrections to MASTER_PLAN §2**:
  - §2.2 says "G3 result not recorded"; it is recorded at 89fc5bf.
  - §2.2 says "detector counting unmeasured"; it is measured at 1837259, and the detector is below the always-0 baseline.
  - §2.2/§2.3 say branches are not pushed; all 8 are on GitHub.
  - §2.3 says "five feature branches"; there are 8.
