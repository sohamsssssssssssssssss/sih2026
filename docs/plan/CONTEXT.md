# SatQuery AI (SIH26167): project context
Last updated: 16 Nov 2026 (T04). Based on `SATQUERY_CONTEXT.md` (16 Nov), with every repo fact re-checked with git. Anything marked **VERIFY** could not be checked from the repo.

> **Superseded in parts (20 Nov 2026).** The GUI is now the Next.js `/ask` app, not Streamlit. The last recorded full suite is 1,366 passed, 47 skipped, 0 failed (`INTEGRATION_LOG.md`). T13/T13b are done and the India land-cover promotion gate failed. For current status see [docs/STATUS.md](../STATUS.md); the rest of this file is the 16 Nov snapshot.

## 1. What this is
- ISRO Smart India Hackathon 2026, problem SIH26167: Agentic AI-based Geospatial Intelligence for Multi-Sensor Satellite Data.
- Product: satellite-imagery VQA + grounding system (optical + SAR), Streamlit GUI, agentic orchestrator.
- Lead: Soham. Building the whole project **solo** since 15 Nov 2026 (6 on the team on paper).
- Repo: github.com/sohamsssssssssssssssss/sih2026. Local: `/Users/soham/sih2026`.
- Grand finale: December 2026, exact date unknown.

## 2. Core thesis
- About a 15x resolution gap: the training data is Sentinel-2 (10 m GSD), while the ISRO eval sensors (Cartosat-2S / RISAT) are sub-metre.
- The gap is measured with a resolution ladder: PSF blur → decimate → sensor noise. Naive bicubic downsampling is banned.

## 3. Rules
See `RULES.md` (MASTER_PLAN §0.3 + §0.2). Also from the context file:
- Smoke test on a Kaggle T4 before any big GPU job.
- Explicit approval is needed before any Kaggle upload and before deleting shards/data.
- Build until T-2 (no fixed code freeze).

## 4. Environment and compute
- MacBook, Apple Silicon, 16 GB RAM. Use `python3.11` for ML deps.
- The python3.11 env is missing `peft`, `streamlit` and `pyarrow`. With that env the full suite gives 987 passed, 7 failed and 12 import errors (task/T06-schema, 16 Nov); all of the failures and errors existed before T06. They will be fixed in T05.
- Compute: Kaggle T4 (free, one GPU session at a time). RTX access is gone.
- Agents: Claude Code + Codex. Prompts are delivered as .txt files.

## 5. Kaggle lessons
- Long jobs must be committed versions (Save Version → Save & Run All). Interactive sessions get cancelled.
- Keep big files out of `/kaggle/working`, or the output save hangs.
- One GPU session at a time; side jobs use accelerator None.
- Committed logs appear only when the cell ends.

## 6. Where things stand today (16 Nov)
| Stream | State | Pointer |
|---|---|---|
| BEN → WorldCover transfer (T02) | **Done.** Kaggle run finished; the retrain reproduces the committed BEN mAP exactly (s1 0.5598, s2 0.6254, s1s2 0.6283). | `RESULTS.md` §1. Outputs in `~/ben-work/transfer-run-1/` (outside git; checkpoints there too). Code 4707d4c. |
| CDVQA G3 (T03) | **Done.** Recorded at 89fc5bf. The adapter beats base and majority, but the controls show it mostly ignores the image pair. | `RESULTS.md` §2, `results/cdvqa-g3/` |
| Core schema (T06) | **Done** at 8042dae, pushed. | `core/schema.py`, `docs/schemas/` on task/T06-schema |
| India adaptation (T13) | In progress. Code at 8453495 (pushed). Running on Kaggle: **VERIFY**, because Kaggle state can't be seen from the repo. | task/T13-india-adapt |
| Flood pack (T14) | In progress. df4a236 (pushed) records a T14 gate; Soham hasn't reviewed it yet. The Sen1Floods11 licence is unclear (DECISIONS.md). | task/T14-flood-pack |
| Flood model (T15) | Hypotheses pre-registered at 57a919d (local branch, not on origin). | task/T15-flood-model |
| Counting (T17) | The DIOR detector does not beat always-0 (exact 0.324 vs 0.500). | 1837259, `results/count-dior/` |

## 7. Branches on origin (`git branch -r`, 16 Nov) and purpose
| Branch | Purpose |
|---|---|
| `main` (0fc4c40, 15 Oct) | Demo floor; nothing merged since 15 Oct |
| `task/T04-plan-files` | These plan files |
| `task/T06-schema` | T06 core data contracts |
| `task/T13-india-adapt` | T13 India adaptation |
| `task/T14-flood-pack` | T14 Sen1Floods11 pack |
| `feature/ben-landcover` | BEN land-cover baseline + WorldCover transfer code |
| `feature/count-grounding` | Count routing to Grounding DINO + DIOR count evidence |
| `analysis/count-presence-gate` | Offline presence-gate analysis on DIOR val |
| `feature/change-vqa` | Change-VQA route behind `SATQUERY_CHANGE_VQA` (off) |
| `app/b13-adapter` | B13 adapter in the app |
| `training/cdvqa-g0` | CDVQA G0–G3 training/eval + G3 results |
| `training/rgb-pilot-b12-b13` | B12/B13 RGB pilot + evidence |
| `cleanup/runner-reconcile` | RGB runner reconciliation docs |
| `keep/count-fix-547467b`, `keep/count-fix-99e7e60` | Safety copies of count-fix states. These are now on origin; I don't know who pushed them (**VERIFY**). |
| `dev`, `archive/*`, `feat/*`, `fix/*`, `part*`, `frontend-gui`, `lead/ladder-demo`, `migration/*`, `research/*`, `streak-fix`, `claude/relaxed-clarke-7c6e7y` | Older branches from before Nov; I didn't check what each is for (**VERIFY**) |

## 8. Worktrees (`git worktree list`, 16 Nov)
| Path | Branch | Purpose |
|---|---|---|
| `/Users/soham/sih2026` | training/cdvqa-g0 | Main checkout |
| `~/sih2026-docs` | task/T04-plan-files | T04 plan files |
| `~/sih2026-spine` | task/T06-schema | T06 |
| `~/sih2026-review` | detached at 8042dae | Review of T06 (by whom: **VERIFY**) |
| `~/sih2026-adapt` | task/T13-india-adapt | T13 |
| `~/sih2026-flood` | task/T14-flood-pack | T14 |
| `~/sih2026-floodmodel` | task/T15-flood-model | T15 |
| `~/sih2026-ben` | feature/ben-landcover | BEN code |
| `~/sih2026-count` | feature/count-grounding | Counting |
| `~/sih2026-gate` | analysis/count-presence-gate | Presence gate |
| `~/sih2026-change` | feature/change-vqa | Change-VQA |
| `~/sih2026-app-b13` | app/b13-adapter | B13 app |
| `~/sih2026-runner-reconcile` | cleanup/runner-reconcile | Runner docs |

## 9. Data outside git
| What | Path | Identity |
|---|---|---|
| BigEarthNet v2, ~111 GB (read-only) | `~/Datasets/BigEarthNet` | — |
| BEN pilot pack, 20,000 patches | `~/ben-work/pack` | MANIFEST sha256 `dc9c43ad…` |
| WorldCover transfer pack, 3,000 patches | `~/ben-work/wc/pack`, zip `~/ben-work/wc/ben-wc-transfer-pack.zip` | MANIFEST `5cc82c71…`; zip `5cc5f49c…` |
| Transfer run outputs | `~/ben-work/transfer-run-1/` | `SHA256SUMS.txt` there; JSON hashes in RESULTS.md |
| Flood data | `~/satquery-data` (download running, do not touch) | **VERIFY** |

## 10. Baselines and models
- VLM: Qwen2.5-VL-3B (pinned rev 66285546), frozen zero-shot baseline.
- RSVQA-LR zero-shot 0.565 (n=200, bbbde2c): **VERIFY**, because no results file is in git.
- The 0.1651 open_accuracy figure from Atharva's PR #1 is unconfirmed (**VERIFY**).
- Primary robustness metric: `open_accuracy`.

## 11. Earlier milestones (from the context file; not re-checked unless marked)
- SAR Phase-0 gate at 345b4a7 (checked: that commit edits `data/sar_gate/annotation_template.md` to remove overclaims).
- Phase 0 frozen at 14e8c14a (the commit exists; "254 tests" is only stated in prose: **VERIFY**).
- Stage-1 tiny-overfit OK. The earlier 150k train shards were invalid (pixel/ID mispairing).
- Grounding DINO zero-shot eval on DIOR-RSVG done (**VERIFY**).
- GUI: Streamlit audit parts 7–11. Branches `part7`…`part11` exist on origin; whether they're finished: **VERIFY**.

## 12. Key code paths
`eval/eval.py`, `eval/ladder.py`, `orchestrator/trace.py`, `data/sar_gate/`, `ben/` (on feature/ben-landcover), `core/schema.py` (on task/T06-schema).

## 13. Working style
- Agent prompts are .txt files with tight scope, an explicit "do not touch" list and standalone context.
- Manual steps go one step at a time.
- Claude Code + Codex split: see DECISIONS.md (f).
