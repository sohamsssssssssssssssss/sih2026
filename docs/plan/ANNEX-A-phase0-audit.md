# SatQuery AI — Phase 0 Repository Audit

**Repo:** `/Users/atharva/SIH` · branch `main` @ `93b21b0` · audited 13 Sep 2026
**Method:** read the code, ran the services, probed the API. Every row below is either executed or read. Nothing is inferred from documentation.
**Machine:** macOS, no CUDA. `backend/.venv` = Python 3.13.0 (repo's `.python-version` says `3.11`).

---

## 1. Capability inventory

| Capability | Status | Evidence | Gap |
|---|---|---|---|
| **Model contract (`Model.infer`)** | **VERIFIED** | `models/base.py:6-15` — ABC, typed, returns `answer`/`confidence`/`evidence`. | Contract is 15 lines. No batching, no async, no error taxonomy. |
| **Mock model** | **VERIFIED** | `models/mock/model.py:12-17` — returns `f"mock answer for: {question}"`, `confidence: 0.5`. | Zero tests of its own. It is a string formatter. |
| **Qwen2.5-VL-3B wrapper** | **PARTIAL** | `models/qwen_vl/model.py` — 106 lines, fully typed, lazy `_load()`, `requires_grad_(False)`, greedy decode, `torch.inference_mode()`. Real transformers code. | **Never executed anywhere in this repo's history on this machine.** `_load()` raises at `model.py:34-37` without CUDA. Every test patches `_load` or `infer` (`demo_gui/test_failure_paths.py:68,80,93`; `demo_gui/test_golden_path.py:47,87,179`; `orchestrator/test_planner.py:224`). No test executes `_generate_answer()`. Only ever proven on a Kaggle T4 (see results artifacts). |
| **Any 3rd model implementation** | **MISSING** | `models/` contains exactly `base.py`, `mock/`, `qwen_vl/` — 153 source lines total. | — |
| **Grounding model (Phase 3)** | **MISSING** | `orchestrator/capabilities.py:27` lists `GROUNDING` in `UNAVAILABLE_CAPABILITIES`. No code. `GET /api/capabilities` returns `{"name":"grounding","available":false,"provider":null}`. | Vocabulary only. |
| **Change-detection / Change-VQA (Phase 4)** | **MISSING** | `orchestrator/capabilities.py:28` `CHANGE_VQA` unavailable. No model, no dataset (see §4). | Vocabulary only. `eval/suites/RECON.md:82-87` calls cdvqa "a track blocker, not a missing suite". |
| **Optical–SAR fusion (Phase 5)** | **MISSING** | `orchestrator/capabilities.py:29` `OPTICAL_SAR` unavailable. No model, **no SAR pixels anywhere on disk**. | Vocabulary only. See §4 — this is the hardest gap. |
| **Deterministic request planner** | **VERIFIED** | `orchestrator/planner.py` (333 lines), 41 tests. Probed live: 5 distinct rules fire correctly (`default_single_image_vqa`, `grounding_spatial_localization`, `change_temporal_compare`, `optical_sar_cross_modal`, `temporal_change_then_grounding`). | Pure regex/token rules. No LLM planning. Fine for Phase 0; will not generalise. |
| **Structured execution plan** | **VERIFIED** | `orchestrator/execution_plan.py` (197 lines), 18 tests. Probed live: the combined question emits 2 steps with `step_2.depends_on = ["step_1"]`. | `required_inputs` is symbolic strings (`"step_1.output"`). No artifact-passing framework — documented honestly at `execution_plan.py:33-35`. |
| **Plan executor** | **PARTIAL** | `orchestrator/executor.py` (78 lines), 8 tests. All-or-nothing validation before any provider runs (`executor.py:44-55`). | Executes **exactly one shape**: a single `single_image_vqa` step (`executor.py:54-55`, `_assert_single_vqa_shape`). Multi-step plans are representable but not executable. |
| **Capability provider registry** | **VERIFIED** | `orchestrator/capabilities.py` (175 lines), 10 tests. Lock-guarded, refuses double-binding (`:96-106`), refuses providers advertising unimplemented capabilities (`Provider.__post_init__:65-69`). | 1 provider, 1 capability. |
| **Model registry** | **VERIFIED** | `orchestrator/registry.py` — 29 lines, name-keyed, type-checked at `register()`. | No versioning, no unregister. Registration is an import side-effect (`:28-29`). |
| **Capability router + trace owner** | **VERIFIED** | `orchestrator/router.py` (181 lines). Validates model output, forbids caller params from forging `capability`/planner/plan metadata (`router.py:134-164`), enforces `execution_mode == "live"`. | Timeout uses a 1-worker `ThreadPoolExecutor` — a timed-out inference is abandoned, **not cancelled** (honest `ponytail:` comment at `router.py:13`). A hung GPU call blocks every later request. |
| **Hash-chained execution trace** | **VERIFIED** | `orchestrator/trace.py` (117 lines), 16 tests. SHA-256 chain, `hmac.compare_digest` comparisons, fails closed on any malformed line (`:66-88`). Probed live: `POST /api/traces/verify` → `{"verified":true,"message":"Chain verified (18 records)"}`. | No signature/HMAC key — a full-file rewrite recomputes cleanly. It detects tampering, not forgery. |
| **Trace persists across restart** | **VERIFIED** | `_ensure_loaded()` re-reads and re-verifies `trace.jsonl` on first touch (`trace.py:53-90`). Confirmed: 18 records survive backend restarts. | — |
| **Backend API (10 endpoints)** | **VERIFIED** | All read-only endpoints probed live on port 8931. `/api/health`, `/api/capabilities`, `/api/resolution`, `/api/sar/{scene}`, `/api/plan`, `/api/traces`, `/api/traces/verify` all return correct payloads. 76 tests in `backend/test_api.py`. | `backend/requirements.txt` **omits `pillow`** although `backend/services.py:16` imports PIL. A fresh venv built per `backend/README.md` cannot import the app. |
| **Scene upload / ingestion** | **PARTIAL** | `backend/services.py:144-202`. Real decompression-bomb guard (`:148-149`), PNG/JPEG allowlist (`:152`), 20 MiB cap (`routes/analyze.py:29,34-36`), uuid4 id, atomic `.tmp` → `replace()`. | **No SHA-256 checksum** of stored bytes. **No frontend path calls it** — `components/imagery/UploadScene.tsx` is a permanently `disabled` button. `sensor`/`gsd`/`location`/`acquisition_date` are hardcoded `None` in the response schema (`schemas.py:26-29`), so an uploaded scene can never be planned against a sensor. |
| **Live inference path end-to-end** | **PARTIAL (never exercised)** | `analyze_scene()` → `execute_plan()` → `route()` → `model.infer()` is wired and covered by mocked tests. | **All 18 records in `trace.jsonl` are `execution_mode: "cached_result"`. Zero `live` records have ever existed on this machine.** The live branch is code-reviewed, not run. |
| **Golden cached fallback** | **VERIFIED** | `services.py:98-126,205-245`. Scoped to exactly one scene + one question (`GOLDEN_SCENE_ID`/`GOLDEN_QUESTION`, `:52-53`) and refused for any multi-step or unavailable-capability plan (`is_golden_eligible_plan`). | The demo's only working answer is one cached string. |
| **Resolution ladder generation** | **VERIFIED** | `eval/ladder.py` (193 lines). Real Gaussian PSF → BOX decimation → Poisson shot + Gaussian read noise (`:45-77`). Native rung is a true zero-degradation anchor (`:52-54`). 1,000 manifest rows on disk = 200 sources × 5 rungs. | **Zero tests.** `degrade()`, `label_answers()`, `find_loveda_pairs()` are untested. `eval/test_ladder_rungs.py` tests the *guard*, not the *generator*. |
| **Degeneracy guard (two-sided)** | **VERIFIED** | `eval/eval.py:137-168`, applied per-rung at `:281-306`. 13 + 14 tests. Confirmed on the committed artifact: rungs 5.0 and 10.0 flagged. | Catches collapse, not bias — stated honestly at `eval.py:144-147` and README:93-96. |
| **Sample-size guard** | **VERIFIED** | `eval.py:229-241`, `MIN_OPEN_SAMPLES = 500`. Computed before inference from gold answers. | — |
| **Result provenance (git SHA / GPU / timestamp)** | **PARTIAL** | `eval.py:171-187,253-272` stamps `git_sha`, `gpu`, `timestamp`, `config` into every report. Verified in all three `results/*.json`. | **No dataset version or checksum in the report.** No `run_id`. `outputs/results_ladder_qwen_full_v2.json` (the source predictions for the committed rescore) has **none** of these fields — only `accuracy`/`model`/`n_samples`/`per_rung`/`suite`. |
| **RSVQA-LR acquisition** | **VERIFIED** | `eval/suites/rsvqa.py` — per-file size + MD5 pins (`:15-28`), zip-slip guard (`:108-110`), resumable curl with stall detection (`:39-71`). All 12 files present on disk at the pinned sizes. | — |
| **Ladder suite** | **VERIFIED** | `eval/eval.py:190-208`, manifest on disk with 1,000 rows. | — |
| **`vrsbench` / `cdvqa` / `resolution_proxy` suites** | **MISSING (named, not real)** | `eval/suites/__init__.py:10` lists them in `SUITE_NAMES`; `:18-24` returns **one placeholder sample whose `expected_answer` is byte-identical to `MockModel`'s output**. | See §3 — this makes the README's own quickstart command produce a fake 100 % score. |
| **Streamlit fallback GUI** | **VERIFIED** | `demo_gui/app.py` (576 lines). Started headless on :8932 → HTTP 200. Module executes top-to-bottom cleanly in bare mode. 35 tests across 8 files. Calls `route()` (`app.py:138`), not `model.infer()`. | Forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` before registry import (`:12-13`). Capability lists are hardcoded tuples (`:34-46`), not read from `capabilities_status()`. |
| **Next.js frontend** | **PARTIAL** | 34 tracked source files, 706 lines. `npx tsc --noEmit` → clean, exit 0. | **Zero tests, no test runner, no test script in `package.json`.** `npm run lint` is broken — `next lint` was removed in Next 16, so it resolves as `next <dir>`: *"Invalid project directory provided, no such directory: .../frontend/lint"*. See §3 for what the UI actually does. |
| **Confidence / uncertainty calibration** | **MISSING** | Exhaustive grep for `confidence\|calibrat\|uncertain\|entropy\|logprob\|temperature_scal\|ece\|brier` across all `.py`/`.ts`/`.tsx`/`.md`. Only hits: the literal field in `Model.infer`'s contract, `MockModel`'s hardcoded `0.5`, Qwen's hardcoded `1.0` with the comment *"Placeholder only: frozen generation does not provide calibrated confidence"* (`models/qwen_vl/model.py:103-104`), and two UI strings reading **"Confidence calibration pending"** (`demo_gui/app.py:369`, `frontend/components/analysis/AnalysisResult.tsx:10`). | **No calibration code exists.** Worse: `confidence` is silently *dropped* — `backend/schemas.py:37-43` `AnalyzeResponse` has no `confidence` field, and `services.py:262-269` `_live_response()` does not copy it. The number never reaches any consumer. This is the plan's "real confidence/uncertainty where claimed" requirement at zero. |
| **CI** | **MISSING** | No `.github/` directory. | — |
| **Docker** | **MISSING** | No `Dockerfile`, no compose file. | — |
| **SQLite / job engine** | **MISSING** | No DB anywhere; state is `trace.jsonl` + committed JSON. | — |
| **SAR imagery / preprocessing output** | **MISSING** | `data/sar_gate/rendered/` **does not exist**. Live probe: `GET /api/sar/mumbai/image` → **404**; `GET /api/sar/mumbai` → `"render_available": false`. | SAR is 417 lines of human prose (§4). |

---

## 2. Per-directory breakdown

### `models/` — 153 source lines, no tests of its own
```
models/__init__.py          5    re-export
models/base.py             15    ABC: infer(image_paths, question) -> dict
models/mock/model.py       17    canned string
models/qwen_vl/model.py   106    real Qwen2.5-VL-3B wrapper
```
The Qwen wrapper is **real and typed**: lazy weight load, explicit ImportError→RuntimeError translation naming the exact deps (`:30-33`), hard CUDA gate (`:34-37`), `eval()` + `requires_grad_(False)` (`:43-45`), file-URI image content, greedy decode with `max_new_tokens=50`, correct prompt-token trimming (`:83-86`). It appends *"Answer with a single word or number only. No explanation."* (`:99`) and validates image paths before touching the GPU (`:96-98`).

**Nothing besides `mock` has a runnable `infer()` on this machine.** There is no grounding, change-detection, fusion, segmentation, or SAR model code of any kind. `models/.DS_Store` is committed noise (untracked).

### `data/` — 13 GB on disk, 5 tracked source files
```
data/dataset.py              52   TileSample TypedDict + validate_tile_sample()
data/download_ladder_data.py 213  LoveDA (md5-pinned) + DOTA (gdown, best-effort)
data/sar_gate/order_scenes.py   193  ASF search + HyP3 RTC submission
data/sar_gate/process_scenes.py 314  gamma0→dB, Lee filter, VV/VH/VV−VH quicklook
data/BEN_TXT_FACTS.md         —   BigEarthNet.txt reconnaissance
```
`TileSample` already declares the Phase 4/5 slots — `sar: float32[3,H,W] | None` and `optical_t2` — and `validate_tile_sample()` enforces dtype/ndim/channel count (`dataset.py:37-48`). The schema is ready; the pixels are not. `load_tile()` (`:52-54`) is an explicit stub.

`data/dataset.py` and `data/download_ladder_data.py` have **zero tests**. So do both `sar_gate` scripts.

### `eval/` — 2 real suites, 3 fake ones
| file | lines | tests | what it does |
|---|---|---|---|
| `eval.py` | 326 | 39 (across 4 files) | CLI runner, `answer_matches`, `summarise`, two-sided `degenerate`, per-rung stratification, provenance stamping, dual-write to `--out` **and** `results/` |
| `ladder.py` | 193 | **0** | GSD degradation generator |
| `plot_ladder.py` | 108 | **0** | stratified curve, hollow markers on degenerate rungs |
| `smoke.py` | 160 | 9 (`check_gpu` only) | 50-step CPU train loop + T4/bf16/flash-attn/VRAM gates |
| `suites/rsvqa.py` | 176 | 2 | official RSVQA-LR acquisition + loader |
| `suites/__init__.py` | 25 | 1 | dispatcher; **3 of 5 names are placeholders** |
| `suites/RECON.md` | — | — | per-suite splits, licences, blockers. High-value; verified against Zenodo/HF on 2026-09-02/03 |

**What the ladder measures:** two questions per tile — *"What is the dominant land cover in this image?"* (open-ended, gold derived from the LoveDA mask's modal class) and *"Is there a building in this image?"* (binary, gold = `any(mask == 2)`) — across five GSD rungs (0.3/1/2/5/10 m) produced by PSF blur → area-average decimation → Poisson+Gaussian noise from real LoveDA pixels. `sensor` stays `"loveda"`, never `"synthetic"`.

**`eval.py` supports:** `--model {mock,qwen2.5vl-3b}` `--suite {rsvqa,vrsbench,cdvqa,ladder,resolution_proxy}` `--out PATH` `--limit N` (default 200) `--full`. Only `rsvqa` and `ladder` resolve to real data.

`eval/suites/__init__.py:18-24` — the placeholder for the other three:
```python
question = f"placeholder question for {name}"
return [{"image_paths": [f"data/{name}_placeholder.png"],
         "question": question,
         "expected_answer": f"mock answer for: {question}"}]
```
That `expected_answer` is exactly what `MockModel.infer` returns. **Any `--model mock --suite {vrsbench,cdvqa,resolution_proxy}` run scores 1.0 by construction.**

### `orchestrator/` — the strongest directory
2,110 lines / 809 of them tests / 93 tests. Registry → capabilities → planner → execution_plan → executor → router → trace, each with a single responsibility and a docstring that states its boundary (`executor.py:3-8` is exemplary). Live probe confirms all five planner rules and the two-step decomposition. The router's anti-forgery param filtering (`router.py:134-164`) is genuinely careful work.

### `backend/` — 1,256 test lines to 714 source lines
`main.py` (CORS pinned to localhost:3000/127.0.0.1 + a localhost regex), `schemas.py` (Pydantic, `max_length=2000` on question), `services.py` (459 lines — the whole live/cached boundary), 4 route modules. Offline env vars are set **before** the registry import (`services.py:19-20`) — correct ordering, easy to break.

Every exception path maps to a distinct HTTP status with a message that leaks nothing (`routes/analyze.py:73-98`; `routes/traces.py:7-10` returns a generic string and `test_api.py:144,174` assert the trace path never appears in a body).

### `frontend/` — 706 lines, 34 files, 0 tests
Next 16.3.4 / React 19.2.8 / Tailwind 3.4 / TS 7.0. `tsc --noEmit` clean. Routes: `/` → `/workspace`, `/resolution`, `/sar`, `/executions`, `/system`. Real recharts ladder curve with `ReferenceDot` for degenerate rungs, real maplibre-gl canvas overlaying the golden scene image, real trace drawer.

`lib/api.ts` calls exactly 5 endpoints: `/api/analyze` (scene_id **hardcoded** to the golden scene, `:22`), `/api/resolution`, `/api/sar/{scene}`, `/api/traces`, `/api/traces/verify`, `/api/health`. It **never** calls `/api/plan`, `/api/capabilities`, or `/api/scenes`.

### `demo_gui/` — still runs
Three tabs (Ask / Resolution Robustness / SAR Validation). Started on :8932 → HTTP 200; module imports and executes cleanly. `execute_query()` (`app.py:124-155`) tries `route()` first and falls back to the committed cached row only on `RuntimeError`/`OSError`, tagging the notice with the reason. It never fabricates: no cached match + no pixels → explicit error (`:137-143`).

### `scripts/` — 73 lines, both live
- `dry_run_rsvqa.py` (41) — real contract check: loads 2 real RSVQA samples, asserts the 256×256 image, asserts the prompt suffix, asserts **`model._model is None`** afterwards so the dry run provably never loaded weights (`:35`). Good.
- `train.py` (32) — loads YAML, looks up the model, finds no `train` attribute, prints *"has no training implementation"*, returns 0. **It is a no-op today** (no model in the repo has a `train` method), but it is 32 lines and it's the declared Stage-1 entry point. Keep.

### `kaggle/` — 252 lines, the only path to real GPU numbers
`run_rsvqa_baseline.py` (68) and `run_ladder_baseline.py` (188). Both `assert torch.cuda.is_available()` in cell 1 before spending anything. The ladder runner handles two mount modes, resolves versioned Kaggle dataset paths, symlinks read-only inputs, and prints per-rung numbers with `<-- DEGENERATE, not a measurement` inline. Untested but well-shaped; this is how every committed number was produced.

### `configs/` — one 7-line YAML
`example.yaml`: `model_name: mock`, `backbone: tiny-linear`, `lr: 0.05`, `batch_size: 4`, `max_steps: 50`, `data_path: data/`, `suite: resolution_proxy`. Consumed by `eval/smoke.py` and `scripts/train.py`. `suite: resolution_proxy` points at the placeholder suite.

### `src/` — **entirely dead**
```
src/sih/{__pycache__,data,eval,models}/__pycache__/*.pyc
```
No `.py` files. No `__init__.py`. **Zero files tracked by git.** It is nothing but 44 KB of orphaned bytecode from a deleted `src/sih` package (config.py, data/loader.py, data/suites.py, eval/metrics.py, models/registry.py — all superseded by the top-level `data/`, `eval/`, `orchestrator/`).

### `notebooks/` — empty. Zero files, tracked or untracked.

### `results/` — the audit trail
| file | tracked | what it claims |
|---|---|---|
| `qwen2.5vl-3b__rsvqa__20260903T175900Z.json` (3.4 MB) | yes | Legacy RSVQA-LR full test result, n=10004, acc 0.513095, open 0.165080, yes-rate 0.3635. Checkpoint revision was unrecorded and is unrecoverable; do not compare new engineering-checkpoint results to it. `git_sha 777707a`, `gpu Tesla T4`. |
| `qwen2.5vl-3b__ladder__rescored__20260904.json` (892 KB) | yes | Ladder, n=2000, per-rung stratified, `degenerate_rungs: ["5.0","10.0"]`. `git_sha c54da56`, `gpu "Tesla T4 (original inference run)"`, plus a `provenance` string stating it is a **re-score of stored predictions, not new inference**. |
| `ladder_curve_qwen_stratified.png` | yes | the README figure |
| `RESCORE_NOTE.md` | yes | explains the 0.5131→0.5142 delta, names the 11 flipped samples, and is pinned by `eval/test_baseline_rescore.py` |
| `mock__resolution_proxy__20260913T175929Z.json` | **untracked** | **`accuracy: 1.0`, `n_samples: 1`, model `mock`** — output of the README quickstart against the fake suite. Delete. |

**Reproducibility:** `git_sha` ✓, `gpu` ✓, `timestamp` ✓, `config` ✓, `provenance` ✓ (rescore only). **`run_id` ✗, dataset version/checksum ✗.** A reader cannot tell from a report which RSVQA download or which ladder seed produced it.

### `outputs/` — tracked despite `/outputs/` in `.gitignore`
`results_ladder_qwen_full_v2.json` (890 KB) and `ladder_curve_qwen_full_v2.png` were committed in `daab09b` *before* the ignore rule, so they remain tracked (`.gitignore` does not untrack). This matters: it is the **source of truth for the rescored ladder**, and it carries **no `git_sha`, no `gpu`, no `timestamp`** — only `{accuracy, model, n_samples, per_rung, suite}`. `outputs/ladder_curve.png` (Sep 7) is untracked and unreferenced.

### `reviews/` — 5 historical audit reports + a README
`reviews/reverification/` documents defects found in `a62c169` and the fixes. Explicitly labelled as *"unchanged historical audit reports for a62c169, not claims about the fixed revision."* Honest and worth keeping. Note it records validation on **Python 3.14 / Streamlit 1.52.2 / 24 tests** — now stale (35 demo_gui tests today).

---

## 3. Documentation claims vs reality

### README.md

| # | Claim | Verdict |
|---|---|---|
| 1 | `:15` — *"`eval/` owns authoritative metrics … using **exact case-insensitive stripped answer matching**"* | **FALSE.** `eval/eval.py:45-72` `answer_matches()` does far more: exact match, **`"1"` ≡ `"yes"`**, **word-boundary regex substring search** (`re.search(rf"\b{expected}\b", raw)`), and **number-word→digit normalisation** for zero–ten. A prediction of *"yes, there are two buildings"* matches gold `"two"`. This is the single most load-bearing false statement in the docs — it describes a stricter scorer than the one that produced every committed number. `RESCORE_NOTE.md` documents the change; the README invariant was never updated. |
| 2 | `:25` — *"`python3 eval/eval.py --model mock --suite resolution_proxy --out results.json`"* as the quickstart | **MISLEADING.** `resolution_proxy` returns one placeholder sample whose `expected_answer` is byte-identical to `MockModel`'s output (`eval/suites/__init__.py:18-24` vs `models/mock/model.py:13-17`). The command **always prints accuracy 1.0** and writes a committed-looking artifact into `results/`. `results/mock__resolution_proxy__20260913T175929Z.json` on disk right now is exactly that: `"accuracy": 1.0`. A judge running the README's first command sees a perfect score from nothing. |
| 3 | `:46-49` — *"`data/sar_gate/rendered/mumbai_coastal.png` … The render is optional; a missing render is reported in the UI."* | **TRUE but the file does not exist.** `data/sar_gate/rendered/` is absent entirely. Live probe: `GET /api/sar/mumbai/image` → 404, `"render_available": false`. The graceful-degradation claim holds; the asset has never been produced. |
| 4 | `:3` — *"Python 3.11 (recorded in `.python-version`)"* | **TRUE of the file, FALSE of the environment.** `.python-version` = `3.11`; `backend/.venv` = **3.13.0**; all `__pycache__` is `cpython-313`. `reviews/reverification/README.md` claims validation on **3.14**. Three Python versions in play. |
| 5 | `:5-19` "Folders and invariants" | **INCOMPLETE.** Lists `configs/ data/ models/ orchestrator/ eval/ scripts/ demo_gui/`. Omits **`backend/` and `frontend/`** — the primary interface — and `kaggle/`, `results/`, `outputs/`, `reviews/`. The README predates the migration and was never merged with `FRONTEND_MIGRATION.md`. It also never mentions `src/` or `notebooks/`, which are dead/empty. |
| 6 | `:11` — *"`infer(...)` must always return `answer`, `confidence`, and `evidence`"* | **TRUE at the model layer, VIOLATED downstream.** Both models return all three. But `confidence` is dropped at the API boundary: `backend/schemas.py:37-43` has no `confidence` field, `services.py:262-269` doesn't copy it. The invariant is enforced where it doesn't matter and abandoned where it does. |
| 7 | `:13` — *"every routed call must append a trace record"* | **TRUE.** `router.py:165-178`, unconditional, and `TracePersistenceError` propagates rather than being swallowed. |
| 8 | `:19` — *"`demo_gui/` must call the router—not a model directly"* | **TRUE.** `app.py:138` calls `route()`; the only `get()` use (`:93`) reads `model.version` for trace metadata. No `.infer()` call in `demo_gui/`. |
| 9 | `:61-69` ladder table + `:164-169` baseline table | **TRUE.** Every figure reproduced from the committed JSONs. Ladder per-rung acc/open/binary/yes-rate and `degenerate_rungs: ["5.0","10.0"]` match exactly. Baseline 0.5142/0.6671/0.1651/0.3675 vs the file's own 0.5131/0.6655/0.1651/0.3635, with the delta explained in `RESCORE_NOTE.md` and pinned by `eval/test_baseline_rescore.py`. This is the most carefully honest documentation in the repo. |
| 10 | `:55` — *"DOTA acquisition can be attempted with `--dataset dota`"* | **TRUE but never succeeded.** `data/raw/dota` does not exist. Path is gdown-on-Google-Drive with a manual-download fallback message (`download_ladder_data.py:174-198`). |
| 11 | `:110` — *"LoveDA is ~6.5 GB"* | **UNDERSTATED.** 6.0 GB of zips **plus** 6.0 GB extracted = **12 GB** on disk. |

### FRONTEND_MIGRATION.md

| # | Claim | Verdict |
|---|---|---|
| 12 | `:30` — *"A: responsive command-center shell and scene workspace"* | **TRUE.** 5 routes, Tailwind responsive classes throughout. |
| 13 | `:31-32` — *"B: Resolution Lab driven by the committed ladder artifact / C: SAR Validation driven by the committed analyst annotation"* | **TRUE.** Both fetch from the API, which reads the committed files. |
| 14 | `:33` — *"D: golden Qwen analysis with explicit live/cached provenance and safe no-GPU fallback"* | **TRUE, but "golden" is doing heavy lifting.** `QueryPanel.tsx:11,30` hardcodes the question **and renders the textarea `readOnly`**; `lib/api.ts:22` hardcodes `scene_id`. The user cannot ask anything. It is a one-button demo of one cached string. |
| 15 | `:34` — *"E (partial…): uploads remain visibly marked in development"* | **TRUE and correctly labelled** — but note the asymmetry: the **backend upload endpoint is real and fully tested**, and the frontend simply never calls it. `UploadScene.tsx` is a `disabled` button. Working capability, zero reach. |
| 16 | `:3-5` — *"without changing the existing Streamlit fallback or any model, orchestration, evaluation, data, or committed result source"* | **TRUE.** Both GUIs run today; `demo_gui/` was untouched. |
| 17 | `:36-37` — *"The golden/cached path is runtime-offline … Neither path initiates an internet download."* | **TRUE.** Offline env vars set before registry import in both `backend/services.py:19-20` and `demo_gui/app.py:12-13`. `frontend/README.md`'s stronger claim (*"does not request map tiles, fonts, imagery … from the internet"*) also holds — the maplibre style is an inline background-only style (`ImageryViewer.tsx:19`) and there is no Google Fonts import. |

### Undocumented divergences (not false, but unrecorded anywhere)

- **`backend/requirements.txt` omits `pillow`** while `backend/services.py:16` imports it. Following `backend/README.md` verbatim produces a venv that cannot import the app.
- **`frontend/components/status/CapabilityStatus.tsx:1-2` hardcodes the capability lists** as string arrays instead of calling `GET /api/capabilities`. The backend maintains a truthful, registry-backed availability snapshot (`capabilities.py:131-145`) and the UI ignores it. The moment a provider is registered for `grounding`, the "Available now" list will be wrong — exactly the failure mode the truthful-status design exists to prevent. `demo_gui/app.py:34-46` has the same hardcoding.
- **`npm run lint` is dead.** `next lint` was removed in Next 16; the script resolves as `next <dir>`.

---

## 4. Dataset inventory — Phase 5 planning input

**On disk in-repo: 13 GB. Plus 445 MiB outside the repo.**

### Present and usable

| dataset | path | size | files | content | tracked in git |
|---|---|---|---|---|---|
| **LoveDA** (extracted) | `data/raw/loveda/{Train,Val}/{Urban,Rural}/{images_png,masks_png}` | **6.0 GB** | **8,382 PNG** | 4,191 RGB tiles + 4,191 matched 7-class masks. Train/Urban 1,156 · Train/Rural 1,366 · Val/Urban 677 · Val/Rural 992 | no (`/data/raw/` ignored) |
| LoveDA (archives, redundant) | `data/raw/loveda/{Train,Val}.zip` | **6.0 GB** | 2 | Train.zip 4,021,669,263 B · Val.zip 2,425,958,254 B. Already extracted (`.Train.extracted` / `.Val.extracted` markers present). md5-pinned in `download_ladder_data.py:17-25` | no |
| **RSVQA-LR** (official Zenodo 6344334) | `data/raw/rsvqa_lr/` | **291 MB** | 785 | **772 `.tif`** Sentinel-2 256×256 (148 MB) + 11 JSON splits + the 95 MB `Images_LR.zip`. 10,004 active test questions. Every file size + MD5 pinned at `eval/suites/rsvqa.py:15-28` and matching on disk | no |
| **Resolution ladder** (derived) | `data/ladder/{0.3,1.0,2.0,5.0,10.0}/` | **383 MB** | **1,000 PNG** + `manifest.jsonl` (1,000 rows) | 200 LoveDA sources × 5 GSD rungs. 0.3 m 320 MB · 1.0 m 47 MB · 2.0 m 12 MB · 5.0 m 2.3 MB · 10.0 m 800 KB | **1 file only** (`0.3/loveda_Train_Rural_images_png_0_gsd0.3.png`, 1.8 MB — the golden scene) |
| **BigEarthNet.txt** annotations | `~/data/bigearthnet_txt/BigEarthNet.txt.parquet` — **outside the repo** | **445 MiB** (466,819,745 B) | 1 | 9,553,962 annotations over 464,044 co-registered S1+S2 patch names. md5 `68628955917d75a280e605bf33e9cea5` — matches `data/BEN_TXT_FACTS.md` exactly. CDLA-Permissive-1.0 | n/a |
| Ingested runtime scenes | `data/runtime/scenes/` | 1.8 MB | 2 PNG | leftovers from upload testing (one 1.8 MB, one 75 B) | no (ignored) |

### Absent

| dataset | needed for | status |
|---|---|---|
| **Any SAR pixels at all** | **Phase 5** | **ZERO bytes on disk.** No `.tif`/`.tiff`/`.npy`/`.nc`/`.SAFE` outside `data/raw/rsvqa_lr` (which is optical S2). `data/sar_gate/` contains **only 507 lines of Python and two Markdown files**. `data/sar_gate/rendered/` does not exist. `data/sar_gate/raw/` does not exist. `jobs.json` does not exist — **no HyP3 job was ever submitted.** |
| **Any paired optical–SAR data** | **Phase 5** | **ZERO.** The only route to it is BigEarthNet.txt's 464k co-registered S1+S2 pairs, whose **imagery is a separate 118 GB `.tar.zst` from Zenodo 10891137 with no per-patch fetch** (`eval/suites/RECON.md:96-99`). That download has not started. This is the single longest-lead item in the project. |
| **BigEarthNet imagery** | Phase 5 training | 118 GB, not downloaded, does not fit Kaggle's ~73 GB scratch without stratified subsampling to ~50k |
| **Any bi-temporal / change data** | **Phase 4** | **ZERO.** `cdvqa` is not on the HF Hub at all (`RECON.md:82-87`). No `optical_t2` pixels exist anywhere. `manifest.jsonl` has `"optical_t2": null` on all 1,000 rows. |
| **Any grounding / bbox data** | **Phase 3** | **ZERO.** No VRSBench (`Images_val.zip` 3.98 GB + `Images_train.zip` 8.36 GB, not downloaded), no DOTA (`data/raw/dota` absent). LoveDA masks are semantic segmentation, not instance boxes — usable as weak supervision only. |
| HRVQA / LRS-VQA | independent eval probes | not downloaded; both **CC-BY-NC-4.0** |

### What the SAR "capability" actually is

`data/sar_gate/annotation_template.md` — 417 lines, five hand-written analyst interpretations: Mumbai coastal, Maharashtra farmland, Western Ghats forest, Konkan coast, Flat inland plain. Each has Water / Urban-built-up / Vegetation / Terrain-artifact sections with physically-grounded reasoning (specular reflection off calm water, double-bounce in built-up areas, layover/foreshortening). `backend/services.py:410-453` parses these into structured `summaries` and serves them with `"human_validation": true`.

**This is real, careful domain knowledge — and it is prose, not a model, not a measurement, and not attached to a single pixel.** The demo's "SAR Validation" tab displays human-written text. That is defensible if labelled as such (and the API field `human_validation: true` does label it), but it is not a Phase 5 capability and it does not shorten the Phase 5 critical path by one day.

### Phase 5 feasibility verdict

**Phase 5 (optical–SAR fusion) currently has a dataset floor of zero.** Everything else — model code, capability vocabulary, planner rule (`optical_sar_cross_modal` fires correctly), `TileSample.sar` schema slot — is ready to receive data that does not exist. The two viable acquisition paths are (a) the 118 GB BigEarthNet Zenodo tarball, which needs a subsampling plan before a single byte is fetched, or (b) running the existing `data/sar_gate/order_scenes.py` → `process_scenes.py` pipeline against a real Earthdata account to produce a handful of hand-made S1 RTC scenes. Path (b) is already written and has never been run.

---

## 5. Dead code and stale artifacts

**Safe to delete outright:**

| path | size | why |
|---|---|---|
| `src/` | 44 KB | **Entirely dead.** No `.py` files, no files tracked by git — only `__pycache__/*.pyc` for a deleted `src/sih` package (config, data/loader, data/suites, eval/metrics, models/registry). All superseded. |
| `notebooks/` | 0 B | Empty since 2 Sep. Zero files, tracked or untracked. |
| `results/mock__resolution_proxy__20260913T175929Z.json` | 1.3 KB | **Untracked `accuracy: 1.0` artifact from the fake placeholder suite.** It sits in the directory `eval.py:318` calls *"the audit trail … commit this"*. Actively dangerous — delete before anyone commits it. |
| `data/raw/loveda/{Train,Val}.zip` | **6.0 GB** | Already extracted (markers present), md5s pinned in `download_ladder_data.py` so they are re-fetchable. Half of the repo's disk footprint. |
| `outputs/ladder_curve.png` | 63 KB | Untracked, Sep 7, referenced by nothing. `results/ladder_curve_qwen_stratified.png` is the README figure. |
| `data/runtime/scenes/*.png` | 1.8 MB | Two leftover uploads, one a 75-byte test fixture. |
| `.DS_Store` × 10+ | ~60 KB | Root, `models/`, `eval/`, `data/`, and every `data/raw/loveda/*`. Ignored but littered. |
| `.coverage` (root, untracked) | 53 KB | Coverage DB left in the tree. |
| `src/sih/**/__pycache__`, all `__pycache__` | — | Ignored; harmless except in `src/`, where it is the *only* content. |

**Not dead — leave alone:**
- `scripts/train.py` — a 32-line no-op today, but the declared Stage-1 entry point and consumed by `configs/example.yaml`.
- `outputs/results_ladder_qwen_full_v2.json` — tracked despite `/outputs/` being in `.gitignore`, and it is the **source of truth for the committed ladder rescore**. Deleting it orphans `results/RESCORE_NOTE.md` and the provenance string in the rescored report.
- `PROJECT_OVERVIEW.pdf` (240 KB, untracked, newly ignored) — personal reference.

**Defects worth a ticket, found in passing:**

1. **Test-isolation leak — the test suite writes to the production audit chain.** `demo_gui/test_failure_paths.py` creates a `TemporaryDirectory` (`:46`) but, unlike `demo_gui/test_golden_path.py:29-33` and `backend/test_api.py:23-27`, it **never patches `orchestrator.trace.TRACE_PATH`**. Its four `self.ask()` calls (`:70,82,98,107`) drive the real Streamlit app through `cached_response()` → `append_record()` → the repo-root `trace.jsonl`. `demo_gui/test_offline.py:37` does the same.
   **Measured directly:** `pytest demo_gui/test_failure_paths.py` (6 passed) appended **3 real `cached_result` records** to the production `trace.jsonl` — 18 → 21 lines. (The file was restored byte-for-byte afterwards and the chain re-verifies at 18 records; `git status` is unchanged from the session start.) Three of the four `self.ask()` calls reach the cached path; the fourth is the deliberate no-match error case.
   The chain still verifies — the records are genuine — but the production evidence file is polluted by every test run, which undermines the "hash-chained audit trail" story the whole demo rests on. One-line fix: the same `patch.object(trace, "TRACE_PATH", ...)` the sibling test file already uses.
   *Separately noted:* a Streamlit session is running on port **8501** outside this audit (`pgrep`), so live demo clicks are also appending to the same file. Between tests and a live demo, `trace.jsonl` is not a controlled artifact today.
2. `backend/requirements.txt` is missing `pillow`.
3. `frontend` `npm run lint` is broken (Next 16 removed `next lint`).
4. `frontend/components/status/CapabilityStatus.tsx` hardcodes capability availability instead of calling `GET /api/capabilities`.
5. `confidence` is produced by both models and silently dropped before it reaches any consumer (`backend/schemas.py:37-43`).

---

## Appendix — commands that would resolve the remaining UNKNOWNs

| unknown | command |
|---|---|
| Does `QwenVLModel._generate_answer()` actually work? | `python kaggle/run_rsvqa_baseline.py --repo-url <url>` on a Kaggle T4. Nothing local can answer this. |
| Coverage of `models/`, `data/`, `eval/ladder.py`, `eval/plot_ladder.py` | `COVERAGE_FILE=/tmp/cov backend/.venv/bin/python -m pytest --cov=models --cov=data --cov=eval --cov-report=term-missing` (statically: `eval/ladder.py`, `eval/plot_ladder.py`, `data/dataset.py`, `data/download_ladder_data.py`, `data/sar_gate/*.py`, `scripts/*.py`, `kaggle/run_rsvqa_baseline.py`, `models/mock/model.py` are imported by **no** test module) |
| Does the frontend build under Next 16? | `cd frontend && npm run build` (not run — it writes `.next/`; `npx tsc --noEmit` passes, and `.next/` from a prior successful build exists on disk) |
| Is the LoveDA extraction complete/uncorrupted? | `python data/download_ladder_data.py` — it re-verifies md5s idempotently |
| Can the SAR gate actually produce a render? | configure `~/.netrc` for `urs.earthdata.nasa.gov`, then `python data/sar_gate/order_scenes.py` → wait 20–90 min/scene → `python data/sar_gate/process_scenes.py`. Never attempted (`jobs.json` absent). |
