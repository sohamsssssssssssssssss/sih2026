# SATQUERY AI — MASTER IMPLEMENTATION PLAN v5.0

> **Superseded by [MASTER_PLAN.md](MASTER_PLAN.md) (16 Nov 2026).** Kept for history.

**Supersedes v4.0.** Baseline re-verified against the repo on **Sun 13 Sep 2026**.
Finale window 8–15 Dec 2026. One finish line.

v4.0's strategy is kept intact: the floor only goes up, cutting a technology is never
cutting a requirement, everything ships. What changed is the *baseline* — v4.0 described
a repo that no longer exists, in both directions. Every claim below was verified by
running the code, not by reading documentation.

---

## 1. WHERE WE ACTUALLY ARE (verified 13 Sep)

**The real floor: one model, one capability, one cached answer.**

| Layer | Verified state |
|---|---|
| Orchestration | **Strong.** 2,110 lines, 93 tests, 96% coverage. Planner, capability routing, multi-step execution plans, hash-chained ledger. All five planner rules verified live. |
| Backend API | **Working.** 11 endpoints, 254 tests passing, FastAPI, CORS locked to localhost. |
| Models | **153 source lines.** `mock` (string formatter) + `qwen_vl` (106 lines, real and typed, **never executed on this machine**, patched out by every test). |
| Live inference | **Never run here.** All 18 ledger records are `execution_mode: cached_result`. |
| Grounding / Change-VQA / Fusion | **Do not exist.** Three strings in `orchestrator/capabilities.py:27-29`. Zero code. |
| Frontend | 613 lines, **zero tests**, `npm run lint` broken, less capable than the Streamlit fallback. |
| Infra | **No CI, no Docker, no SQLite, no job engine.** |
| GPU | **Unrented.** Hard deadline **1 Nov** or Phase 9's GPU-OOM fault goes untested. |

### Phase 0 exit criteria — status at close

| Criterion | Status |
|---|---|
| Dated audit doc | ✅ **Now delivered** — Annex A |
| One-page threat model | ✅ **Now delivered** — Annex B |
| GPU provider selected + one real inference call | ❌ **NOT DONE — the critical path** |
| Golden path working, screen-recorded | ⚠️ Working (verified); recording outstanding |
| API v1 schema frozen | ✅ `backend/schemas.py` |
| Infra architecture locked against repo reality | ✅ Annex C — all v4.0 cuts confirmed correct |

---

## 2. RE-BASELINE — v4.0 IS WRONG IN BOTH DIRECTIONS

PRs #7–#12 merged 9 Sep, during Phase 0, and shipped work v4.0 schedules for later.

**Ahead of plan (~3 weeks of credit):**

| v4.0 schedules | Reality |
|---|---|
| Phase 1 builds controlled upload | Built (`ingest_scene`). Held under every hostile probe. Missing only SHA-256. |
| Phase 1 moves inference off the event loop | Built — `router.py:14` `ThreadPoolExecutor(max_workers=1)` |
| Phase 2 builds the hash-chain ledger | Built. Survives restart, fails closed on tamper. |
| Phase 6 builds orchestration | **~85% done.** ~6 h remains, not 5 days. |
| Phase 9 builds hostile-input rejection | All six categories already reject (live 400 Mpx bomb → 422) |
| Phase 9 offline bundle | Frontend already offline-clean — no `next/font`, no CDN, maplibre fetches zero tiles |

**Behind plan (assumed present, absent):** SQLite job engine · Docker/Compose · **CI** ·
SHA-256 checksums · every Phase 3/4/5 dataset · any calibration code.

**Net: reallocate, do not rebuild.** ~24 h freed from Phase 6 → Phase 9 fault injection
and an offline rehearsal one week early.

---

## 3. ⚠️ FOUR ZERO-FABRICATION VIOLATIONS — FIX THIS WEEK

Zero fabrication is the project's absolute claim and its entire pitch. All four are live
today. Total cost ≈ 1 day.

**V1 — Fabricated coordinates (30 min).** `ImageryViewer.tsx` hardcodes
`center: [72.88, 19.08]`, captions `19.08° N / 72.88° E`, and georeferences the scene to a
Mumbai bounding box. The scene is **LoveDA — a Chinese dataset**. The backend deliberately
refuses to claim a location (`schemas.py:28 location: None = None`); the frontend overrode
that refusal with invented data. An ISRO judge seeing a map pinned to Mumbai showing
Chinese farmland is the worst available outcome.

**V2 — `confidence: 1.0` (15 lines).** `models/qwen_vl/model.py:104` ships a hardcoded
`1.0` in the API payload. The code comment is honest and the UI says "calibration
pending", but the *payload* states a false number. Replace with a real token-logprob at
zero extra inference cost (Annex D §1).

**V3 — False README scorer claim (10 min).** `README.md:15` documents "exact
case-insensitive stripped answer matching". `eval/eval.py:45-72` actually does exact
match **plus** word-boundary substring search **plus** number-word→digit normalisation.
Every committed number came from the loose scorer while the README advertises a strict
one. Under judge questioning this is the most dangerous line in the repo.

**V4 — The fake 100% (1 h).** `--suite resolution_proxy` returns one placeholder whose
`expected_answer` is byte-identical to `MockModel`'s output → `accuracy: 1.0`, `n=1`. Worse,
`eval.py` prints "← commit this" after every run. Delete the untracked
`results/mock__resolution_proxy__*.json` and gate the suggestion on a real sample count.

---

## 4. ⚠️ THE SCIENTIFIC BLOCKER — FIX IN W2, ONE DAY

**`eval/eval.py` can score exactly one thing: string equality.**

No IoU. No metric dispatch. No per-condition stratification. No `run_id` or
`dataset_version`. Phases 3, 4 and 5 all produce non-string outputs.

It is dangerous *because it looks mature* — degeneracy guards, sample-size floors,
rescore provenance notes, `git_sha()`. Feed it a bounding box and it returns `False`,
completes without error, and writes JSON that reads as a measurement.

**Nothing in Phases 3–8 is trustworthy until this is fixed. It is the first task of W2.**

Same day, two more: the ladder source cap is hardcoded at 200 (MDE 13 pts — too coarse to
detect any Phase 8 effect; raise to 1,000), and `models/qwen_vl/model.py:34` hard-raises
on `not torch.cuda.is_available()` while **MPS is available and built** here (torch 2.14).
Thirty minutes unlocks all local iteration. *MPS numbers are for iteration only — every
reportable baseline still comes from the rented GPU, or comparability with committed
results is broken.*

---

## 5. TRACK REBALANCE — THE STRUCTURAL FIX

Frontend scope alone is **~182 h**. Track B also owns infra, testing and hardening
(~150–200 h). Against ~286 h of realistic capacity that is **~1.5× over** on the true
11-week build window. This is not fixable by better scheduling.

**Move infrastructure (Phase 1 job engine, Docker, CI) to Track A**, which has ~24 h
freed from Phase 6 and no capability work until Phase 3 (6 Oct). Raise it in W2, while it
is still a scheduling conversation rather than an emergency.

**Ownership risk.** Track A owns Phases 2, 3, 4, 5, 7 *and* 8 — every capability. One bad
week has no slack behind it. Mitigations: Track B owns the Phase 4 evaluation harness; a
written bus-factor note per phase; **every phase exit screen-recorded** (2 min — a
regression record and finale B-roll).

**Buffer (v4.0 had none).** Bank **3 days each from Phases 3, 4 and 6** into a floating
9-day buffer. Drawn only at a checkpoint, only by joint agreement, never to add scope.
Unused buffer rolls to Phase 9.

---

## 6. CORRECTED CALENDAR

v4.0's day labels are shifted one day: 8 Sep 2026 is a **Tuesday**; "code freeze Fri 5 Dec"
is a **Saturday**. Re-anchored to true Mondays:

| Week | Dates | Phase |
|---|---|---|
| W1 | Mon 07 – Sun 13 Sep | Phase 0 — **closes today** |
| W2 | Mon 14 – Sun 20 Sep | Week 0 fixes · CI · eval harness · **GPU rental** · **SAR spike Sat 19** |
| W3 | Mon 21 – Sun 27 Sep | Phase 1 — job engine, SHA-256, upload UI |
| W4 | Mon 28 Sep – Sun 04 Oct | Phase 2 — VQA hardening, provenance · **CP1** |
| W5–W6 | Mon 05 – Sun 18 Oct | Phase 3 — Grounding · **Oct dry run Sat 24** · **CP2** |
| W7–W8 | Mon 19 Oct – Sun 01 Nov | Phase 4 — Change-VQA · **GPU deadline 1 Nov** · **CP3** |
| W9–W10 | Mon 02 – Sun 15 Nov | Phase 5 — Fusion · **CP4 hard gate** |
| W11 | Mon 16 – Sun 22 Nov | Phase 6 — Orchestration (~6 h) + banked buffer · **CP5** |
| W12 | Mon 23 – Sun 29 Nov | Phase 7 — Ladder + Indian proxy |
| W13 | Mon 30 Nov – Sun 06 Dec | Phase 8 (Track A) ∥ Phase 9 (Track B) |
| — | Mon 07 Dec → finale | Phase 10 — freeze at **T−2**, rehearsal |

**T−2 rule.** Build until T−2 days before the confirmed finale. After T−2: bug fixes,
deployment fixes, regression fixes, verification, rehearsal, emergency recovery only.

---

## 7. WEEK 0 — Mon 14 – Sun 20 Sep (new phase, non-negotiable)

Everything here is a prerequisite for something downstream. Nothing here is optional.

| # | Task | Owner | Cost |
|---|---|---|---|
| 1 | **Rent the GPU, run one real inference call** | A | 3 h |
| 2 | Fix V1–V4 (§3) | A+B | 1 d |
| 3 | **Metric dispatch in `eval.py`** + `run_id`/`dataset_version` (§4) | A | 1 d |
| 4 | **CI** — GitHub Actions, pytest + coverage + hostile-input | A | 2 h |
| 5 | MPS unlock, ladder cap → 1,000 | A | 1 h |
| 6 | Fix `npm run lint`, add Prettier, drop 2 dead deps | B | 2 h |
| 7 | Playwright + `npx playwright install chromium` **while there is internet** | B | 3 h |
| 8 | **SAR spike (Sat 19)** — STAC metadata only, no downloads | A | 1 d |
| 9 | Write the **5-minute demo script** | A+B | 2 h |
| 10 | Pin `requirements.txt`; fix `.python-version` (says 3.11, venv is 3.13 — will break CI) | A | 30 m |
| 11 | Patch `TRACE_PATH` in `demo_gui/test_failure_paths.py` — it writes 3 records to the production ledger on every test run | B | 15 m |

**CI measured:** the 169 gating tests run in **2.64 s** on a six-package venv (no
torch/transformers/streamlit) because `QwenVLModel` imports torch lazily. Two hours total.

**The demo script comes first.** v4.0 is capability-ordered and defers the script to
Phase 10 — which is exactly how the project reached today with `/api/plan` and
`/api/capabilities`, its best work, invisible to any judge. Revise it at every checkpoint.
**If a capability has no line in the script, it gets no UI time.**

---

## 8. PHASES 1–10 (re-baselined)

### Phase 1 — Durable jobs + artifact integrity · W3
**Reduced scope** — upload and off-event-loop execution already exist.
SQLite job table + **one daemon thread** wrapping the existing `analyze_scene()` unchanged.
`router.py`'s single-worker executor already serialises inference; the job engine adds zero
new concurrency. Idempotency via `UNIQUE(idempotency_key)`; backpressure cap 16 under one
`BEGIN IMMEDIATE` → 429 when full. Restart recovery via per-process `worker_epoch`: any
non-current-epoch `RUNNING` row is orphaned by definition. Retries bounded and selective —
`model_unavailable` and `trace_persistence_failed` never retry. Cancellation is honest:
queued jobs truly cancel; a `RUNNING` GPU call cannot be interrupted, so the answer is
discarded and we **say so**.
**`/api/analyze` stays untouched as the fallback floor.** SHA-256 on artifacts (20 MiB =
7.2 ms). **Leave `trace.jsonl` alone** — add one `job.trace_record_hash` column; migrating
costs ~8 h to recreate properties it already has, and a flat hash-chained file is *more*
auditable to a judge than DB rows.
*Verified, not asserted:* SIGKILL mid-transaction → committed state survived, uncommitted
rolled back, `integrity_check ok`. 8 threads × 50 rows → 50 unique claims. 1600 commits in
0.22 s, zero `SQLITE_BUSY`.
**Frontend:** Phase 1's honest job is **reaching parity with Streamlit** — free-text
questions, working upload, sensor selection. Streamlit has all three; Next.js has none.
**Exit:** 10+ scenes upload→analyze; hostile suite green in CI; job state survives worker kill; golden path unmodified.

### Phase 2 — VQA hardening + provenance · W4 · CP1
Ledger already done. Remaining: typed Qwen wrapper `validate()/execute()/health()/describe()`,
strict golden lookup (exact scene+question+GSD), **and fix `/api/traces/verify` to re-read
from disk** — it currently verifies memory only, so on-disk tampering returns
`{"verified": true}` until restart, and the Executions screen shows 6 of 18 real records.
**Exit:** 20 real VQA queries with typed provenance; zero fabricated confidence in manual review; ledger survives kill-restart; near-miss returns explicit no-answer.

### Phase 3 — Grounding · W5–W6 · CP2
**Floor: LoveDA masks already on disk** (6.0 GB, 8,382 PNGs) → connected components →
boxes/polygons. Zero download, zero acquisition risk. **Upgrade:** DIOR-RSVG (natural
language) — Google-Drive hosted, and this repo *already failed on Drive for DOTA*, so treat
it as needing a human with a browser. Qwen2.5-VL grounds natively (absolute pixel coords,
not 0–1000); GroundingDINO as tool-fallback. Metric **Acc@0.5 IoU, n=500, Wilson CI**, plus
a `mean_box_area_fraction` guard — the grounding analogue of the always-yes guard.
Ships **confidence Part A** (token-logprob, §3 V2).
**Exit:** 15+ scenes with typed evidence; reported number with run_id + git SHA; explicit no-evidence state.

### Phase 4 — Bi-temporal Change-VQA · W7–W8 · CP3
LEVIR-CD + LEVIR-CC (2.5 GB, academic-only). Gold derived from change masks exactly as
`ladder.py:label_answers()` already does, so `summarise()`, the degeneracy guard and
`open_accuracy` work unchanged. **Three conditions × 400: pair / single-frame control /
identical-pair null** — the third is a hallucination probe and is the strongest anti-fabrication
evidence in the project. *Correction: `RECON.md` calls CDVQA a "track blocker, not on HF Hub" —
it is on GitHub (`YZHJessica/CDVQA`). Not blocked.*
**Track B owns this evaluation harness** (bus-factor mitigation).
**Exit:** 10+ real pairs; single-frame control documented; null condition reported.

### Phase 5 — Optical-SAR fusion · W9–W10 · CP4 HARD GATE
**Highest risk in the project, and worse than v4.0 assessed.** Verified: `data/sar_gate/`
holds four files — two scripts, two markdown docs. **No `rendered/`, no `jobs.json`** — and
`order_scenes.py` writes `jobs.json` on submission, so its absence proves **no HyP3 job was
ever submitted**. Zero SAR rasters anywhere on disk. RTC processing is 20–90 min per scene
and needs Earthdata credentials nobody has verified. **Both halves are at zero.**
Plan: self-acquired S1+S2 over the 5 committed Indian AOIs; SEN12MS fallback;
BigEarthNet-MM rejected (118 GB monolithic Zenodo tarball — longest-lead item in the
project). **Decision-level (late) fusion, not an architecture.** Headline =
`fused − max(single)`.
**CP4 is the one sanctioned cut**, gated strictly on data availability, never on effort.
If cut: `NOT DEPLOYED`, capability screen updated, **zero partial UI** — and this costs
nothing, because `/api/capabilities` already returns `available: false` for `optical_sar`.
**Exit:** 5+ paired scenes with a reported metric, **OR** a clean NOT DEPLOYED state with zero fake output.

### Phase 6 — Orchestration · W11 · CP5
**~85% built.** Allowlist, all-or-nothing execution and trace-forgery resistance are done
and tested. Remaining **~6 h**: (1) `route()` is **unbounded when `timeout_seconds is None`**
— the safe path is opt-in; make 120 s the default (30 min, highest value in the phase);
(2) traces carry no `job_id`/`request_id`; (3) `executor.py` at 83%; (4) one HTTP-layer
capability-unavailable test.
**⚠️ Resolve in W2: "tool invocation" is undefined.** If *tool* means capability provider,
this phase is done. If it means external tools, none exist and it is new feature scope in
the last three weeks. **Settle the definition before W11.**
Freed time → Phase 9 + banked buffer.

### Phase 7 — Ladder + Indian/high-res proxy · W12
**The headline differentiator — protect this time above all else.**
Regenerate the ladder at **1,000 sources**. Indian proxy = self-acquired Sentinel-2 over the
committed AOIs + ESA WorldCover gold, plus a **GSD↔ISRO-sensor table** (0.3 m ≈ Cartosat-3,
5 m ≈ LISS-IV 5.8 m, 10 m ≈ S2).
**State plainly that LoveDA is Chinese — a resolution proxy, not a geography proxy.** That
honesty is also what fixes V1 at the root. Every number carries run_id + git SHA +
dataset_version. Selects the Phase 8 target slice.
Ships **confidence Part B** (ECE + temperature scaling on RSVQA-LR val, already on disk) —
**demoted to SHOULD SHIP**.

### Phase 8 — RS fine-tuning · W13, Track A
**Mandatory, non-negotiable** (SIH requirement) regardless of which way the numbers move.
**Zero acquisition risk:** the RSVQA-LR train split is already downloaded and
integrity-checked (291 MB, md5-pinned) — and at 10 m Sentinel-2 it is *exactly* the GSD of
the ladder's collapsed rungs. LoRA r=16, vision tower frozen.
**October dry run (Sat 24 Oct, ~6 h)** — 20 samples, 50 steps, **the real December config**.
Nine assertions: peft/transformers compat, LoRA target modules resolve by name, vision
tower frozen, `check_gpu()` gates on real hardware, peak VRAM, adapter reload *through the
registry*, `eval.py` end-to-end. Proves nothing about accuracy — **name the output `smoke_`,
never `results_`.** Turns December into "swap in the real slice" instead of "debug CUDA".
Checkpoint goes live **only if it wins**; otherwise the prior stays and the experiment is
still reported honestly. Floor unaffected either way.

### Phase 9 — Hardening consolidation · W13, Track B
Fault matrix on the **real** box: kill worker mid-job · kill DB · corrupt artifact · GPU OOM
· timeout · disk pressure — each with a verified non-silent recovery.
**Add a process-liveness check to the System Status screen.** During this session the
Streamlit fallback exited cleanly while API and frontend stayed up, and *nothing surfaced
it* — the floor died silently. That is the exact failure this phase exists to catch.
Load test to a realistic judging-panel ceiling (**5–20 concurrent, not 1000**). Offline
rehearsal with WiFi off and `--reload` dropped. Dependency scan. Back up `results/`,
`trace.jsonl` and SAR artifacts; screenshot the ledger head hash.
**⚠️ GPU OOM cannot be tested without the rented box — hard deadline 1 Nov, or declare it untested.**

### Phase 10 — Freeze, ablations, rehearsal · W13 → finale
Full E2E across shipped capabilities. Ablation table from Phase 7/8 numbers. Limitations
section matching the capability screen **exactly**. Hostile demo rehearsal: odd questions,
bad uploads, network drop mid-session, "what can't it do". **Freeze at T−2.** Minimum 3 full
run-throughs. Written fallback plan for the 3 likeliest venue failures.
**Exit:** 3 rehearsals, zero crashes, zero fabricated output, capability screen matches deployed reality exactly.

---

## 9. RISK REGISTER

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| R1 | **GPU unrented** — gates Phases 2,3,4,5,7,8 + F4 fault test | 🔴 | W2 task #1. Hard stop 1 Nov. |
| R2 | **`eval.py` scores only strings** — Phases 3/4/5 unmeasurable | 🔴 | W2, one day, before any dependency |
| R3 | **Phase 5 at absolute zero** — no data, no job ever submitted | 🔴 | Sept spike; CP4 sanctioned cut |
| R4 | Four live zero-fabrication violations | 🔴 | §3, one day, this week |
| R5 | Track B ~1.5× over capacity | 🟠 | Move infra to Track A |
| R6 | Track A owns every capability phase | 🟠 | Track B owns Phase 4 harness; bus-factor notes |
| R7 | "Tool invocation" undefined in Phase 6 | 🟠 | Settle in W2 |
| R8 | Live Qwen path never executed here | 🟠 | MPS unlock + GPU rental |
| R9 | `_slug()` traversal latent (HIGH if route becomes `{scene:path}`) | 🟡 | Allowlist the key — one line |
| R10 | Error handlers leak absolute paths | 🟡 | Sanitise `resolution.py:12`, `sar.py:14` |
| R11 | Ledger tamper-evident, not tamper-proof | 🟡 | Document honestly; do not overclaim to judges |
| R12 | No inter-phase buffer | 🟡 | 9-day banked buffer |

---

## 10. ANNEXES

| | Document | Contents |
|---|---|---|
| A | `ANNEX-A-phase0-audit.md` | Capability inventory, per-directory audit, false README claims, dataset sizes |
| B | `ANNEX-B-threat-model.md` | One-page threat model, ranked findings, Phase 9 checklist |
| C | `ANNEX-C-infrastructure.md` | Job engine DDL + code, CI YAML, Docker, fault matrix |
| D | `ANNEX-D-science.md` | Per-phase datasets, harnesses, Sept spike, Oct dry run, confidence plan |
| E | `ANNEX-E-frontend.md` | Workload maths, priority order, API client spec, per-phase UI, testing |

**Scans clean as of 13 Sep:** zero secrets in tracked files, across all 79 commits, or in
the `.next` bundle. `npm audit` 0 vulnerabilities. `pip-audit` 8 findings — all install/test-time
noise except `accelerate` (no fix version, needs local checkpoint write access).

---

*v5.0 — 13 Sep 2026. Every claim verified against the running system.*
