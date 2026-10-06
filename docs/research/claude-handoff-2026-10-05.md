# Handoff: SatQuery state after the 5 Oct 2026 Claude session

Read this first if you are picking up SatQuery (SIH26167, "Ask your satellite
data anything") without context. It covers the state of the code, what was
verified and how, what is still open, and the plan the team agreed to.

`docs/research/next-session-checklist.md` and the earlier phase handoffs
predate this session. Where they disagree with this file, this file is newer.

---

## 1. Where things stand

| Item | State |
|---|---|
| `main` | `8ecbed6`, the merge of PR #17. Contains all work below. |
| PR #16 | Merged. Phase 0 fixes plus Phase 1 backend and frontend (13 commits). |
| PR #17 | Merged. Docker and deploy setup, plus the Grounding DINO text-encoder readiness fix. |
| Working branch | `claude/relaxed-clarke-7c6e7y`, restarted from `main` and carrying only this file. Both PRs from it are merged, so new work is a new PR. |
| CI | Green on the PR #17 head `2e3c9ac` (backend py3.11 and py3.13, frontend build). |
| Light test suite | `320 passed, 1 skipped` (`pytest backend orchestrator models`) |
| Frontend | `tsc` clean, 28 vitest tests, 15 node tests, `next build` passes |
| GPU | **Never available in this session.** Live Qwen and Grounding DINO inference is unverified in this codebase state. |

---

## 2. What this session changed

### Phase 0: stop the bleeding (PR #16)

- **CI:** it was red because `models/qwen_vl/test_stage1.py` imported torch in the light CI environment. That test is now skipped with `pytest.importorskip`.
- **Trace ledger (`orchestrator/trace.py`):**
  - Every append is `fsync`ed.
  - A torn final line (no trailing newline) is quarantined to `trace.jsonl.torn-<UTC>` and the file is truncated, so a crash no longer makes every `/api/analyze` return 503.
  - Malformed newline-terminated lines and hash mismatches still fail closed.
  - `verify_chain()` with no argument now re-reads the file on disk. Before, it checked only the in-memory copy, so tampering went undetected.
- **`/api/health`:** reports `ready`, `degraded` or `unavailable` (503), with `checks.trace` and `checks.capabilities`. Before, it always said "ready". The frontend status card and sidebar now use it.
- **Logging:** module loggers in `backend/services.py` and `backend/routes/analyze.py`. Every 5xx logs a traceback. User text is truncated by `log_excerpt()`, and uploaded bytes are never logged.
- **Paths:** no absolute server paths in responses or traces. `models/paths.py:public_path()` returns repo-relative paths, or just the file name for anything outside the repo.
- **Notices and README:**
  - The cached-replay notice no longer claims live inference failed.
  - The README's analyze example now sends `execution_mode: "cached_result"`.
  - The rasterio advice is fixed.
  - The README no longer says to delete `trace.jsonl` to recover.

### Phase 1: something a judge can use (PR #16 and PR #17)

- **Killable inference (`orchestrator/worker.py`):**
  - Models with `isolated = True` (Qwen, Grounding DINO) run in a long-lived `spawn` worker process that keeps weights warm.
  - A timeout terminates the process. The next request respawns it.
  - A crash raises `WorkerCrashed`, which the backend maps to 503.
  - Errors come back as `RemoteInferenceError`, with the original message kept.
  - Non-isolated models (change, optical-SAR) run in a per-call thread.
  - Before this, a single `ThreadPoolExecutor(max_workers=1)` let one hung job block every request.
- **Uploaded scenes:** `GET /api/scenes` now includes `uploads`, built from runtime manifests: newest first, capped at 200, bad manifests skipped. `GET /api/scenes/uploads/{scene_id}` is new, and new manifests store `uploaded_at`.
- **Frontend (`components/workspace/Workspace.tsx`):**
  - Real upload: drag and drop, metadata fields (modality, acquisition time with timezone, sensor, polarization), and the server's `detail` shown as-is.
  - Scene picker covering pinned, curated and uploaded scenes, a free-text question, and an optional second scene for pair questions.
  - 422 and 503 detail objects rendered readably, an evidence list, and `normalized_xyxy` overlays on the image.
  - A "Show committed result (cached replay)" button that appears only after a live 503 on the pinned golden scene and question. It never retries automatically.
- **Deploy:**
  - `Dockerfile.backend` with `cpu` and `gpu` targets, `frontend/Dockerfile`, `docker-compose.yml` plus `docker-compose.gpu.yml`, and a Caddy proxy with one URL and automatic HTTPS.
  - `deploy/provision_models.py`, `deploy/smoke.sh`, and `make docker-cpu|docker-gpu|deploy-smoke`.
  - `SATQUERY_CORS_ORIGINS` for exact extra origins.
  - Full guide in `docs/deploy.md`.
- **Grounding readiness:** Grounding DINO's packaged config loads `bert-base-uncased` via `from_pretrained`. Readiness now checks the HF cache for it, so health no longer reports grounding as available when the first query would fail.

### Verified end to end (not just unit tests)

- A script against the real FastAPI app with real GeoTIFFs covered all of these:
  - upload
  - change detection
  - a torn-write recovery returning 200
  - health reporting `degraded` on CPU
  - on-disk tampering detected by verify
- A headless Chromium run against `next start` plus `uvicorn` covered:
  - a live 503, then the replay button, then a cached "Yes" with the `VERIFIED CACHED RESULT` badge;
  - two GeoTIFFs in the wrong date order, which gave a readable 422 (`acquisition_order_invalid`);
  - two GeoTIFFs in the correct order, which gave a live change answer with 4 evidence items and repo-relative paths.
- The CPU Docker stack was built and run behind Caddy, and `deploy/smoke.sh cpu` passed. The GPU image failed at export when the sandbox disk filled, so the imports inside it are unverified.

---

## 3. Honest assessment (why the work is ordered this way)

An earlier review in this session rated the project about **3/10 as a real-world product**, though the engineering discipline is good:

- Without a GPU, the only product is a cached "Yes" plus a pixel-difference counter that **ignores the question text**. Change answers are identical whatever is asked.
- Frozen Qwen2.5-VL-3B on RSVQA-LR scores 0.165 open accuracy. It collapses to the yes-prior at 5 m and 10 m, which is Sentinel-2's resolution.
- The planner is keyword regex. "Find all aircraft", "Show the ships" and "Detect oil spills in this radar image" do not route to grounding. "Where has the forest decreased?" yields a multi-step plan that is **not executable**.
- The docs-to-code ratio was very high (7k+ lines of Markdown, a master plan at v5).

After Phases 0 and 1 the estimate is about 5/10, and the remaining phases are what move it.

**Strategic direction agreed with the user:** own **Indian monsoon flood mapping**. Optical imagery is clouded out during the monsoon and SAR sees through cloud, so it is the niche where the existing optical-SAR, CDSE and `asf_search`/`hyp3_sdk` plumbing actually matters. The target demo:

> "Which villages near Patna flooded on 20 Aug that weren't flooded on 1 Aug?"
>
> Sentinel-1 is fetched automatically, water is segmented and differenced, and the answer comes back as a village list, hectares and a GeoJSON map, with a calibrated confidence, one deliberate abstention, and a verified audit trail.

---

## 4. What to do next, in order

### Immediately (needs the team or a GPU)

1. **First real GPU run.** On a VM with a T4, L4 or A10 (24 GB preferred), the R570+ driver and at least 80 GB of disk, follow `docs/deploy.md`: run `make docker-gpu`, then `deploy/provision_models.py`, then `make deploy-smoke`. This is the first time live Qwen and Grounding DINO will run on this codebase.
2. **Measure, don't assume:** actual VRAM use (the 9–11 GB figure is an estimate), cold-load time inside the worker, which counts against the 120 s `MODEL_EXECUTION_TIMEOUT_SECONDS`, and whether killing the worker frees VRAM.

### Phase 2: own the flood problem (most important)

- [ ] Pick 3–5 past Indian flood events (Bihar, Assam, Kerala) with before and after dates.
- [ ] Fetch Sentinel-1 automatically from CDSE or ASF for an AOI and date range.
- [ ] SAR preprocessing: calibration, terrain correction, and co-registration across dates.
- [ ] Train or adopt SAR water segmentation (Sen1Floods11). It must beat Otsu thresholding on held-out IoU.
- [ ] A "flood change" capability that differences two water masks into newly flooded polygons.
- [ ] Tiling and stitching for full scenes (today's change model handles small rasters).
- [ ] GeoJSON output in real coordinates with hectares, which opens in QGIS.
- [ ] Village-boundary overlay (Census or LGD), so the answer is "which villages".
- [ ] Optical cloud detection: "optical is cloudy, using SAR".

### Phase 3: understand questions

- [ ] A small LLM extracts `{place, dates, intent}`, with the keyword planner kept as fallback. Also geocode the place name to an AOI.
- [ ] A test set of 100 real-phrasing questions with routing accuracy of at least 90%.
- [ ] Change answers must use the question, or say explicitly what they cannot answer.

### Phase 4: know when it doesn't know (the research contribution)

- [ ] Calibrate confidence for VQA and the flood model, targeting ECE < 0.05. Abstain below a threshold and say why.
- [ ] Risk-coverage curve, for example "at 70% coverage, 90% accuracy".
- [ ] Run the existing QLoRA RSVQA pipeline and beat the 0.165 open-accuracy baseline on the held-out split.

### Phase 5–8 (later)

- **Phase 5:** benchmark against a classic baseline and one remote-sensing VLM, and release a 200–500 question Indian flood and change evaluation set with a geo-blocked split (reuse `data/bigearthnet_split.py`).
- **Phase 6:**
  - an HMAC-keyed trace (the hash chain is unkeyed today, so anyone with write access can re-forge it);
  - append-only storage with pagination for `/api/traces`;
  - API keys and rate limits;
  - multi-process-safe tracing (today it must be `--workers 1`).
- **Phase 7:** 2–3 real outside users run it on a past flood and compare against official reports.
- **Phase 8:** code freeze one week before the finale (8–15 Dec 2026), a rehearsed demo, an offline recorded fallback, and a one-page results sheet.

**What to cut:** cinematic-intro work, new plan documents (freeze `docs/plan/`), expanding the Streamlit UI, BigEarthNet captioning unless it serves floods, and more planner keyword rules.

### Smaller open items

- `TRACE_PATH` is hard-coded to the repo root. Docker works around it with a symlink, but an env var would be cleaner.
- `configs/model_artifacts.json` has `null` revisions and SHA-256s, and `bert-base-uncased` is not in it.
- Old records in an existing `trace.jsonl` may still contain absolute paths. They are hash-chained, so rotate the file rather than edit it.
- `IntegrityBadge` always shows "Hash chained". It could call `POST /api/traces/verify`.
- Uploads show a pending state, not byte progress.

---

## 5. Working on this repo: setup and gotchas

```bash
# Python (backend + light tests); there is no torch in the light env, matching CI
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install -r backend/requirements.txt
PYTHONPATH=. /tmp/venv/bin/python -m pytest backend orchestrator models -q   # expect 320 passed, 1 skipped

# Frontend
cd frontend && npm ci && npx tsc --noEmit && npx vitest run && node --test tests/workspace.test.mjs && npm run build
git checkout frontend/next-env.d.ts   # `next build` rewrites it; never commit that change

# Run locally
PYTHONPATH=. /tmp/venv/bin/uvicorn backend.main:app --port 8000
cd frontend && npm run dev      # http://localhost:3000/workspace
```

- **`frontend/AGENTS.md`:** this Next.js version (16.x) has breaking changes. Read `frontend/node_modules/next/dist/docs/` before using Next APIs.
- **Tests must patch `orchestrator.trace.TRACE_PATH`** and the runtime dirs (`backend.services.INGESTED_SCENE_DIR`, `INGESTED_RASTER_DIR`, `SCENE_MANIFEST_DIR`) to temp dirs. CI fails if the real `trace.jsonl` is modified.
- **Worker tests:** spawned processes don't see in-test monkeypatches. Use `orchestrator/_test_fakes.py` as the worker factory, or patch `orchestrator.router.get` with non-isolated stand-ins.
- **Safety invariants are deliberate. Do not weaken them:**
  - no silent fallback to cached answers;
  - no fabricated confidence, boxes, GSD or location;
  - unknown metadata stays `null`;
  - CI greps the frontend for hard-coded coordinates.
- **The planner routes on keywords.** Change detection needs two uploaded TIFFs with acquisition timestamps, in date order. Optical-SAR needs one optical TIFF and one SAR TIFF.
- **Killing processes:** `pkill -f <pattern>` in a shell whose own command line contains that pattern kills the shell itself. Use `pgrep -f '[u]vicorn'` style patterns.
- **Sandbox Docker:** builds go through a TLS-intercepting proxy, and the GPU image (~4.5 GB) can exhaust the disk. A real VM won't have these problems.
- **Model identity in the repo:** don't put AI model identifiers in commits, PRs or code.
