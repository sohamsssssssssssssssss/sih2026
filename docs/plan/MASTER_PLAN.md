# SatQuery AI: Master Plan
Version 1.0, written 16 Nov 2026. Owner: Soham. Repo: `sohamsssssssssssssssss/sih2026` (local `/Users/soham/sih2026`).

Companion file: `SATQUERY_CONTEXT.md` (the state of the project today). This file is the direction. When the two disagree, the repo wins, then CONTEXT, then this plan.

Labels used everywhere in this file:
- **MEASURED** = came from eval code, traceable to a results file + commit.
- **TARGET** = a goal we set. Not a result. Never quote a TARGET as a result.
- **VERIFY** = believed true, not re-checked. Check before relying on it.

---

## Part 0. How any AI model must use this file

Any model (Opus in chat, Claude Code, Codex) has no memory between sessions. The repo is the memory.

### 0.1 Files that carry the project (create in H0, keep at repo root under `docs/plan/`)
| File | Purpose | Who updates |
|---|---|---|
| `MASTER_PLAN.md` | This file. Direction, architecture, phases. | Only on Soham's say |
| `CONTEXT.md` | Current state, branches, numbers, paths. | End of every session |
| `TRACKER.md` | Task table: id, status, branch, gate, evidence link. | End of every task |
| `RULES.md` | Section 0.3 of this file, verbatim. | Rarely |
| `DECISIONS.md` | Dated log: decision, why, what was rejected. | When a decision is made |
| `RESULTS.md` | Every MEASURED number: value, n, results file, commit. | When a result lands |

### 0.2 Session protocol (every session, every model)
1. Read `RULES.md`, `CONTEXT.md`, `TRACKER.md`. Read the plan section for the current task only.
2. State in 3 lines: the task id, the gate, what will not be touched.
3. Do one task. One branch per task: `task/<id>-<slug>`.
4. Stop at the gate. Show evidence (test output, results file, hash). Do not start the next task.
5. Update `TRACKER.md`, `CONTEXT.md`, and `RESULTS.md` if a number landed. Commit. Push.

For Opus in chat writing prompts for Claude Code: each prompt is a .txt file, opens with a 3 to 5 line context brief, names the task id, lists "do not touch", states the gate, and tells the agent to stop at the gate.

### 0.3 Rules (non-negotiable)
1. No number in any deck, README or UI unless it is MEASURED.
2. Write the hypothesis and the pass/fail threshold before running an eval. Commit it first.
3. Never fabricate coordinates, labels, results or citations. Unknown is written as unknown.
4. Fail closed: a tool that is not validated is registered as `experimental_unvalidated`, off by default, and the UI says so.
5. Every training run is a committed config before launch. Seed fixed. Staged sizes (tiny overfit first).
6. Long Kaggle jobs run as committed versions. Big files stay out of `/kaggle/working`. One GPU session at a time.
7. Ask Soham before: any Kaggle upload, any data deletion, any spend, any force-push, any merge to `main`.
8. Never touch Atharva's VLM training code or annotation files.
9. Resolution degradation is PSF blur, then decimate, then sensor noise. Never bicubic.
10. Controls are part of every eval (blank, shuffled, swapped). A model that scores the same on a blank image has learned nothing.
11. Datasets are deleted only after: results verified, pack hash recorded, source + rebuild steps written down.
12. The floor only goes up: `main` always runs the demo. Work lands on `main` only through a passing gate.
13. If a task is taking 2x its estimate, stop and report. Do not improvise a bigger design.

---

## Part 1. What "10 out of 10" means here

Nobody can promise a score. This is what we control, written as checks a judge can see.

| # | What judges see | Check (TARGET) |
|---|---|---|
| 1 | It works live | 12 golden scenes answer end to end, offline, on the Mac, twice in a row |
| 2 | It is multi-sensor for real | Same place answered from optical, from radar, and from both, with measured numbers for each |
| 3 | It is about India | Every headline number is measured on held-out Indian regions |
| 4 | It does not bluff | Unanswerable questions are refused with a reason; refusal precision is measured |
| 5 | Every answer shows proof | Mask or boxes on a map, confidence, and a trace hash per answer |
| 6 | The numbers are honest | Every number on screen links to a results file + commit |
| 7 | It solves a real problem | Monsoon mode: flood extent through cloud from radar, before/after |
| 8 | The builder understands it | Soham can explain each number, each failure, and each limit |

The differentiator in one line: **most teams will show a VLM that answers; SatQuery shows a system that knows when its answer can be trusted, and proves it on Indian data.**

Facts about the finale (see Sources): SIH 2026 Grand Finale is a 36-hour non-stop event at nodal centres in December 2026; exact date not published as of the sources found. Team status for the finale: unknown (open question Q1).

---

## Part 2. Where the project stands (progress ledger)

### 2.1 Done and MEASURED
| Item | Result | Evidence |
|---|---|---|
| Qwen2.5-VL-3B zero-shot, RSVQA-LR | 0.565 acc, n=200 | commit bbbde2c, results_v2.json |
| B13 RGB adapter, RSVQA-LR val (389 Q) | macro 0.696 vs base 0.468 | commit 20596be, `training/rgb-pilot-b12-b13` |
| B13 counting | 0.24 with real and with blank image (it guesses "0") | same |
| CDVQA G1 base zero-shot, test1-400 | macro 0.395 vs majority 0.44 | `training/cdvqa-g0` |
| BEN land-cover baselines (test macro mAP) | s1 0.5598, s2 0.6254, s1s2 0.6283 | commit 99f27ea (VERIFY path) |
| SAR Phase-0 gate | 5 scenes annotated, overclaims removed | commit 345b4a7 |
| Phase 0 freeze | 254 tests | SHA 14e8c14a |

### 2.2 Built, not yet measured
- **BEN → WorldCover transfer** (commit 4707d4c on `feature/ben-landcover`, not pushed). Pack: 10 regions x 300 patches (India: punjab, delhi, kerala, sundarbans, assam, rajasthan; Europe: vienna, alps, lisbon, kaunas). Kaggle notebook set up 16 Nov; run pending.
- **CDVQA G2 adapter** trained (1,000 steps). **G3 eval** launched 13 Nov; result not recorded.
- **Count routing** to Grounding DINO with a resolution gate (`feature/count-grounding`). Detector counting unmeasured.
- **Change-VQA route** behind a flag (`feature/change-vqa`, in `~/sih2026-change`).
- **B13 in app** (`app/b13-adapter`).
- Hash-chained trace (`orchestrator/trace.py`), resolution ladder (`eval/ladder.py`), Streamlit GUI (audit parts 7 to 11 unstarted).

### 2.3 Known problems
- Nothing from the last weeks is pushed or merged. A disk failure today would lose it.
- Five feature branches, no integration. `main` does not show the current system.
- The VLM cannot count (B13 evidence). The VLM alone is not a safe answer engine.
- Radar level shift: Europe WorldCover patches already sit +0.25 to +0.86 std above BEN on VV, so part of any India drop is data format, not geography.
- Rajasthan optical bands sit up to +3.2 std above BEN. Alps sample is biased (998 of 1,504 cells usable).
- No end-to-end evaluation of the whole agent exists.
- No free compute beyond Kaggle. RTX access gone.
- Solo builder.

### 2.4 Data on hand
| Data | Where | Use |
|---|---|---|
| BigEarthNet (S1+S2), ~111 GB | `~/Datasets/BigEarthNet` | Land-cover training |
| ben-pilot-20k | Kaggle dataset | Baseline training pack |
| WorldCover transfer pack, 566 MB | `~/ben-work/wc/` + Kaggle | India vs Europe transfer eval |
| RSVQA-LR, LoveDA, DOTA, DIOR, DIOR-RSVG, CDVQA | various | VQA, ladder, detection, change |
| 5 HyP3 RTC SAR scenes (Maharashtra, Gujarat) | repo `data/sar_gate/` | SAR reading |

---

## Part 3. The product

### 3.1 What the user does
1. Opens a map. Picks a place in India, or uploads an image (GeoTIFF or PNG).
2. Asks a question in plain language.
3. Gets: the answer, proof drawn on the map, a confidence level, which sensors were used, and a trace id.
4. If the question cannot be answered from this image, gets a refusal with the reason and what would be needed.

### 3.2 Question types (the contract)
| Type | Example | Tool | Sensor |
|---|---|---|---|
| Land cover | "How much of this is cropland?" | Land-cover model | S2, S1, S1+S2 |
| Presence | "Is there water here?" | Land-cover model / water index | S2, S1 |
| Flood (monsoon mode) | "Where is it flooded?" "What changed since before the rain?" | Flood model + before/after | S1 (through cloud), S2 if clear |
| Change | "What changed between these dates?" | Change model / differencing | S2 pairs, S1 pairs |
| Count / locate | "How many ships?" "Where are the storage tanks?" | Grounding DINO | High-res optical only |
| Open description | "Describe this area" | VLM (Qwen2.5-VL-3B + B13) | RGB |
| Context | "Which district is this? Is it flood-prone?" | Lookup (gazetteer, static facts) | none |
| Unanswerable | "How many cars?" at 10 m | Refusal with reason | n/a |

### 3.3 How one question flows
```
image/AOI + question
  -> INGEST     sensor, bands, GSD, CRS, date, cloud %, nodata %   -> Scene
  -> ROUTE      question type + needed GSD + needed sensor         -> Plan
  -> GATE       can this scene answer this? (resolution, cloud, sensor)  -> go / refuse
  -> RUN TOOL   specialist model or VLM                            -> ToolResult
  -> VERIFY     second, independent check                          -> agree / disagree
  -> CALIBRATE  confidence from held-out data; abstain below line  -> Answer
  -> TRACE      hash-chained record of every step                  -> trace id
  -> RENDER     answer + overlay + confidence + sensors + trace id
```

### 3.4 Architecture
```
frontend/         Next.js + MapLibre (H4). Streamlit stays as fallback UI.
api/              FastAPI. /ask /scenes /trace/{id} /health /capabilities
core/
  schema.py       Pydantic models (Part 4.2). Single source of truth.
  ingest.py       read image, extract metadata, build Scene
  router.py       question -> Plan (rules first, VLM fallback)
  gate.py         answerability checks (resolution, cloud, sensor)
  registry.py     tool registry, fail-closed, status per tool
  verify.py       cross-checks
  calibrate.py    temperature scaling + conformal thresholds
  trace.py        existing hash chain
tools/
  landcover.py  flood.py  change.py  detect.py  vlm.py  context.py
data/
  packs/          versioned packs + MANIFEST.json
  index/          tiles.parquet (queried with DuckDB)
  golden/         12 demo scenes with cached answers
eval/
  eval.py  ladder.py  bench/  controls.py
training/         configs/*.yaml, kaggle cells, checkpoints by hash
docs/plan/        the six files from Part 0.1
```
Design rules: every tool has the same interface (`Scene, Query -> ToolResult`); tools never call each other; only `core/` orchestrates; models load once at startup; nothing in `api/` or `tools/` reads BigEarthNet.

---

## Part 4. Data

### 4.1 Sources (all free)
| Source | What | Use |
|---|---|---|
| BigEarthNet v2 | S1+S2 patches, Europe, 19 labels | Train land cover |
| ESA WorldCover 2021 | 10 m global land cover + S1/S2 composites | India eval + India adaptation labels |
| Sen1Floods11 | Paired S1/S2 flood chips, 11 events, hand labels; test split includes India chips (VERIFY licence + exact India count) | Train + test flood model |
| Copernicus S1/S2 via free STAC | Any place, any date | Before/after pairs for Indian flood events; demo scenes |
| DIOR / DOTA / DIOR-RSVG | High-res optical with boxes | Detector counting eval |
| CDVQA, RSVQA-LR | VQA benchmarks | VLM + change eval |
| NISAR (ASF L-band, Bhoonidhi S-band) | ISRO-NASA radar, public since 2026 | Post-hackathon (Part 12). One sample scene in the demo if time allows |
| Bhuvan / NRSC flood layers | Reference flood maps (VERIFY access terms) | Sanity reference only |

Not available free: sub-metre Cartosat. The resolution ladder is the honest answer to that gap: it shows measured accuracy versus GSD and what the system refuses at each level.

### 4.2 Core data structures (`core/schema.py`)
```python
class Scene:            # one image the system can reason about
    scene_id: str               # sha256 of pixels + metadata
    sensor: Literal["S2","S1","RGB_HR","S1S2","UNKNOWN"]
    bands: list[str]            # canonical order, e.g. ["B02","B03","B04","B08"]
    units: str                  # "reflectance_x1e4" | "dB" | "uint8"
    gsd_m: float                # ground sample distance
    crs: str; bounds: tuple[float,float,float,float]
    acquired: date | None
    cloud_pct: float | None; nodata_pct: float
    source: str                 # pack id or "upload"
    region: str | None          # held-out group key, e.g. "assam"

class Patch:            # one training/eval sample inside a pack
    patch_id: str; scene_id: str
    window: tuple[int,int,int,int]
    split: Literal["train","val","test","holdout_india"]
    region: str
    labels: dict                # task -> label (fractions, mask path, boxes)

class PackManifest:     # one per pack, MANIFEST.json
    pack_id: str; version: str; created: datetime
    builder_commit: str         # code that built it
    files: dict[str,str]        # relative path -> sha256
    n_patches: int; splits: dict[str,int]; regions: dict[str,int]
    band_order: list[str]; units: str
    norm_stats: dict            # per-band mean/std from TRAIN split only
    checks: dict[str,bool]      # nodata, pairing, alignment, leakage
    sources: list[str]          # upstream URLs / dataset versions

class Query:
    text: str
    qtype: Literal["landcover","presence","flood","change","count","locate","open","context"]
    needs_gsd_m: float | None   # coarsest GSD that can answer it
    needs_sensor: list[str]

class Evidence:
    kind: Literal["mask","boxes","fractions","diff","text"]
    uri: str | None             # file with mask/boxes (GeoJSON / COG)
    summary: dict               # e.g. {"cropland": 0.41, "water": 0.07}

class ToolResult:
    tool: str; tool_version: str; checkpoint_sha: str | None
    value: Any; raw_confidence: float | None
    evidence: list[Evidence]; latency_ms: int

class Answer:
    status: Literal["answered","refused","low_confidence"]
    text: str
    confidence: float | None          # calibrated
    refusal_reason: str | None        # "gsd_too_coarse" | "cloud" | "tool_unvalidated" | "checks_disagree" | "out_of_distribution"
    sensors_used: list[str]
    primary: ToolResult | None
    verification: dict                # check name -> agree/disagree/skipped
    trace_id: str

class EvalRecord:       # one row in results files
    eval_id: str; commit: str; pack_id: str; checkpoint_sha: str | None
    metric: str; value: float; n: int; ci95: tuple[float,float] | None
    control: str | None               # "blank" | "shuffled" | "swapped" | None
    hypothesis_ref: str               # path to the pre-committed hypothesis
```

### 4.3 Cleaning pipeline (one builder per pack, same nine steps)
1. **Fetch** with checksums. Record source URL and version.
2. **Decode** to canonical band order and units. Assert, never assume (WorldCover S2 files are B04,B03,B02,B08; S1 is already dB).
3. **Align** S1 and S2 to one grid. Assert same CRS, transform and shape.
4. **Filter**: drop patches with nodata, cloud above threshold, or saturated bands. Log the drop count per region (the Alps bias is a logged fact, not a surprise).
5. **Pair check**: every pixel array carries its id; re-open 1% at random and compare hash to manifest. This is the check that would have caught the invalid 150k shards.
6. **Label map**: one table, BEN 19 classes <-> coarse classes (trees, shrub_grass, cropland, built, water, wetland) <-> WorldCover codes. Classes with no mapping (WorldCover 60 bare) are marked unlabelled, never guessed.
7. **Split**: geographic blocks. Held-out Indian regions never appear in training. Assert zero overlap by block id.
8. **Stats**: per-band mean/std from the train split only. Stored in the manifest. Each checkpoint stores the stats it was trained with.
9. **Manifest + tests**: write `MANIFEST.json`, run pack tests (shape, dtype, ranges, class balance, leakage, duplicates), then freeze. A changed pack is a new version.

### 4.4 Storage and scale
- BigEarthNet stays on disk as the raw source. Training reads packs, not raw.
- Packs are numpy memory-maps or shards with an index, sized to fit Kaggle (under about 15 GB each).
- `data/index/tiles.parquet` lists every patch (id, pack, region, split, bounds, sensor, date). DuckDB queries it in milliseconds.
- The demo bundle is separate and small: 12 golden scenes + checkpoints + cached answers, about 1 to 3 GB.
- Large rasters are read with windowed reads (rasterio), tiled to model input size, and merged with overlap.

---

## Part 5. Models

### 5.1 Tool roster
| Tool | Model | Status | Priority |
|---|---|---|---|
| Land cover (s1 / s2 / s1s2) | Small CNN/ViT from `ben/landcover.py` | Baseline MEASURED on BEN; India transfer pending | P0 |
| Land cover, India-adapted | Same, adapted on Indian WorldCover regions | Not started | P0 |
| Flood | Small U-Net on S1 (+S2 when clear), Sen1Floods11 | Not started | P0 |
| Change | (a) S1/S2 differencing baseline (b) CDVQA adapter | (a) not started (b) G3 pending | P1 |
| Detect / count | Grounding DINO zero-shot, tuned thresholds | Built, unmeasured | P1 |
| VLM open VQA | Qwen2.5-VL-3B + B13 adapter | MEASURED | P0 (exists) |
| Context lookup | Gazetteer + static facts, no model | Not started | P2 |
| Foundation backbone bake-off | TerraMind tiny/small (IBM + ESA, open on Hugging Face) as encoder for land cover and flood | Not started | P2, gated |

### 5.2 Training method (every model)
1. Hypothesis + threshold committed.
2. Config committed (`training/configs/<run>.yaml`): pack id, model, seed, steps, lr, batch.
3. Tiny overfit (16 to 64 samples) must reach near-zero loss. If not, the pipeline is broken; fix before scaling.
4. Staged scale: 2k, then 20k, then full pack. Stop scaling when the val gain per stage is under 1 point.
5. Each run saves: checkpoint (named by sha), norm stats, config, logits on test + holdout, results file.
6. Controls run in the same cell: shuffled labels, blank input.
7. Results go to `RESULTS.md` with commit + pack id.

### 5.3 The India adaptation experiment (the scientific core)
Question: does a Europe-trained model work in India, and can a small amount of Indian data fix it?
- Step A (pending run): Europe-trained model on WorldCover Europe vs India. Gives format shift (Europe minus BEN), geography shift (India minus Europe), total drop (India minus BEN).
- Step B: adapt on 3 Indian regions (for example punjab, kerala, assam), test on the other 3 held out (delhi, sundarbans, rajasthan). Then swap and repeat, so every region is tested once while held out.
- Three adaptation arms, cheapest first: (1) re-fit normalisation only, (2) fine-tune the last layers, (3) full fine-tune. Report all three.
- Labels are WorldCover fractions (a map product, not ground truth). Say so on screen.
- Output: one chart, "before vs after adaptation, per held-out Indian region, per sensor mode". This chart is the centre of the pitch.
- TARGET: adapted model recovers at least half of the geography drop on held-out regions. If the measured drop is small, the finding is "it transfers", and that is also a result.

### 5.4 Flood model (monsoon mode)
- Train: Sen1Floods11 hand-labelled chips; S1-only model first (works through cloud), S1+S2 second.
- Test: the official test split, reported per event; India chips reported separately.
- Baseline to beat: Otsu threshold on VH (the dataset ships it). If the model does not beat Otsu on India chips, ship Otsu and say so.
- Demo: 2 to 3 real Indian flood events as before/after S1 pairs (candidates: Assam 2022, Kerala 2018, Bihar 2019; pick by data availability). Output: flood mask, flooded area in km2, flooded cropland area (intersect with land-cover tool).
- Verification: permanent water is subtracted using a pre-event scene; S2 water index cross-checks when the sky is clear.

### 5.5 VLM policy
- The VLM answers open description questions only. It is never the source for counts, areas or change amounts.
- Pinned: Qwen2.5-VL-3B rev 66285546 + B13 adapter. No backbone swap before the finale.
- Serving on the Mac is task T12: measure latency with torch MPS and with MLX 4-bit. If the adapter cannot be served within about 15 s per answer, golden scenes use cached answers (labelled as cached) and live runs use the base model.

### 5.6 Compute budget (₹0)
One Kaggle account: about 30 GPU hours per week, sessions up to 12 h (see Sources; the quota shows in account settings).
| Job | GPU hours (estimate) |
|---|---|
| Transfer run (pending) | 0.3 |
| India adaptation, 3 arms x 2 folds x 3 modes | 6 to 8 |
| Flood model, staged | 3 to 5 |
| Detector counting eval | 2 |
| Change baseline + G3 follow-up | 3 to 6 |
| TerraMind bake-off (optional) | 4 to 6 |
| Buffer for reruns | 8 |
Total fits inside 2 weeks of quota. Paid compute is not needed for the hackathon plan. The only spend worth considering: about ₹2,000 to ₹5,000 of rented GPU for one larger run, and only after finale selection is confirmed.

---

## Part 6. The trust layer (why the answers are legitimate)

### 6.1 Answerability gate (before any model runs)
| Check | Rule | Refusal reason |
|---|---|---|
| Resolution | Query needs GSD finer than the scene has (table of object size vs GSD, backed by ladder results) | `gsd_too_coarse` |
| Cloud | Optical scene cloud % above threshold and no radar available | `cloud` |
| Sensor | Question needs a sensor that is missing | `sensor_missing` |
| Tool status | Tool is `experimental_unvalidated` | `tool_unvalidated` |
| Distribution | Band statistics more than k std from the training stats (the Rajasthan case) | `out_of_distribution` (answer with warning, or refuse) |

### 6.2 Verification (after the model runs)
- **Cross-sensor**: land cover from S1 vs from S2. Disagreement lowers confidence.
- **Cross-method**: flood model vs Otsu; water class vs water index; detector count vs VLM count (VLM is the weaker witness).
- **Self-consistency**: fractions sum to about 1; flooded area cannot exceed AOI area; change with identical before/after must be "no change".
- **Perturbation**: small shifts or flips of the input must not flip the answer.
Each check writes agree / disagree / skipped into `Answer.verification`.

### 6.3 Calibration and abstention
- Temperature scaling per tool on the val split.
- Split conformal prediction on held-out Indian data to set the abstention line. This gives a statement that can be defended: "on held-out Indian regions, when the system answers, the error rate is at most X%". X is a TARGET until measured (aim 10%).
- Report selective accuracy (accuracy on answered questions) together with coverage (share answered). Never one without the other.

### 6.4 Controls (run in every eval, shown in the UI's evidence tab)
| Control | Expected behaviour |
|---|---|
| Blank image | Score falls to chance, or the system refuses |
| Shuffled labels | Score at prevalence level |
| Swapped before/after | Change answers flip or the system notices |
| Identical before/after | "No change" |
| Wrong sensor fed in | Refused at the gate |
| Degraded resolution (ladder) | Accuracy falls smoothly; refusals rise |

### 6.5 When it still gets something wrong
- The trace shows which tool, which checkpoint, which checks ran. The error can be found in minutes.
- A "known failures" page in the app lists real failure cases with the reason. Showing this to judges is a strength.

---

## Part 7. Backend implementation

### 7.1 Stack
Python 3.11. FastAPI + Pydantic v2 + Uvicorn. PyTorch (MPS on Mac, CUDA on Kaggle). Rasterio + NumPy. DuckDB + Parquet. PyTest. Frontend: Next.js + MapLibre GL (H4); Streamlit kept as fallback.
Not used before the finale: LangGraph, vLLM, any vector database. The router is a small rules table with a VLM fallback; it is faster, testable and deterministic.

### 7.2 API
| Endpoint | Does |
|---|---|
| `POST /scenes` | Upload or register an AOI; returns `Scene` |
| `POST /ask` | `{scene_id(s), question}` -> `Answer` |
| `GET /trace/{id}` | Full hash-chained trace |
| `GET /capabilities` | Tool registry with status + MEASURED numbers per tool |
| `GET /golden` | List of demo scenes |
| `GET /health` | Models loaded, versions, commit |

### 7.3 Performance plan
- Models load once at startup; small models stay warm; the VLM loads lazily.
- Cache key: `(scene_id, question_normalised, tool_version)`. Golden scenes are pre-warmed.
- Latency TARGETS: router under 100 ms; land cover and flood under 2 s on the Mac; detector under 5 s; VLM under 15 s or cached.
- Measure p50 and p95 per tool in the bench. Show them in `/capabilities`.

### 7.4 Testing
- Unit tests per tool with a 4-patch fixture pack.
- Contract tests: every tool returns a valid `ToolResult`; every `/ask` returns a valid `Answer`.
- Golden test: the 12 golden scenes produce the stored answers (within tolerance). This test gates every merge to `main`.
- Offline test: run the golden test with networking off.

---

## Part 8. Evaluation: SatQuery-Bench-India

One benchmark for the whole agent. This is new work nobody else at the finale is likely to have.

- **Size**: about 300 questions (TARGET), built from templates over held-out data, so ground truth is computed, not hand-written.
- **Mix**: land cover fractions (80), presence (40), flood (50), change (40), count/locate (40), unanswerable (50).
- **Ground truth**: WorldCover fractions; Sen1Floods11 hand labels; DIOR/DOTA boxes; CDVQA labels; unanswerable items are built by construction (question needs finer GSD than the scene, or cloud-covered optical with no radar).
- **Metrics**: accuracy per type; selective accuracy + coverage; refusal precision and recall on unanswerable items; calibration error; control collapse; latency p50/p95.
- **Runs**: full system; then ablations (no verifier, no gate, VLM-only). The VLM-only row is the comparison that shows why the system design matters.
- **Rules**: bench frozen by hash before the first full run; hypotheses committed first; held-out regions never used in adaptation.

---

## Part 9. Hackathon plan (16 Nov to finale)

Finale date unknown. Plan so that a complete, honest demo exists by **30 Nov** (the floor), and everything after only raises it. If the date turns out earlier, cut from the bottom of the cut list (9.4).

### H0. Secure and consolidate (16 to 18 Nov)
| Task | Gate |
|---|---|
| T01 Push every local branch to GitHub (no merges) | `git branch -r` lists all 6+ branches |
| T02 Finish the BEN transfer Kaggle run; download outputs | `CELL DONE`; retrain within about 0.01 to 0.02 of 0.5598 / 0.6254 / 0.6283 (GPU nondeterminism), else investigate |
| T03 Record CDVQA G3 result | Numbers + controls in `RESULTS.md` |
| T04 Create `docs/plan/` with the six files | Files exist, CONTEXT matches repo |
| T05 Integration branch: merge feature branches into `integration/h0` one at a time, tests green after each | Full test suite passes; then merge to `main` with Soham's approval |

### H1. The spine (19 to 22 Nov)
| Task | Gate |
|---|---|
| T06 `core/schema.py` with all models from Part 4.2 + contract tests | Tests pass |
| T07 Tool registry (fail-closed) + wrap existing tools (VLM, detector, land cover) behind one interface | `/capabilities` lists tools with status |
| T08 Router + answerability gate (rules table, GSD table from ladder) | 40 routing test cases pass, including 10 refusals |
| T09 FastAPI `/ask` end to end with trace | One golden scene answered through the API, trace verifies |
| T10 Point Streamlit at the API | Old UI works on the new backend |
| T11 Analyse transfer results; write the 3 shift numbers | Chart + `RESULTS.md` entries |
| T12 VLM serving on Mac: measure latency, pick serving mode | Decision in `DECISIONS.md` with numbers |

### H2. Sensor tools (23 to 27 Nov)
| Task | Gate |
|---|---|
| T13 India adaptation packs + 3 arms x 2 folds (Part 5.3) | Before/after chart from MEASURED numbers |
| T14 Flood pack from Sen1Floods11 (cleaning pipeline) | Manifest + pack tests pass |
| T15 Flood model, staged; compare to Otsu per event | Results per event; ship the better one |
| T16 Indian flood before/after scenes (2 to 3 events) | Scenes in golden set with masks + areas |
| T17 Detector counting eval on DIOR/DOTA subset; set thresholds | Counting MAE + refusal rule MEASURED |
| T18 Change baseline (differencing) + decide on CDVQA adapter from G3 | Change tool registered with honest status |

### H3. Trust layer (28 to 30 Nov)
| Task | Gate |
|---|---|
| T19 Verifier checks (Part 6.2) | Each check has a test with an agree and a disagree case |
| T20 Calibration + conformal abstention | Selective accuracy + coverage on held-out India MEASURED |
| T21 Build and freeze SatQuery-Bench-India | Bench hash recorded |
| T22 Full bench + ablations + controls | Results table in `RESULTS.md` |
| T23 12 golden scenes final; offline golden test | Passes twice with networking off |
**30 Nov = floor reached.** From here `main` is always demo-ready.

### H4. Map front end (1 to 5 Dec)
| Task | Gate |
|---|---|
| T24 Next.js + MapLibre shell: map, AOI pick, upload, ask box | Talks to `/ask` |
| T25 Overlays: masks, boxes, before/after slider | Golden scenes render |
| T26 Evidence panel: confidence, sensors, checks, trace id, link to results | Every number on screen traces to a file |
| T27 Capabilities + known-failures pages | Built from `/capabilities` and `RESULTS.md`, no hand-typed numbers |
| T28 Offline bundle (local tiles, no CDN) | Works with Wi-Fi off |

### H5. Finale readiness (6 Dec until T-2 days)
| Task | Gate |
|---|---|
| T29 Full rehearsal x3 on the Mac, timed | No failures in the last 2 runs |
| T30 Finale kit (9.3) packed and tested on a clean user account | Checklist signed off |
| T31 Q&A sheet: every number, every limit, every "why not X" | Soham answers all without notes |
| T32 Freeze at T-2. Only fixes after that | Tag `finale-rc` |

### 9.3 Finale kit (the finale is a 36-hour build at a nodal centre)
- Repo at tag `finale-rc` + all checkpoints + packs needed for the demo, on the Mac and on a USB drive.
- Kaggle notebooks pre-created for any retrain, with datasets attached.
- A ready list of small "live upgrades" to build during the 36 hours, since judges and mentors check progress in rounds: (1) a new Indian region added to the bench, (2) a new question template, (3) a mentor-suggested AOI answered live, (4) NISAR sample scene loaded, (5) Hindi question input.
- Mentor feedback log: every suggestion written down, with a response by the next round.
- Phone hotspot + offline mode. Charger. Printed one-page architecture.

### 9.4 Cut list (drop from the bottom if time is short)
1. Keep always: spine, land cover + India adaptation chart, flood mode, gate + refusals, golden scenes, offline.
2. Keep if possible: calibrated confidence + conformal line, bench + ablations, detector counting.
3. Drop first: Next.js UI (Streamlit stays), TerraMind bake-off, context lookup, CDVQA adapter in app, NISAR sample, Hindi input.

### 9.5 Fallback ladder
| If this fails | Fall back to |
|---|---|
| India adaptation does not help | Report the measured gap honestly; ship the best arm; gate flags low confidence in India |
| Flood model loses to Otsu | Ship Otsu + permanent-water subtraction |
| S1+S2 fusion is not better than S2 | Ship S2 when clear, S1 when cloudy; report the numbers |
| CDVQA adapter fails controls | Differencing baseline only; adapter stays off |
| VLM too slow on Mac | Cached answers for golden scenes (labelled), base model live |
| Next.js not ready | Streamlit |
| Kaggle quota runs out | CPU jobs on Mac for small models; wait for weekly reset; paid GPU only with approval |
| Venue internet fails | Offline bundle |
| Laptop fails | USB kit + repo on GitHub + a teammate's laptop pre-installed in H5 |

---

## Part 10. Features beyond the submitted PPT

The PPT promised: agentic model selection, optical + SAR, temporal change, visual evidence, natural language, hash-chained trace, resolution ladder, flood and damage impact. The plan delivers those and adds:

| New feature | Why it matters | Cost |
|---|---|---|
| Answerability gate + reasoned refusal | A system that says "this image cannot answer that" is what an operational user needs | Low |
| Self-verification across sensors and methods | Catches VLM guessing; visible in the UI | Medium |
| Conformal abstention with a measured error bound | A defensible statement, rare at student level | Medium |
| Measured India transfer + adaptation | Directly answers "will this work on Indian data?" | Medium |
| SatQuery-Bench-India | The team brings its own benchmark, with ablations | Medium |
| Monsoon mode | Radar through cloud, flood area and flooded cropland | Medium |
| Known-failures page | Honesty shown as a feature | Low |

### Hard ideas, ranked
| Idea | Verdict |
|---|---|
| Resolution-aware refusal driven by ladder measurements | Do it (it is the gate) |
| Conformal guarantees on held-out geography | Do it |
| Leave-regions-out India adaptation study | Do it |
| TerraMind as encoder vs own baseline | Optional, gated: only if H3 ends early |
| Monitoring agent: a standing question re-asked on each new satellite pass, with alerts | After the hackathon (Part 12) |
| Label-free test-time adaptation to a new region | After the hackathon |
| Generating optical images from radar (or the reverse) | No. It invents pixels; it conflicts with Rule 3 |
| Training a foundation model from scratch | No. Not possible on free compute, and not needed |
| Swapping the VLM backbone before the finale | No. Adapters and evals are tied to the pinned model |

---

## Part 11. Risks

| Risk | Likelihood | Impact | Response |
|---|---|---|---|
| Work lost before push | Medium | Severe | T01 today |
| Not selected for finale | Unknown | High | Plan value survives: Part 12 starts in December instead of January |
| Finale earlier than expected | Medium | High | Floor on 30 Nov; cut list |
| Solo overload | High | High | One task at a time; gates; cut list; no new scope before the floor |
| Agent drifts or breaks working code | Medium | High | Part 0 protocol; golden test gates merges |
| A result is wrong or irreproducible | Medium | Severe | Rule 1, 2, 10; retrain check in T02 |
| Kaggle quota | Medium | Medium | Budget in 5.6; buffer; staged runs |
| Mac memory (16 GB) | Medium | Medium | Lazy VLM load; one big model in memory at a time |
| Judges ask about sub-metre ISRO data | High | Medium | Ladder results + refusal behaviour; say plainly that no Cartosat data was available |
| Overlap with other commitments in late Nov | Unknown | Medium | Open question Q3 |

---

## Part 12. After the hackathon: the long plan

The project becomes a standing AI project: an open, honest question-answering layer for satellite data over India. Each phase has a goal, deliverables, a gate, and a stop condition. Dates shift by a month if the finale outcome changes things.

### Phase A. Make it public and reusable (Jan to Mar 2027) — ₹0
Goal: anyone can run it, and it can be reused for other events.
- A1 Clean release v1.0: one-command install, docs, licence check on every dataset and model, demo data bundle.
- A2 Public demo on a free host (Hugging Face Space, small models only; VLM answers cached).
- A3 Publish SatQuery-Bench-India (questions, builder code, baseline results, leaderboard table in the README).
- A4 Technical report (8 to 10 pages): system, India transfer study, abstention results. Post as a preprint.
- A5 Mission packs: the same engine with swappable packs so it fits open-innovation themes: flood (exists), crop monitoring, urban growth, coastal and mangrove change, forest loss, water bodies. Each pack = AOIs + question templates + tool config + bench slice.
- Gate: a stranger clones the repo and runs the golden test in under 30 minutes.
- Stop condition: none; this phase is cheap and makes everything later possible.

### Phase B. Real Indian data and time (Apr to Sep 2027) — ₹0 to ₹5,000
Goal: move from Sentinel-only to ISRO sources and from single images to time series.
- B1 NISAR: L-band from ASF and S-band from Bhoonidhi are public. Add ingest, normalisation and a first land-cover/flood eval on NISAR. Being early on NISAR over India is a real opening.
- B2 Bhoonidhi open data (Resourcesat, EOS-04 where access allows; VERIFY terms): ingest + ladder points at real ISRO GSDs.
- B3 Time series: per-AOI stacks, change-point detection, seasonal baselines (crop cycles, reservoir levels).
- B4 Label-efficient adaptation: active learning tool (the model picks the patches most worth labelling); test-time adaptation without labels.
- B5 Bench v2: more regions, more seasons, NISAR slice, human-checked subset.
- B6 Submit a workshop paper (remote sensing or ML-for-Earth venues; pick by deadline at the time).
- Gate: one result on real ISRO-sensor data that was not possible in Phase A.
- Stop condition: if ISRO data access is blocked for two months, stay on Sentinel + NISAR-ASF and move on.

### Phase C. From answers to monitoring (Oct 2027 to Sep 2028) — ₹0 to low
Goal: the system watches places, not just answers once.
- C1 Monitoring agent: a standing question ("tell me when this reservoir drops below X", "flag new built-up area in this ward") re-evaluated on each new pass, with alerts and a trace per alert.
- C2 India EO adapter: a set of small adapters on an open foundation backbone, trained across Indian agro-climatic zones, released openly. Compare against Phase A models on Bench v2.
- C3 API + Python client; usage examples for students and researchers.
- C4 One real pilot: a college department, an NGO, or a district disaster cell uses it for one season. Collect what they needed that it could not do.
- C5 Derived indicators (crop area, flood exposure, built-up growth, night-lights where available) exported as clean time series, for research and for economic or market analysis.
- Gate: one external user running it without help.
- Stop condition: no external user after a real 3-month effort means the project stays a research/portfolio asset and stops growing in scope.

### Phase D. Choose the fork (Oct 2028 to 2029)
Decide based on what Phase C showed:
- **Research path**: full paper on honest EO question answering under geographic shift; Bench v3 as a community resource.
- **Product path**: hosted monitoring service for one sector (agri-insurance, disaster response, or infrastructure), with paid compute covered by users or grants.
- **Portfolio path**: freeze as a well-documented flagship; keep the bench maintained.
All three keep the same core: measured answers, refusal when unsure, proof per answer.

### What carries across all phases
- The six plan files, the rules, the bench, the golden test.
- A quarterly review: what was MEASURED, what was cut, what changed in the plan, written to `DECISIONS.md`.

---

## Part 13. Budget summary
| Period | Planned spend | Possible spend (needs approval) |
|---|---|---|
| Now to finale | ₹0 | ₹2,000 to ₹5,000 GPU rental, only after finale selection is confirmed and only if a run cannot fit Kaggle |
| Phase A | ₹0 | Domain name (optional) |
| Phase B | ₹0 | ₹0 to ₹5,000 compute |
| Phase C+ | ₹0 | Hosting if a pilot needs uptime; seek student credits and grants first |

---

## Part 14. Open questions for Soham
- Q1 Is the team confirmed for the finale? If yes, the date and nodal centre.
- Q2 Merge policy: may the agent merge the integration branch to `main` after tests pass, or does Soham approve each merge? (Plan assumes approval each time.)
- Q3 Any other commitments between now and the finale that take full days? The 30 Nov floor assumes most days are available.
- Q4 Which Indian flood events matter most to show (Assam, Kerala, Bihar, Maharashtra)?
- Q5 Should the remaining teammates get non-technical tasks from this plan (bench question review, flood event research, Q&A sheet)?

---

## Sources (checked 16 Nov 2026)
- SIH 2026 timeline (finale in December 2026, 36-hour, nodal centres): https://blogs.reskilll.com/smart-india-hackathon-2026-launched-timeline-registration-how-to-participate/ and https://www.startupgrantsindia.com/competitions/smart-india-hackathon-2026-official
- TerraMind (IBM + ESA, open on Hugging Face, S1/S2 pretraining): https://huggingface.co/collections/ibm-esa-geospatial/terramind-10 and https://arxiv.org/pdf/2504.11171
- Sen1Floods11 (4,831 chips, 11 flood events, paired S1/S2, hand labels): https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf and https://github.com/cloudtostreet/Sen1Floods11
- Kaggle free GPU quota (about 30 h per week, 12 h sessions): https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- NISAR public data (L-band via ASF since July 2026; S-band via Bhoonidhi): https://asf.alaska.edu/notices/nisar-l-band-data-now-publicly-available/ and https://bhoonidhi.nrsc.gov.in/NISAR/NisarProducts.html
