# SatQuery AI

## What SatQuery Is

> "Ask your satellite data anything."

SatQuery AI is a remote-sensing question-answering system built for SIH26167. It routes natural-language questions over Sentinel-1/2 imagery to registered tools, refuses when its gates or tool status say an answer is not supported, and records every answer in a SHA-256 hash-chained trace. Every model tool is experimental or disabled; see the evidence-linked summary in [docs/STATUS.md](docs/STATUS.md).

---

## Current Status

For a one-page summary with a source for every number, see [docs/STATUS.md](docs/STATUS.md).

Judge Q&A: [docs/judge/JUDGE-QA.md](docs/judge/JUDGE-QA.md).

### /ask (current demo path)
- **Live scenes**: `/ask` finds a scene by place and date. Sentinel-2 L2A comes from Earth Search and Sentinel-1 RTC from Microsoft Planetary Computer, with a 120 × 120 window read per band (`backend/fresh_scenes.py`). Five cached backup chips with SHA-256 checksums cover provider outages (`data/fresh-backups/manifest.json`). The overall fetch deadline is 20 seconds, including place lookup, STAC search and band reads. At expiry, Ask shows “Live fetch timed out after Ns — showing cached scene from <date>” for a matching backup or refuses with a reason when none exists. The S2 per-read timeout is 60 seconds (`ecb651e`, `5a778ba`).
- **Gates**: scenes outside the India rectangle [68, 6, 98, 38], optical scenes above 50% cloud and scenes above 50% nodata are refused. These thresholds are provisional guards, not calibrated cutoffs (`core/gate.py:12`).
- **Land cover — experimental**. Ask uses the Europe-trained BigEarthNet checkpoints and returns `confidence: null`. Built areas ranked first in live Ludhiana (**0.955**) and Delhi (**0.949**), and cropland ranked first on the Punjab fixture (**0.378**). These are uncalibrated presence scores. The Earth Search BOA offset fix (`84d7f57`) corrected the tested optical inputs. The India-adapted model was evaluated and is not shipped because calibration did not transfer between held-out Indian regions. [Recorded API results](docs/plan/INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups) · [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)
- **Flood water — experimental**: per-scene Sentinel-1 Otsu open-water mask. India-test IoU 0.631 against a 0.65 gate: FAIL, method not tuned afterwards (`outputs/T07c-flood.md`). The flood U-Net is parked (`docs/plan/DECISIONS.md`).
- **Mac single-image VQA — experimental**. The demo launcher enables the pinned 8-bit MLX B13 worker with `SATQUERY_VQA_MLX=1` (integration `1273a83`). Ask refuses counting. On the 400-question RSVQA-LR test sample, B13 scored **73.25%** macro (95% CI **69.99–76.50%**) against **54.50%** for base and **57.25%** for train-majority. The real-image advantage over a grey image was **11.75 percentage points** (95% CI **7.50–16.00**). There were **0 malformed answers**, median tool latency was **0.393 s**, and G6 run 2 kept at least **3.227 GiB** available against a **3 GiB** requirement. [Mac runbook](docs/RUNBOOK.md) · [Fresh-test gate](docs/gates/VQA-MAC-FRESH-TEST.md) · [Fresh-test results](results/vqa-mac-fresh-test/REPORT.md)
- **Mac parity with CUDA remains failed**. The base model exceeded the CUDA malformed-answer count in both attempts. The bf16 attempt also missed the memory requirement. The fresh test evaluated the shipped 8-bit model on held-out data with its own gate. [Parity attempt 1](results/vqa-mac-parity/REPORT.md) · [Parity attempt 2](results/vqa-mac-parity/attempt2/REPORT.md)
- **Change QA — disabled**: change questions receive `not_answerable`. T18 found CDVQA presence labels predictable from question text alone (82.8%) and pixel differencing (56.0%) below train majority (56.2%) (`docs/plan/DECISIONS.md`, commit `252ea5e`).

The integration log has the latest test totals, including the production frontend Playwright runs. [Integration log](docs/plan/INTEGRATION_LOG.md) · [Judge-style HTTP acceptance](docs/plan/E2E_ACCEPTANCE.md)

### AVAILABLE (API building blocks)
- **Controlled image ingestion**: `POST /api/scenes` validates PNG, JPEG, TIFF, and GeoTIFF uploads up to 20 MiB. Raster metadata is observed separately from user declarations.
- **Deterministic orchestration**: `POST /api/plan` selects capabilities, checks input shape and provider readiness, and emits auditable execution steps. Unsupported multi-step plans fail closed.
- **Single-image VQA**: `Qwen/Qwen2.5-VL-3B-Instruct` executes from offline local artifacts on CUDA, with opt-in MPS for local iteration. Confidence is `null` when uncalibrated.
- **Grounding**: Grounding DINO Swin-T returns normalized bounding-box evidence with fixed thresholds (`box=0.35`, `text=0.25`).
- **Pair compatibility**: Optical-SAR and bi-temporal requests are checked for modality, acquisition metadata, overlap, dimensions, CRS, and affine grid before dispatch.
- **Deterministic optical-SAR analysis**: Joint Sentinel-2 index and Sentinel-1 backscatter summaries consume both co-registered inputs. This is not a learned fusion model.
- **Deterministic change analysis — disabled**: the `change-deterministic` tool is disabled by decision and `change_vqa` is advertised as unsupported (`core/tool_status.json`, commit `252ea5e`).
- **Sentinel-1 water change (before the change-QA decision)**: Two co-registered VV/VH/dataMask SAR scenes sent to `change_vqa` reported new and receded open water in hectares and as GeoJSON polygons, from a split-based Otsu baseline (bimodal tiles, after Chini et al. 2017) that abstains when no part of the scene shows an open-water mode, with a ±1 dB sensitivity range. Where village boundaries exist (the Bihar and Kerala catalogue events), the answer names the most flooded villages with hectares. Confidence is `null`. See `docs/change-baseline.md`.
- **Sentinel-1 fetch from CDSE**: `POST /api/sentinel1/pairs` or `python -m backend.sentinel1 --event kosi-2024` (or an explicit `--bbox`, `--before` and `--after`; needs `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`) picks a same-orbit-track pair, renders terrain-corrected gamma0 for both dates on one UTM grid (tiled and mosaicked up to 50 km per side) and ingests them as scenes. It also records how cloudy Sentinel-2 was over the AOI near the post-event date. Verified live on all three catalogued events; results are in `data/manifests/cdse/flood-runs.v1.json`. The endpoint has no authentication and runs one fetch at a time, so don't expose it publicly yet.
- **Cryptographic audit trace**: Live and cached-result executions are distinguished in a SHA-256 hash-chained trace.
- **Explicit cached replay**: The designated committed VQA result is returned only when the caller requests `execution_mode: "cached_result"`; live failure never silently falls back.

Historical Kaggle Tesla T4 smoke verification passed through the repository provider path for Qwen and Grounding DINO. See `docs/gpu-smoke.md`. This proves runtime execution, not model accuracy or throughput.

### PARTIAL / BLOCKED ON GPU OR DATA
- **Remote-sensing adaptation**. The Mac demo serves the B13 adapter for single-image VQA. The CDVQA G3 adapter retained much of its performance on identical image pairs, and India land-cover adaptation failed promotion after passing T13b. [Mac VQA test](results/vqa-mac-fresh-test/REPORT.md) · [CDVQA results](docs/plan/RESULTS.md) · [India promotion decision](results/landcover-india-promotion/DECISION.md)
- **Multi-step change-to-grounding execution**: The plan can be represented, but execution remains explicitly unavailable until an intermediate spatial artifact contract exists.
- **Benchmark and ISRO/SAC evidence**. The fresh B13 evaluation covers a sample from RSVQA-LR. CDVQA, optical-SAR, national coverage and ISRO/SAC require separate evidence. [Fresh-test results](results/vqa-mac-fresh-test/REPORT.md) · [Requirement gaps](docs/research/ps-requirement-gap-matrix.md)

See `docs/research/ps-requirement-gap-matrix.md` for requirement-level status and next evidence.

---

## Prerequisites

- **Python**: Declared version is **Python 3.11** (recorded in [`.python-version`](.python-version)). The system has also been verified under Python 3.14 on macOS arm64.
- **Node.js & npm**: Node.js 18+ (tested on Node v25.9.0 with npm 11+).
- **GPU acceleration**. The PyTorch Qwen and Grounding DINO providers use CUDA for reportable runs. Apple Silicon can run the pinned MLX B13 worker for `/ask`. [Mac runbook](docs/RUNBOOK.md) · [Provider smoke test](docs/gpu-smoke.md)

---

## Clone

```bash
git clone https://github.com/sohamsssssssssssssssss/sih2026.git
cd sih2026
```

---

## Backend Setup

### Minimal API Environment

For running the API service and CPU-safe contract tests:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

The backend requirements include Pillow, NumPy, and Rasterio because upload and pair-validation paths import them directly.

### Full Development / Test Environment

For running the complete test suite, evaluations, and scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r requirements.txt
```

> **macOS note**: Rasterio is required by TIFF ingestion and deterministic pair providers. Use a compatible wheel or environment rather than omitting it.

---

## Run the demo locally

On the integration Mac, use the existing Python 3.11 `.venv` and install the frontend with `npm ci --prefix frontend`. For a fresh backend venv, install the pinned Mac packages with `.venv/bin/python -m pip install -r requirements-lock.txt`. The existing demo examples use the WorldCover pack at `~/ben-work/wc/pack`, the Sen1Floods11 pack at `~/satquery-data/flood/packs/sen1floods11-hand-v1`, and the local BEN checkpoint paths in [land-cover configuration](configs/landcover.json). `SATQUERY_WORLDCOVER_DIR` and `SATQUERY_FLOOD_DIR` override pack paths. Punjab and Mumbai optical GeoTIFFs are shipped in [demo fixtures](tests/fixtures/README.md).

From the repository root, start both servers with one command:

```bash
bash scripts/run-demo.sh
```

Open [Ask](http://127.0.0.1:3000/ask). Select **Find a scene by place and date**, enter a place name or decimal latitude and longitude separated by a comma, and choose a sensor. The optional end date makes the UTC date window inclusive. You can also run a local demo example or upload a GeoTIFF.

Sentinel-2 selects the least-cloudy matching Earth Search item. Its scene-level cloud cover is the catalog percentage for the full Sentinel-2 scene. Sentinel-1 selects the latest matching dual-polarisation RTC item from Microsoft Planetary Computer. Each request reads a 120 × 120 window per band. Live requests use public services and require no API key.

If a provider is unavailable, Ask identifies and analyses a separate cached scene for the selected sensor. The [backup manifest](data/fresh-backups/manifest.json) lists the five chips, bounding boxes in WGS84 longitude/latitude, acquisition dates, source item IDs, and SHA-256 checksums. Live acquisition has a 20-second overall deadline including STAC and all bands. Override it in seconds with `SATQUERY_LIVE_FETCH_TIMEOUT`. On expiry, Ask shows a cached scene for the requested place when available, with its acquisition date. Run `SATQUERY_OFFLINE=1 bash scripts/run-demo.sh` to verify the fallback without contacting providers. In offline mode, the place-name field accepts the matching backup names or decimal coordinates. A working provider with no matching scene yields a no-scene message.

The launcher builds the frontend and starts it in production mode. Use `bash scripts/run-demo.sh --dev` for hot reload. The API runs on port 8000. Stop both servers with Ctrl+C.

With Google Chrome installed, stop any running demo and run `npm --prefix frontend run test:e2e` to exercise the local examples, upload, and live STAC flow. Playwright starts a production demo and writes screenshots to [Prompt 17 screenshots](docs/screens/p17/). Run `SATQUERY_OFFLINE=1 npm --prefix frontend run test:e2e -- --grep "offline fallback"` to exercise the offline case.

---

## Frontend Setup

```bash
cd frontend
npm install
```

To run the frontend development server:

```bash
npm run dev
```

The frontend will start at `http://localhost:3000` (or `http://localhost:3001` if port 3000 is occupied).

---

## Run Backend

From the repository root:

```bash
backend/.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Or using the provided Makefile:

```bash
make api
```

- **Default Host**: `127.0.0.1` / `localhost`
- **Default Port**: `8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`

---

## Run Frontend

From the repository root:

```bash
cd frontend && npm run dev
```

Or using the provided Makefile:

```bash
make frontend
```

- **Default URL**: `http://localhost:3000` (or `http://localhost:3001`)
- **Main Workspace**: `http://localhost:3000/workspace`
- **Cinematic Landing & Story**: `http://localhost:3000/intro`
- **Execution Audit Log**: `http://localhost:3000/executions`
- **Resolution Lab**: `http://localhost:3000/resolution`
- **SAR Analysis**: `http://localhost:3000/sar`
- **System Telemetry**: `http://localhost:3000/system`

---

## Quick Health Check

Verify the backend service is responding and see what it can actually run:

```bash
curl -s http://localhost:8000/api/health
```

**Recorded Response** (Mac CPU, 18 Nov 2026, from [docs/plan/E2E_ACCEPTANCE.md](docs/plan/E2E_ACCEPTANCE.md)):

```json
{
  "status": "degraded",
  "mode": "offline-first",
  "checks": {
    "trace": { "ok": true, "detail": null },
    "capabilities": {
      "single_image_vqa": { "available": false, "reason_code": "CUDA_UNAVAILABLE" },
      "grounding": { "available": false, "reason_code": "CUDA_UNAVAILABLE" },
      "optical_sar": { "available": true, "reason_code": null }
    }
  },
  "disabled_by_decision": {
    "change-deterministic": "Change questions are not supported because the methods are unvalidated. T18 found that image differencing did not beat the majority baseline on CDVQA, while question text alone strongly predicted the labels."
  }
}
```

The check is cheap: it verifies the persisted `trace.jsonl` hash chain and asks each registered provider for readiness without loading weights or running inference.

| `status` | HTTP | Meaning |
|---|---|---|
| `ready` | 200 | Trace history is intact and every capability's provider is available. |
| `degraded` | 200 | Trace history is intact, but at least one capability is unavailable (normal on CPU-only machines where GPU models are missing); see `checks.capabilities[*].reason_code`. |
| `unavailable` | 503 | Trace history is unreadable or fails verification, so every `/api/analyze` request will fail; see `checks.trace.detail`. |

---

## Basic End-to-End Flow

### 1. Upload a Scene
Upload a local PNG or JPEG satellite image tile:

```bash
curl -X POST http://localhost:8000/api/scenes \
  -F "file=@data/ladder/0.3/loveda_Train_Rural_images_png_0_gsd0.3.png"
```

**Generated Scene ID Format**: `scene_<uuid4_hex>` (e.g., `scene_4f8c92a1...`).

**Expected Response (HTTP 201)**:

```json
{
  "scene_id": "scene_4f8c92a1...",
  "filename": "loveda_Train_Rural_images_png_0_gsd0.3.png",
  "format": "PNG",
  "width": 512,
  "height": 512,
  "sensor": null,
  "gsd": null,
  "location": null,
  "acquisition_date": null
}
```

*Note: Unobserved metadata fields remain explicitly `null` to prevent synthetic hallucinations.*

### 2. Plan a Request
Submit an analytical query to generate a deterministic execution plan:

```bash
curl -X POST http://localhost:8000/api/plan \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "loveda_LoveDA_images_png_0_gsd0.3",
    "question": "Is there a building in this image?",
    "sensor": "optical",
    "capability": null
  }'
```

**Expected Response (HTTP 200)**:

```json
{
  "planner_version": "0.1.0",
  "rule_id": "single_image_vqa_default",
  "requested_capability": null,
  "selected_capability": "single_image_vqa",
  "executable": true,
  "reason": "Single scene provided without change keywords; defaulted to single-image VQA.",
  "required_inputs": ["single_scene"],
  "missing_inputs": [],
  "provider_available": true,
  "provider": "Qwen/Qwen2.5-VL-3B-Instruct",
  "unavailable_reason": null,
  "execution_plan_version": "0.1.0",
  "steps": [
    {
      "step_id": "step_1",
      "capability": "single_image_vqa",
      "depends_on": [],
      "required_inputs": ["single_scene"],
      "provider_available": true,
      "provider": "Qwen/Qwen2.5-VL-3B-Instruct"
    }
  ],
  "unavailable_capabilities": []
}
```

### 3. Analyze
Execute single-image visual question answering. `execution_mode` defaults to `"live"`, which needs CUDA and local Qwen weights; without them the request fails closed with `503`. The example below explicitly requests the pinned cached result so it works on any machine:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "loveda_LoveDA_images_png_0_gsd0.3",
    "question": "Is there a building in this image?",
    "sensor": "optical",
    "execution_mode": "cached_result"
  }'
```

**Expected Response (HTTP 200)**:

```json
{
  "answer": "Yes.",
  "execution_mode": "cached_result",
  "results_artifact": "results/qwen2.5vl-3b__ladder__rescored__20260904.json",
  "model": {
    "name": "Qwen/Qwen2.5-VL-3B-Instruct",
    "version": "frozen-qwen2.5vl-3b-instruct"
  },
  "trace": {
    "record_hash": "...",
    "prev_hash": "...",
    "route": "single_image_vqa",
    "model_name": "qwen2.5vl-3b",
    "timestamp": "..."
  },
  "notice": "Cached replay requested; showing the exact committed result for this scene and question. No live inference ran."
}
```

### 4. Inspect Audit Traces
Retrieve chronological execution logs:

```bash
curl -s http://localhost:8000/api/traces
```

### 5. Verify Cryptographic Trace Chain
Verify SHA-256 integrity and linkage across the audit log:

```bash
curl -X POST http://localhost:8000/api/traces/verify
```

**Expected Response**:

```json
{
  "verified": true,
  "message": "Chain verified (N records)"
}
```

---

## Explicit cached replay

For a narrow offline demonstration, the repository includes one pinned result that must be requested explicitly with `execution_mode: "cached_result"`:

- **Pinned Scene ID**: `loveda_LoveDA_images_png_0_gsd0.3`
- **Pinned Question**: `Is there a building in this image?`
- **Capability**: `single_image_vqa`
- **Artifact Source**: [`results/qwen2.5vl-3b__ladder__rescored__20260904.json`](results/qwen2.5vl-3b__ladder__rescored__20260904.json)

### Exact Fallback Invariants
1. **Intentionally Narrow**: Replay succeeds *only* when the scene ID, question, and capability match the exact pinned values and cached mode was explicitly requested.
2. **Provenance Disclosure**: Responses explicitly return `execution_mode: "cached_result"` and state the source artifact path.
3. **Not a Generic Cache**: Dynamic queries, arbitrary questions, and uploaded user scenes are **never** served from cache.
4. **No unrelated replay**: Grounding, change, and optical-SAR requests never use the cached VQA artifact.

---

## GPU / CUDA Behavior

- **Live model inference**: Reportable Qwen and Grounding DINO runs require PyTorch with CUDA. Qwen also has an explicit MPS development mode whose outputs are not CUDA baseline measurements.
- **Fail-closed design**: Live Qwen or Grounding DINO requests fail with `503 Service Unavailable` when CUDA, dependencies, configuration, or local artifacts are unavailable. Cached replay is never selected automatically.
- **Engineering Rule**: Do **not** modify failure handling to silently generate mock answers or synthetic confidence values on CPU. Factual failure is a safety invariant.

---

## Grounding status

- **Provider**: `grounding-dino-swint`, using the official Swin-T OGC configuration and local checkpoint contract.
- **Routing**: Spatial localization queries select `grounding` and execute only when CUDA, dependencies, configuration, and checkpoint are ready.
- **Evidence**: Normalized `xyxy` boxes, labels, and finite model detection scores. Empty detections remain explicit.
- **Verification boundary**: Historical Kaggle T4 smoke passed. The DIOR-RSVG runner exists, but no new benchmark result was produced during the canonical migration.

---

## Run Tests

### Run Backend API Tests

```bash
python3 -m pytest backend/test_api.py -q
```

### Run Full Python Test Suite

```bash
python3 -m pytest -q
```

*The full run recorded on 20 Nov 2026 passed 1,401 tests with 47 skipped and 0 failed. [Integration log](docs/plan/INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture)*

---

## Frontend Production Build

Validate TypeScript compilation, Tailwind CSS styling, and Next.js static asset optimization:

```bash
cd frontend && npm run build
```

No build output is recorded in the repository; see the latest build run.

---

## Deploy

To run the full stack on one NVIDIA GPU VM behind one public URL (Docker Compose with a Caddy reverse proxy and automatic HTTPS), run `make docker-gpu` and then `make deploy-smoke`. A CPU-only variant is `make docker-cpu`. Before you start, read [`docs/deploy.md`](docs/deploy.md). It covers VM sizing, offline provisioning of model artifacts, and the demo-day checklist, and it lists which parts have not yet been tested on a GPU. Stop the stack with `make docker-down`.

---

## Runtime Data & Hygiene

- **Uploaded Scenes**: Stored in `data/runtime/scenes/<scene_id>.png`.
- **Audit Traces**: Appended to [`trace.jsonl`](trace.jsonl) in the repository root.
- **Git Ignore Policy**: Both `data/runtime/` and `trace.jsonl` are strictly ignored in [`.gitignore`](.gitignore).
- **Hygiene Rule**: Never commit runtime-generated scenes, temporary files, or local execution traces to git.

---

## Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **Backend returns 503 on `/api/analyze`** | Running without CUDA hardware or local model weights. | Expected behavior. Test using the golden demo query or deploy on a CUDA-enabled GPU. |
| **Provider unavailable (503)** | Required CUDA, dependency, configuration, or local model artifact is absent; or the requested multi-step plan is not executable. | Inspect `/api/capabilities`, provision artifacts offline, or use a supported deterministic single-step capability. |
| **Upload rejected (422 / 413)** | The upload is corrupt, unsupported, unsafe, or exceeds 20 MiB. | Provide a valid PNG, JPEG, TIFF, or GeoTIFF within the documented limits. |
| **Frontend cannot connect to backend** | Backend server is stopped or running on a different port. | Ensure backend is active at `http://localhost:8000`. Check with `curl http://localhost:8000/api/health`. |
| **Trace integrity error (503)** | `trace.jsonl` was edited or corrupted somewhere other than its final line. | An incomplete final line from a crash is recovered automatically and its bytes are kept in `trace.jsonl.torn-<timestamp>`. Anything else is treated as tampering and fails closed: check `/api/health` and `POST /api/traces/verify`, inspect the file, and keep a copy as evidence before rotating it. |
| **`ModuleNotFoundError: No module named 'PIL'`** | Minimal backend venv created without Pillow. | Run `pip install pillow` inside your backend virtual environment. |
| **`rasterio` build error on macOS** | Missing system GDAL C-libraries. | Install a prebuilt wheel (`pip install --only-binary=:all: rasterio`) or GDAL via Homebrew. Rasterio cannot be omitted: the backend imports it at startup. |

---

## Safety & Engineering Invariants

1. **Unknown Metadata Remains Null**: Ground-sample distance (`gsd`), sensor, location, and acquisition timestamps are never guessed. If not explicitly extracted, they remain `null`.
2. **No Fabricated Confidence**: Confidence scores reflect measured model probabilities or are omitted; synthetic mock confidences are prohibited.
3. **No Hallucinated Detections**: The system will never return mock bounding box coordinates.
4. **Codebase Review**: All changes to orchestration, model contracts, or schemas require pull request review.
5. **Runtime Cleanliness**: Runtime artifacts must remain strictly untracked.

---

## Development Status & Scientific Baselines

- **Stage 0 Baseline**: Measured on official RSVQA-LR test split (10,004 questions) using frozen `Qwen/Qwen2.5-VL-3B-Instruct`:
  - **Open Accuracy**: `0.1651` *(Headline metric; uninflated by binary yes-rate collapse)*
  - **Binary Accuracy**: `0.6671`
  - **Aggregate Accuracy**: `0.5142`
  - **Binary Prediction Yes-Rate**: `0.3675` *(Passed two-sided degeneracy guard [0.15, 0.85])*
- **Resolution ladder v2**. On 200 LoveDA tiles, building-presence accuracy was **0.790 at 0.3 m**. The yes-rate reached **0.970 at 5 m** and **1.000 at 10 m**, where accuracy equalled the always-yes baseline of **0.700**. Dominant-class accuracy was below its majority baseline at every rung. Ladder v3 is preregistered and awaits a run. [V2 analysis](results/ladder-v2/ANALYSIS.md) · [V3 gate](docs/gates/LADDER-V3.md)
- **Post-Phase-0 progress**. Grounding, pair validation, deterministic optical-SAR, offline readiness, GPU smoke infrastructure, land cover, flood Otsu and the `/ask` flow are implemented. The Mac demo serves the B13 adapter. Change QA is disabled. [Current status](docs/STATUS.md)

### Next Phase

Current open work is listed in [docs/STATUS.md](docs/STATUS.md). The India land-cover promotion is closed. [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)
