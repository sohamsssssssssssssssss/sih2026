# SatQuery AI

## What SatQuery Is

> "Ask your satellite data anything."

SatQuery AI is a capability-oriented remote-sensing visual intelligence system built for SIH26167. It translates natural-language queries into structured, evidence-backed geospatial reasoning over Earth observation imagery, enforcing cryptographic auditability, resolution awareness, and fail-closed safety constraints.

---

## Current Status

### AVAILABLE
- **Controlled image ingestion**: `POST /api/scenes` validates PNG, JPEG, TIFF, and GeoTIFF uploads up to 20 MiB. Raster metadata is observed separately from user declarations.
- **Deterministic orchestration**: `POST /api/plan` selects capabilities, checks input shape and provider readiness, and emits auditable execution steps. Unsupported multi-step plans fail closed.
- **Single-image VQA**: `Qwen/Qwen2.5-VL-3B-Instruct` executes from offline local artifacts on CUDA, with opt-in MPS for local iteration. Confidence is `null` when uncalibrated.
- **Grounding**: Grounding DINO Swin-T returns normalized bounding-box evidence with fixed thresholds (`box=0.35`, `text=0.25`).
- **Pair compatibility**: Optical-SAR and bi-temporal requests are checked for modality, acquisition metadata, overlap, dimensions, CRS, and affine grid before dispatch.
- **Deterministic optical-SAR analysis**: Joint Sentinel-2 index and Sentinel-1 backscatter summaries consume both co-registered inputs. This is not a learned fusion model.
- **Deterministic change analysis**: Co-registered multispectral or RGB pairs produce heuristic magnitude, changed-pixel, coverage, and spatial-extent evidence. This is not semantic change understanding.
- **Sentinel-1 water change**: Two co-registered VV/VH/dataMask SAR scenes sent to `change_vqa` report new and receded open water in hectares and as GeoJSON polygons, from a split-based Otsu baseline (bimodal tiles, after Chini et al. 2017) that abstains when no part of the scene shows an open-water mode, with a ±1 dB sensitivity range. Where village boundaries exist (the Bihar and Kerala catalogue events), the answer names the most flooded villages with hectares. Confidence is `null`. See `docs/change-baseline.md`.
- **Cryptographic audit trace**: Live and cached-result executions are distinguished in a SHA-256 hash-chained trace.
- **Explicit cached replay**: The designated committed VQA result is returned only when the caller requests `execution_mode: "cached_result"`; live failure never silently falls back.

Historical Kaggle Tesla T4 smoke verification passed through the repository provider path for Qwen and Grounding DINO. See `docs/gpu-smoke.md`. This proves runtime execution, not model accuracy or throughput.

### PARTIAL / BLOCKED ON GPU OR DATA
- **Remote-sensing adaptation**: The offline RSVQA-LR QLoRA pipeline and held-out comparison runner are ready. No adapter has been trained and no improvement is claimed.
- **Sentinel-1 fetch from CDSE**: `python -m backend.sentinel1 --bbox W S E N --before START END --after START END` (needs `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`) picks a same-orbit-track pair, renders terrain-corrected gamma0 for both dates on one UTM grid and ingests them as scenes. `--event kosi-2024` (or `silchar-2022`, `kerala-2018-periyar`) uses a sourced AOI and date windows from `data/manifests/flood_events.v1.json`. Tested only against a mocked Sentinel Hub; the first live run is pending credentials.
- **Multi-step change-to-grounding execution**: The plan can be represented, but execution remains explicitly unavailable until an intermediate spatial artifact contract exists.
- **Benchmark and ISRO/SAC evidence**: Existing frozen evaluations and runners do not establish adapted-model, CDVQA, optical-SAR, national, or ISRO/SAC performance.

See `docs/research/ps-requirement-gap-matrix.md` for requirement-level status and next evidence.

---

## Prerequisites

- **Python**: Declared version is **Python 3.11** (recorded in [`.python-version`](.python-version)). The system has also been verified under Python 3.14 on macOS arm64.
- **Node.js & npm**: Node.js 18+ (tested on Node v25.9.0 with npm 11+).
- **CUDA / GPU Acceleration**: An NVIDIA GPU with CUDA support is required for live `Qwen2.5-VL` model inference. On systems without CUDA (e.g., local macOS or CPU-only Linux), the system operates in offline-first mode, serving the verified golden query or failing closed with `503`.

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

**Expected Response** (CPU-only machine without the GPU model stack):

```json
{
  "status": "degraded",
  "mode": "offline-first",
  "checks": {
    "trace": { "ok": true, "detail": null },
    "capabilities": {
      "single_image_vqa": { "available": false, "reason_code": "DEPENDENCY_UNAVAILABLE" },
      "grounding": { "available": false, "reason_code": "DEPENDENCY_UNAVAILABLE" },
      "change_vqa": { "available": true, "reason_code": null },
      "optical_sar": { "available": true, "reason_code": null }
    }
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

*Run the command above on the reviewed commit for the current authoritative count; migration checkpoint counts are recorded in commit handoff notes.*

---

## Frontend Production Build

Validate TypeScript compilation, Tailwind CSS styling, and Next.js static asset optimization:

```bash
cd frontend && npm run build
```

**Expected Output**:
```
✓ Compiled successfully
✓ Generating static pages (8/8)
Finalizing page optimization ...
```

---

## Deploy

To run the full stack on one NVIDIA GPU VM behind one public URL (Docker Compose with a Caddy reverse proxy and automatic HTTPS), run `make docker-gpu` and then `make deploy-smoke`. A CPU-only variant is `make docker-cpu`. Before you start, read [`docs/deploy.md`](docs/deploy.md). It covers VM sizing, offline provisioning of model artifacts, and the demo-day checklist, and it lists which parts have not yet been tested on a GPU.

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
- **Resolution Robustness Ladder**: Evaluated on 200 real LoveDA scenes degraded across 5 rungs (0.3m, 1m, 2m, 5m, 10m). Rungs at 5m and 10m exhibit collapse to the gold yes-prior and are flagged as degenerate.
- **Post-Phase-0 progress**: Grounding, pair validation, deterministic optical-SAR and change providers, offline readiness, GPU smoke infrastructure, and the RSVQA-LR adaptation pipeline are implemented. Learned adaptation and the remaining held-out evaluations have not run.

## Development Status

The next execution phase is RSVQA-LR provisioning, a bounded QLoRA dry run/training job, base-versus-adapter validation, and one real bi-temporal smoke. No adapted model exists yet.
