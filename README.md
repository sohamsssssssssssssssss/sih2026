# SatQuery AI

## What SatQuery Is

> "Ask your satellite data anything."

SatQuery AI is a capability-oriented remote-sensing visual intelligence system built for SIH26167. It translates natural-language queries into structured, evidence-backed geospatial reasoning over Earth observation imagery, enforcing cryptographic auditability, resolution awareness, and fail-closed safety constraints.

---

## Current Status

### AVAILABLE
- **Controlled Image Ingestion**: Upload validation (`POST /api/scenes`) supporting PNG and JPEG up to 20 MiB with decompression-bomb guards, format normalization, and path-traversal prevention.
- **Deterministic Planner**: Rule-based request planning (`POST /api/plan`) that analyzes input questions, scene requirements, and sensor intent to select appropriate capabilities without LLM non-determinism.
- **Structured Execution Plan**: Generation of explicit, multi-step dependency graphs (`execution_plan_version: 0.1.0`) separating plan representation from execution.
- **Single-Image VQA Path**: Complete visual-question-answering execution pipeline for optical imagery via `Qwen/Qwen2.5-VL-3B-Instruct`.
- **Cryptographic Audit Trace**: SHA-256 hash-chained, append-only execution log (`trace.jsonl`, `GET /api/traces`, `POST /api/traces/verify`) guaranteeing verifiable provenance for every routed decision.
- **Golden Cached Fallback**: Exact, verified committed fallback (`results/qwen2.5vl-3b__ladder__rescored__20260904.json`) for the designated benchmark query when running offline or without GPU acceleration.
- **Frontend / Backend Integration**: Full-stack integration between the Next.js 16 workspace UI and the FastAPI backend service.

### UNAVAILABLE / IN DEVELOPMENT
- **Grounding Provider**: Capability vocabulary and planning rules (`grounding`, `grounding_spatial_localization`) are defined, but the Grounding DINO provider is **not yet registered** in the active codebase. Grounding requests fail closed with `503 Service Unavailable`.
- **Bi-Temporal Change-VQA (`change_vqa`)**: Planned multi-scene change detection chain is represented in the planner, but no change detection execution provider is currently registered.
- **Optical–SAR Fusion (`optical_sar`)**: SAR false-color interpretation reference materials are available (`/sar`), but automated multimodal fusion models are not yet deployed.
- **Remote Sensing Fine-Tuning**: Currently running frozen foundation checkpoints; domain-adapted weights are in development.

*Unavailable capabilities intentionally fail closed and never silently fall back to unrelated models or fabricated outputs.*

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

For running the API service and offline verification:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
pip install pillow
```

> **Important**: `backend/services.py` requires `PIL` (Pillow) for safe image verification, format conversion, and dimensions extraction. Because `pillow` is defined in root `requirements.txt` rather than `backend/requirements.txt`, install `pillow` explicitly when setting up a minimal environment.

### Full Development / Test Environment

For running the complete test suite, evaluations, and scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r requirements.txt
```

> **macOS Note**: If `rasterio` fails to install due to system GDAL bindings, omit it for local API and frontend development. Core VQA, ingestion, planning, and evaluation paths do not require `rasterio`.

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

Verify the backend service is responding:

```bash
curl -s http://localhost:8000/api/health
```

**Expected Response**:

```json
{
  "status": "ready",
  "mode": "offline-first"
}
```

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
Execute single-image visual question answering:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "scene_id": "loveda_LoveDA_images_png_0_gsd0.3",
    "question": "Is there a building in this image?",
    "sensor": "optical"
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
  "notice": "Offline demonstration: showing the exact committed result for this pinned query."
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

## Golden Offline / Cached Demo

To enable zero-dependency evaluations and demonstrations without local GPU infrastructure, the repository includes a pinned golden path:

- **Pinned Scene ID**: `loveda_LoveDA_images_png_0_gsd0.3`
- **Pinned Question**: `Is there a building in this image?`
- **Capability**: `single_image_vqa`
- **Artifact Source**: [`results/qwen2.5vl-3b__ladder__rescored__20260904.json`](results/qwen2.5vl-3b__ladder__rescored__20260904.json)

### Exact Fallback Invariants
1. **Intentionally Narrow**: Fallback activates *only* when the scene ID, question, and capability match the exact pinned values.
2. **Provenance Disclosure**: Responses explicitly return `execution_mode: "cached_result"` and state the source artifact path.
3. **Not a Generic Cache**: Dynamic queries, arbitrary questions, and uploaded user scenes are **never** served from cache.
4. **No Fallback for In-Development Capabilities**: Grounding, change detection, and multimodal fusion queries never fall through to the golden VQA answer.

---

## GPU / CUDA Behavior

- **Live Model Inference**: Real-time forward passes with `Qwen2.5-VL-3B` require PyTorch with CUDA acceleration (`torch.cuda.is_available() == True`).
- **Fail-Closed Design**: When run in an environment without CUDA or without downloaded model weights, live requests on non-golden queries fail closed with `503 Service Unavailable` (`"Live model inference is unavailable."`).
- **Engineering Rule**: Do **not** modify failure handling to silently generate mock answers or synthetic confidence values on CPU. Factual failure is a safety invariant.

---

## Grounding Status

- **Capability Identifier**: `grounding`
- **Planning Rule**: Queries containing spatial localization markers (*"where"*, *"locate"*, *"bounding box"*) are mapped by the planner to `selected_capability: "grounding"`, producing a plan with `rule_id: "grounding_spatial_localization"`.
- **Current Execution Status**: **Unavailable**. No grounding model provider (e.g. Grounding DINO) is currently registered in `orchestrator/capabilities.py`.
- **Honest Failure**: Attempting to execute grounding returns `503 Service Unavailable` (`"Required capability is not currently available."`).
- **No Hallucinations**: Neither the backend nor the frontend fabricates bounding boxes when a grounding model is absent.
- **Evaluation**: Zero-shot DIOR-RSVG evaluation pipelines are in development and will be committed alongside model weights.

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

*Verified status in current environment: all unit, contract, and route tests pass cleanly (76 passed on API suite, 254 passed across full test harness).*

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
| **Capability unavailable (503)** | Requesting `grounding`, `change_vqa`, or `optical_sar`. | These capabilities are currently in development and fail closed. |
| **Upload rejected (422 / 413)** | Uploaded file is corrupt, not a PNG/JPEG, or exceeds 20 MiB. | Provide a standard PNG or JPEG image tile under 20 MiB. |
| **Frontend cannot connect to backend** | Backend server is stopped or running on a different port. | Ensure backend is active at `http://localhost:8000`. Check with `curl http://localhost:8000/api/health`. |
| **Trace integrity error (503)** | `trace.jsonl` has been manually edited or corrupted. | The hash chain verifies previous record hashes. Remove `trace.jsonl` to reinitialize a clean audit chain. |
| **`ModuleNotFoundError: No module named 'PIL'`** | Minimal backend venv created without Pillow. | Run `pip install pillow` inside your backend virtual environment. |
| **`rasterio` build error on macOS** | Missing system GDAL C-libraries. | Omit `rasterio` for local API/frontend development; core pathways do not depend on it. |

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
- **Phase 1 Progress**: Deterministic planner, structured multi-step execution plans, and controlled scene ingestion complete. Grounding DINO integration and threshold evaluation underway.
