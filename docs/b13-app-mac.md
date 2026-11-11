# B13 adapter on the Mac

Worktree: `/Users/soham/sih2026-app-b13`, branch `app/b13-adapter`, based on `0fc4c40`.
The concurrent training checkout was read-only: reference JSON/images and a check for legacy generated test fixtures. No files there were changed.
Training code was inspected with `git show training/rgb-pilot-b12-b13:...`.

## Existing VQA path

- `models/artifacts.py` resolves local artifacts only; `configs/model_artifacts.json` pins the Qwen revision and both weight hashes, with `SATQUERY_QWEN_MODEL_DIR` as the override.
- `models/qwen_vl/model.py` lazily loads Qwen and its processor. The existing base path prefers CUDA; MPS requires `SATQUERY_ALLOW_MPS=1`; CPU is refused. It appends the short-answer instruction, greedily generates up to 50 tokens, decodes only generated tokens, and strips whitespace. No semantic answer parser is applied.
- `orchestrator/registry.py` constructs the provider; `orchestrator/planner.py` selects its capability; `orchestrator/executor.py` calls `orchestrator/router.py`.
- `orchestrator/worker.py` owns a warm spawned inference process, serializes requests, and kills it on timeout. The API currently allows 120 seconds, including cold loading.
- `orchestrator/router.py` validates output and appends the hash-chained record through `orchestrator/trace.py`.
- `backend/services.py` handles exact cached replay separately from live execution (`find_cached_result`, `_cached_response`, `_live_response`). A live failure never silently replays a base-model answer. Explicit `cached_result` requests still replay the original committed result, even with B13 enabled, and retain cached provenance.
- `backend/schemas.py` describes the response. The Next frontend consumes `/api/analyze`; no frontend changes are necessary to request live inference. Provenance is available in the JSON and trace.

## Implemented behavior

`SATQUERY_VQA_ADAPTER` is unset by default. Set it **before starting the API**; restart the API/worker when changing it. Flag off preserves the base inference contract and device gate.

Flag on verifies the complete B13 adapter weight AND configuration SHA256, verifies the pinned base revision and both base shard hashes using the existing `scripts/stage1_runtime.py::verify_checkpoint`, loads fp16 without bitsandbytes, and attaches frozen PEFT LoRA. MPS still requires the existing opt-in; otherwise the adapter may run on CPU. MPS placement or generation RuntimeErrors retry on CPU and record the fallback. CUDA remains supported.

`models/qwen_vl/b13.py` copies the contract reached from `scripts/rgb_pilot_package.py` through `scripts/evaluate_remote_sensing_adapter.py::_answer` and `training/remote_sensing.py::messages` on the training branch: image first, original question plus `Answer with a single word or number only. No explanation.`, explicit 392×392 resize, normal Qwen chat template, greedy decoding, 32 new tokens, skip special tokens, no tokenization cleanup, strip output whitespace. Agreement/accuracy use stripped case-insensitive exact match, as the evaluation policy specifies; generated answers retain their original case.

The response and trace params carry `model_provenance` (B13, full hashes, base revision, actual device, precision, unquantized execution, CPU fallback) and `answer_source=b13_adapter`. Metadata travels back from the worker, rather than being guessed from its parent.

English count templates (`How many …`, `What is the number of …`, `Count …`) select Grounding DINO. The returned answer is the number of detection boxes, with numeric `count`, original box evidence, and `answer_source=grounding_dino_count` in response and trace. Explicit single-image VQA requests receive the same count routing. No adapter fallback is allowed for counts. The existing Grounding DINO provider requires CUDA and local artifacts, so counts are unavailable on this Mac. Detection counts are threshold-dependent object detections, not validated dense-building counts. Unavailable execution does not append a successful-answer trace.

## Verified adapter

Archive: `~/Downloads/b13-final-750.zip`

- ZIP SHA256: `a719340e3e504d4c047383d6ffb3d2f516ad5053c5d40810eb26254ff424b513`
- Extracted only `b13-resume-112-b/adapter/` to `~/models/satquery-b13-adapter/`, outside Git.
- Weight SHA256: `7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a`
- Config SHA256: `716251f9cfbb73ed5fb408e41f2eb2b30281b8f98fb7c6a92a1dd0b9204e3644`

## Run

An isolated `.venv` with system site packages was created in this worktree. PEFT is installed there, without changing the training environment. For a fresh environment on this Mac:

```bash
cd /Users/soham/sih2026-app-b13
python3.11 -m venv --system-site-packages .venv
.venv/bin/pip install -r backend/requirements.txt 'transformers==4.49.0' 'peft==0.15.2' 'qwen-vl-utils==0.0.8' torch torchvision accelerate huggingface-hub
export SATQUERY_VQA_ADAPTER="$HOME/models/satquery-b13-adapter"
export SATQUERY_ALLOW_MPS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# REQUIRED: provision the pinned checkpoint separately; this is a placeholder.
export SATQUERY_QWEN_MODEL_DIR=/absolute/path/to/verified/qwen2.5-vl-3b-instruct
.venv/bin/python -m uvicorn backend.main:app --port 8000
```

The base must be revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, with the shard hashes in `configs/model_artifacts.json`. Revision proof must be present as `config.json`'s `_commit_hash` or `.cache/huggingface/download/config.json.metadata`, per the existing verification helper. The adapter processor/tokenizer files are deliberately not substituted for the pinned base processor.

In a second terminal:

```bash
cd /Users/soham/sih2026-app-b13/frontend
npm ci
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Use live mode and a single RGB image. For CPU-only adapter execution, set `SATQUERY_ALLOW_MPS=0` before API startup. To restore the original base path, `unset SATQUERY_VQA_ADAPTER` and restart.

## Reproduce parity

```bash
cd /Users/soham/sih2026-app-b13
.venv/bin/python scripts/check_b13_app_parity.py \
  --reference /Users/soham/sih2026/results/rgb-pilot-b13/rgb-b13-evaluation-01/result.json \
  --images /Users/soham/sih2026/packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/images \
  --adapter /Users/soham/models/satquery-b13-adapter \
  --out /tmp/b13-mps-parity.json
```

Add `--cpu` and choose a different fresh output file for CPU. The runner verifies the frozen identity, adapter identity, and selected image hashes; selects the first ten records of each type in the reference's fixed order; and uses the app's model loader, router, result validation, and hash-chained trace. It runs in process to measure model RSS, avoiding child-process RSS omissions. It does not exercise HTTP upload or worker IPC; unit/integration tests cover the provenance boundary.

The runner reports load time separately, warm seconds/question, peak process RSS, and sampled MPS driver memory. The MPS sample is not a true kernel allocation peak. Exit code 2 means incomplete prerequisites/execution. Count answers are detector outputs, so their agreement with Kaggle adapter counts is a routing comparison, not numerical adapter parity.

## Local parity result and limitations

No local base was found: the manifest default is Kaggle-only, `SATQUERY_QWEN_MODEL_DIR` was unset, no Qwen snapshot exists in the local Hugging Face cache, and a home-directory search found no Qwen weight shards/index. No base download was performed.

The reference contains all 389 per-question predictions. The frozen 40-question selection is 10 each of comp, count, presence, rural_urban. Local completed predictions: **0/40**. Load time, inference peak memory, seconds/question, local accuracy, agreement, and numerical drift are **not measured**. Process RSS from a failed preflight is not model memory.

| Type | Selected | Kaggle adapter correct on selection | Local completed | Local agreement / accuracy |
|---|---:|---:|---:|---|
| comp | 10 | 10/10 | 0 | unavailable |
| count | 10 | 0/10 | 0 | unavailable; CUDA detector required |
| presence | 10 | 8/10 | 0 | unavailable |
| rural_urban | 10 | 9/10 | 0 | unavailable |

These are reference-only subset numbers, not a new validation estimate. The full 389-question training result is not remeasured here. fp16 versus Kaggle 4-bit drift remains unknown. Real 16-GB unified-memory feasibility and the 120-second cold-request timeout also remain unverified until the checkpoint is provisioned.

## Files changed

- `models/qwen_vl/model.py`, `models/qwen_vl/b13.py`: opt-in adapter loading, pinned identity, evaluation contract, CPU fallback.
- `orchestrator/planner.py`, `orchestrator/router.py`: detection-count routing and source/provenance tracing.
- `backend/services.py`, `backend/schemas.py`: expose provenance, source, numeric counts.
- `requirements.txt`: PEFT dependency.
- `models/qwen_vl/test_b13.py`, `scripts/test_check_b13_app_parity.py`: regression and parity-selection tests.
- `scripts/check_b13_app_parity.py`: offline parity runner.
- `results/b13-app-mac-{cpu,mps}-parity.json`: separate incomplete CPU/MPS prerequisite reports.
- `docs/b13-app-mac.md`: inspection, setup, evidence, limitations.

The adapter files and isolated virtual environment are outside the commit. No changes were made to the concurrent training checkout or branch, and nothing was pushed or merged.

## Validation

- `.venv/bin/python -m pytest backend orchestrator models scripts/test_check_b13_app_parity.py -q`: **675 passed**, 12 subtests passed. Includes 14 new adapter tests plus the new parity-selection/scoring test.
- `.venv/bin/python -m pytest -q`: **1014 passed, 5 failed, 40 setup errors**, 28 subtests passed. All remaining failures/errors are missing `packages/stage1-t4-smoke` generated fixtures (configs, packaged runner, and manifest). That directory is untracked/absent from main. A read-only check of the training checkout's legacy package found that its runner source differs from main; it was not copied or imported. The full-suite requirement therefore remains unmet.
- Real PEFT 0.15.2 loaded the extracted LoRA configuration with rank 16, alpha 32, and all four attention targets.
- ZIP, complete adapter weight/config digests, and all 40 selected image hashes matched.
- CPU and MPS parity preflights both exited 2 with the base checkpoint unavailable. No inference measurements or drift claims were made.
- `git diff --check` passed. Frontend tests/build were not run; frontend files were not changed.

The local isolated environment uses torch 2.10.0, transformers 4.49.0, PEFT 0.15.2, and qwen-vl-utils 0.0.8. Compared with the reference, PyTorch and CPU/MPS kernels differ; actual numerical drift must still be measured.
