# T05 integration log — 2026-11-17

Starting branch: integration/spine-v1. Starting origin/main: `c429b6584ed3b6528dffccf550c17a393997b8ba`.
Remote fetched before creation. Nothing is merged to main. Only this worktree is
used for writes. No force-push, branch deletion, history rewrite, VLM training
code edit, annotation edit, or Kaggle action is authorized.

## Pre-merge inventory

Counts below are branch commits ahead/behind origin/main (symmetric history),
not counts of changed files. Files are changes since the merge base with main.

### task/T04-plan-files

Head: `9bde585c3cb74cb687c41661e289cb4b6d437090`. Ahead: **0**. Behind: **6**.

No changed files; already an ancestor of origin/main.

### task/T09-ask-api

Head: `a45d1f948b70042f8bb1edc95edb2639a643c0f2`. Ahead: **10**. Behind: **7**.

```text
A	backend/ask_scenes.py
A	backend/ask_service.py
A	backend/config.py
M	backend/main.py
M	backend/requirements.txt
A	backend/routes/ask.py
M	backend/services.py
A	backend/test_ask.py
A	ben/landcover.py
A	ben/transfer.py
A	configs/answerability.json
A	configs/landcover.json
A	core/answer.py
A	core/canonical.py
A	core/gate.py
A	core/registry.py
A	core/router.py
M	core/schema.py
M	core/schema_version.py
A	core/test_canonical.py
A	core/test_registry.py
A	core/test_router.py
M	core/test_schema.py
A	core/tool_status.json
A	core/tools/__init__.py
A	core/tools/base.py
A	core/tools/change.py
A	core/tools/flood_otsu.py
A	core/tools/grounding_dino.py
A	core/tools/landcover.py
A	core/tools/optical_sar.py
A	core/tools/qwen_vl.py
A	core/tools/test_adapters.py
A	core/tools/test_flood_otsu.py
A	core/tools/test_landcover.py
A	docs/hypotheses/T07c.md
A	docs/plan/ROUTER_REPORT.md
A	docs/plan/ROUTER_SPEC.md
A	docs/plan/TOOLS_INVENTORY.md
M	docs/schemas/Answer.json
M	docs/schemas/EvalRecord.json
M	docs/schemas/Evidence.json
M	docs/schemas/PackManifest.json
M	docs/schemas/Patch.json
M	docs/schemas/Query.json
A	docs/schemas/README.md
M	docs/schemas/Scene.json
M	docs/schemas/ToolResult.json
M	docs/schemas/ToolStatus.json
M	orchestrator/trace.py
A	outputs/T07c-flood.json
A	outputs/T07c-flood.md
A	outputs/T07c-landcover.json
A	outputs/T07c-landcover.md
A	outputs/T07c-tools.md
A	outputs/T07c-validation.json
A	requirements-test.txt
M	requirements.txt
A	results/t08-router-validation.json
A	scripts/evaluate_t07c.py
```

### task/T14-flood-pack

Head: `df4a2363a875ce8e2905b1167cd72b1567528a60`. Ahead: **0**. Behind: **6**.

No changed files; already an ancestor of origin/main.

### task/T13-india-adapt

Head: `a0d7002cb9a43162e297f98a795694982791acb5`. Ahead: **21**. Behind: **8**.

```text
M	backend/schemas.py
M	backend/services.py
A	backend/test_count_grounding.py
A	backend/test_count_resolution.py
A	ben/adapt.py
A	ben/landcover.py
A	ben/run_adapt_gpu.txt
A	ben/run_adapt_seeds_gpu.txt
A	ben/run_landcover_gpu.txt
A	ben/run_transfer_gpu.txt
A	ben/test_adapt.py
A	ben/test_landcover.py
A	ben/test_transfer.py
A	ben/transfer.py
A	configs/count_grounding.json
A	docs/b13-app-mac.md
A	docs/ben-landcover.md
A	docs/count-grounding.md
A	docs/hypotheses/T13.md
A	docs/hypotheses/T13b.md
A	kaggle/eval_count_detection.py
A	kaggle/fetch_dior_cpu.sh
A	kaggle/run_count_dior_gpu.txt
M	kaggle/run_grounding_dior_rsvg.py
A	kaggle/test_eval_count_detection.py
A	models/grounding_dino/count.py
M	models/grounding_dino/model.py
A	models/grounding_dino/resolution.py
A	models/grounding_dino/test_count.py
A	models/grounding_dino/test_resolution.py
A	models/qwen_vl/b13.py
M	models/qwen_vl/model.py
A	models/qwen_vl/test_b13.py
M	orchestrator/planner.py
M	orchestrator/router.py
M	requirements.txt
A	results/b13-app-mac-cpu-parity.json
A	results/b13-app-mac-mps-parity.json
A	results/b13-app-mac-pinned-mps-final-parity.json
A	results/b13-app-mac-pinned-mps-final-parity.trace.jsonl
A	results/b13-app-mac-pinned-mps-measurements.json
A	results/ben-landcover/ANALYSIS.md
A	results/ben-landcover/SOURCE.md
A	results/ben-landcover/analysis.json
A	results/ben-landcover/analyze_ben_landcover.py
A	results/ben-landcover/ben-landcover-compare-s1s2-vs-s2.json
A	results/ben-landcover/ben-landcover-compare-s2-vs-s1.json
A	results/ben-landcover/ben-landcover-progress.txt
A	results/ben-landcover/ben-landcover-s1-test-idx.npy
A	results/ben-landcover/ben-landcover-s1-test-labels.npy
A	results/ben-landcover/ben-landcover-s1-test-logits.npy
A	results/ben-landcover/ben-landcover-s1.json
A	results/ben-landcover/ben-landcover-s1s2-test-idx.npy
A	results/ben-landcover/ben-landcover-s1s2-test-labels.npy
A	results/ben-landcover/ben-landcover-s1s2-test-logits.npy
A	results/ben-landcover/ben-landcover-s1s2.json
A	results/ben-landcover/ben-landcover-s2-test-idx.npy
A	results/ben-landcover/ben-landcover-s2-test-labels.npy
A	results/ben-landcover/ben-landcover-s2-test-logits.npy
A	results/ben-landcover/ben-landcover-s2.json
A	results/count-dior/ANALYSIS.md
A	results/count-dior/SOURCE.md
A	results/count-dior/analyze_count_dior.py
A	results/count-dior/count-dior-provisioning.json
A	results/count-dior/dior-count-val-evaluation.json
A	results/count-dior/dior-count-val-seed17-1000.json
A	scripts/check_b13_app_parity.py
A	scripts/test_check_b13_app_parity.py
```

## T13 scope boundary (recorded before any merge)

Only these T13-specific files are approved for the final tree:

- ben/adapt.py
- ben/test_adapt.py
- ben/run_adapt_gpu.txt
- ben/run_adapt_seeds_gpu.txt
- docs/hypotheses/T13.md
- docs/hypotheses/T13b.md

BEN architecture/normalizer/projection helpers are already supplied by T09.
Do not import T13's inherited backend/model/count/B13 changes or results, result
arrays, or pending T13b analysis. If the --no-ff merge is conflict-free, retain
only the listed file changes from it before committing. A conflict in model,
schema, loader, or result code stops and aborts that merge, even when it is outside
the allowlist. Trivial docs/.gitignore/adjacent conflicts may be resolved and logged.

## Environment and sequential merge/test record

Pending: isolated Python 3.11 venv, dependency installation, and per-branch runs.
