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
- ben/test_transfer.py (imported BEN19 fixture)
- ben/test_landcover.py (imported make_pack fixture)
- ben/transfer.py (T13 shared coarse_logits/load_wc dependency)
- ben/run_adapt_gpu.txt
- ben/run_adapt_seeds_gpu.txt
- docs/hypotheses/T13.md
- docs/hypotheses/T13b.md

BEN architecture/normalizer/projection helpers are already supplied by T09.
Inspection found T13 also needs its coarse_logits and load_wc additions in
ben/transfer.py; that loader/model file is included in the code allowlist and
remains subject to the mandatory conflict stop rule.
Do not import T13's inherited backend/model/count/B13 changes or results, result
arrays, or pending T13b analysis. If the --no-ff merge is conflict-free, retain
only the listed file changes from it before committing. A conflict in model,
schema, loader, or result code stops and aborts that merge, even when it is outside
the allowlist. Trivial docs/.gitignore/adjacent conflicts may be resolved and logged.

## Environment and sequential merge/test record

Pending: isolated Python 3.11 venv, dependency installation, and per-branch runs.

### Dependency setup, attempt 1

Created .venv with Python 3.11.15. Ran venv pip against requirements.txt,
backend/requirements.txt, and explicit peft/streamlit/pyarrow/hypothesis.
Resolver failed before installation: `No matching distribution found for
groundingdino` (no compatible versions listed). No package is substituted and
repository requirements are unchanged. The remaining identical requirements
will be installed from a temporary list excluding only this unavailable entry.
The missing optional provider distribution will remain an explicit limitation.
Full installer output: /private/tmp/t05-install.log.

### Dependency setup, completed available set

Installed all available requested requirements into .venv, plus scikit-learn
(already required by the approved T09 branch). peft, streamlit and pyarrow are
installed. pip check reports no broken installed requirements. groundingdino
remains unavailable from the configured index; no replacement was installed.
Exact installed distributions are frozen in requirements-lock.txt. No model or
training source was changed. The initial RESULTS.md generator check is bit-identical.

### Merge 1 — task/T04-plan-files @ 9bde585

`git merge --no-ff task/T04-plan-files`: Already up to date (ancestor of starting
main). No merge commit or conflict. Full test command uses the venv on PATH:
`python3.11 -m pytest -q`. Result: **1,066 passed, 5 failed, 40 setup errors,
0 import/collection errors**, exit 1, 53.62 s. Raw log:
/private/tmp/t05-after-t04.log. Dependency import blockers are resolved; the
unavailable optional groundingdino package does not cause an import error.

All 45 unsuccessful tests fail on missing untracked package assets under
packages/stage1-t4-smoke: config.yaml, config-full.yaml,
config-last4-vision-fp32.yaml, revision.json, package-manifest.json, or the
packaged src/scripts/run_stage1_smoke.py. No package assets are tracked in any
Git history at that path. Root training scripts themselves exist unchanged.
The four old FP32 smoke tests now pass with peft installed, and the old
bigearthnet_shards failure now passes with pyarrow installed. The two old
package smoke tests remain missing-fixture failures. Imports newly succeeding
expose additional missing-package setup errors; these are not merge regressions.

Rule 4 boundary: no source, test expectations, training code, data or result
code was changed. Authentic package configs/captions/patch archive cannot be
reconstructed from an approved path in this checkout. Asked for an approved
fixture source outside protected worktrees; no fixture is invented or copied
from /Users/soham/sih2026 or another worktree. Each unsuccessful test below has
that same missing-package cause and remains unfixable within the current inputs:

- FAILED scripts/test_stage1_full.py::test_config_full_yaml_has_no_placeholder_steps — missing untracked package fixture; no code/data fix applied.
- FAILED scripts/test_stage1_smoke.py::test_t4_smoke_package - FileNotFoundErro... — missing untracked package fixture; no code/data fix applied.
- FAILED scripts/test_stage1_smoke.py::test_t4_package_entry_and_stages - FileN... — missing untracked package fixture; no code/data fix applied.
- FAILED scripts/test_stage1_t4_package.py::test_package_manifest_integrity - F... — missing untracked package fixture; no code/data fix applied.
- FAILED scripts/test_stage1_t4_package.py::test_updated_package_runner_matches_source — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_cpu_resume_restores_all_state_without_gpu_checkpoint_copy — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[vision_fp32] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[vision_trainable_last_n_blocks] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[learning_rates] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[seed] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[steps] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[captions] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[contract] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[shuffle] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_relocated_identical_config_preserves_identity — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_eval_cpu_load_and_shuffled_arm_keep_training_identity — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_resume_variant.py::test_missing_identity_and_bad_data_position_are_rejected — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_local_dir_metadata_proves_revision — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_missing_source_fails - FileN... — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_metadata_mismatch_fails - Fi... — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_weight_mismatch_fails - File... — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_smoke_missing_source_never_reaches_model — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[6666666666666666666666666666666666666666] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[7777777777777777777777777777777777777777] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[None] — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_missing_weight_file_is_recorded_failure — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_empty_expected_weights_never_passes — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_revision_gate.py::test_failed_proof_is_not_overwritten — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_verify_passes_when_hashes_match — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_verify_aborts_on_weight_mismatch — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_verify_aborts_on_revision_mismatch — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_passes_first_attempt_and_records_rate — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_falls_back_to_vision_fp32 — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_both_attempts_fail — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_resume_stops_and_resumes_with_continuity — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_resume_fails_beyond_threshold — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_overfit_gates_pass_and_steps_sized_from_smoke — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_overfit_shuffle_gate_fails_when_ratio_low — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_stage_extrapolate_uses_measured_rate — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_happy_path_runs_stages_in_registered_order — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_main_stops_at_first_failed_stage — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_memory_matrix_records_control_oom_and_variant — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_explicit_last4_fp32_reaches_resume_and_overfit — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_resume_only_runs_exact_variant_and_keeps_previous_evidence — missing untracked package fixture; no code/data fix applied.
- ERROR scripts/test_stage1_t4_package.py::test_resume_execution_failure_is_preserved — missing untracked package fixture; no code/data fix applied.
