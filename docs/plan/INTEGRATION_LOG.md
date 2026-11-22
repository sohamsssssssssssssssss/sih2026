# T05 integration log — 2026-11-17

Current state: completed after Soham's authorized resolution; see the final
resumption section below. Earlier stop records are retained as history.

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

### Merge 2 — task/T09-ask-api @ a45d1f9

`git merge --no-ff task/T09-ask-api`: clean merge; commit
54ae066a7afee28434045adadc626ebc8e5982ba. No conflicts or resolutions.
Full `python3.11 -m pytest -q` in the locked venv: **1,322 passed, 5 failed,
2 skipped, 40 setup errors, 0 import/collection errors**, exit 1, 39.48 s.
Raw log: /private/tmp/t05-after-t09.log. The 45 unsuccessful test identities
match the T04 missing-package baseline exactly; both requested real CPU API
end-to-end tests passed. CUDA-only legacy adapter tests are skipped.
No model/data/result/training code was edited to make tests pass.

## Numeric invariance check after T09

Reran `.venv/bin/python3.11 docs/plan/build_results.py` using the approved external
transfer-run inputs. RESULTS.md SHA-256 before/after: `a0d7e3508c9bf687600c8ca728ed2d8e1771e595caf2c1a1a27a1ec22e23b371`.
The diff against both the pre-merge file and origin/main is empty. No result
number or generated line changed; there is no numeric diff to explain.

## Tracked-file size/privacy audit after T09

All tracked files over 5,000,000 bytes are listed below. Every one was already
in origin/main; no large file has been introduced by the approved integration.

| Path | Bytes | Origin |
|---|---:|---|
| data/manifests/bigearthnet/stage1-train-rebuild-ids-loop10.v1.json | 8703700 | pre-existing main |
| data/manifests/bigearthnet/stage1-train-shard-ids-audit.v1.json | 8700714 | pre-existing main |
| frontend/public/videos/satquery-intro.mp4 | 11224386 | pre-existing main |

Every tracked non-code path under data/ (including metadata, documentation and
annotation templates) is listed below. All are pre-existing main paths and none
was modified by this integration. Listing an annotation path does not modify it.

| Path | Bytes |
|---|---:|
| data/BEN_TXT_FACTS.md | 12526 |
| data/BIGEARTHNET_PREP.md | 25667 |
| data/boundaries/README.md | 1831 |
| data/boundaries/villages.v1.geojson | 370467 |
| data/demo/grounding/07272.jpg | 70739 |
| data/demo/manifest.json | 4311 |
| data/demo/resolution/loveda_LoveDA_images_png_0_gsd1.0.png | 246167 |
| data/demo/resolution/loveda_LoveDA_images_png_0_gsd10.0.png | 2661 |
| data/demo/resolution/loveda_LoveDA_images_png_0_gsd2.0.png | 62097 |
| data/demo/resolution/loveda_LoveDA_images_png_0_gsd5.0.png | 9871 |
| data/eval/README.md | 6846 |
| data/eval/questions.v1.jsonl | 33333 |
| data/flood/DECISIONS.md | 3236 |
| data/flood/README.md | 2394 |
| data/flood/REPORT.md | 5881 |
| data/ladder/0.3/loveda_Train_Rural_images_png_0_gsd0.3.png | 1845014 |
| data/manifests/bigearthnet/annotation-audit.v1.json | 3915 |
| data/manifests/bigearthnet/caption-geo-split.v1.csv.gz | 1308343 |
| data/manifests/bigearthnet/caption-geo-split.v1.csv.json | 717 |
| data/manifests/bigearthnet/norm_contract.json | 1777 |
| data/manifests/bigearthnet/pilot-candidates.v1.json | 386786 |
| data/manifests/bigearthnet/reference-map-audit.v1.json | 104072 |
| data/manifests/bigearthnet/reference-map-members.v1.txt | 18000 |
| data/manifests/bigearthnet/s1-members.v1.txt | 28972 |
| data/manifests/bigearthnet/s2-members.v1.txt | 204000 |
| data/manifests/bigearthnet/source-registry.v1.json | 2951 |
| data/manifests/bigearthnet/stage1-loop3-checks.v1.json | 347 |
| data/manifests/bigearthnet/stage1-loop4-contract-audit.json | 4915 |
| data/manifests/bigearthnet/stage1-loop4-preregistration.json | 2401 |
| data/manifests/bigearthnet/stage1-norm-loader-check.v1.json | 1179 |
| data/manifests/bigearthnet/stage1-norm-options.v1.json | 12408 |
| data/manifests/bigearthnet/stage1-norm-study.v1.json | 3068328 |
| data/manifests/bigearthnet/stage1-norm-train-ids.v1.json | 233334 |
| data/manifests/bigearthnet/stage1-real-validation.v1.json | 145958 |
| data/manifests/bigearthnet/stage1-shards-eval-2000.v1.json | 3362 |
| data/manifests/bigearthnet/stage1-shards-train-150000.attempt1-aborted-archive-order.v1.json | 25693 |
| data/manifests/bigearthnet/stage1-shards-train-150000.v1.json | 108678 |
| data/manifests/bigearthnet/stage1-shards-train-150000.v2.json | 18686 |
| data/manifests/bigearthnet/stage1-shards-train-2000.v1.json | 4091 |
| data/manifests/bigearthnet/stage1-split-verification.v1.json | 692 |
| data/manifests/bigearthnet/stage1-train-rebuild-ids-loop10.v1.json | 8703700 |
| data/manifests/bigearthnet/stage1-train-selection-150000.v1.json | 2617 |
| data/manifests/bigearthnet/stage1-train-shard-ids-audit.v1.json | 8700714 |
| data/manifests/cdse/flood-runs.v1.json | 9596 |
| data/manifests/cdse/scenes.v1.json | 3958 |
| data/manifests/cdse/sensor-necessity-results.v1.json | 4836 |
| data/manifests/flood_events.v1.json | 5397 |
| data/sar_gate/annotation_template.md | 24666 |
| data/sar_gate/sar_cheatsheet.md | 2583 |

No data/ path is added or modified. No Sen1Floods11 pack directory or its raw
S1/S2/hand-label arrays is tracked. The pack remains external and its licence
remains unconfirmed. T09's per-chip metric report is results evidence, not the
pack. No credential-like filenames or standard private-key/AWS/GitHub-token
signatures were found in newly introduced text files. Existing main's files
are inventoried without deleting or rewriting them.

### Merge 3 — task/T14-flood-pack @ df4a236

`git merge --no-ff task/T14-flood-pack`: Already up to date (ancestor of starting
main). No merge commit or conflict. Full `python3.11 -m pytest -q` in the same
locked venv: **1,322 passed, 5 failed, 2 skipped, 40 setup errors,
0 import/collection errors**, exit 1, 37.72 s. Raw log:
/private/tmp/t05-after-t14.log. Unsuccessful identities are identical to T09
and the T04 missing-package baseline. No allowed environment/path-only code fix
can supply the absent authentic training-package assets from this checkout.
No source, training, data, annotation, or result code was changed for these tests.

### Merge 4 — task/T13-india-adapt (next)

Will attempt --no-ff --no-commit at recorded head a0d7002. Only the recorded
adaptation code/test/run-cell/hypothesis allowlist may enter the final tree.
Any protected conflict aborts this merge with both sides preserved in this log.

## Mandatory stop — T13 merge aborted

The fourth merge stopped with a single add/add conflict in **ben/transfer.py**.
This is protected model/evaluation/data-loader code, not a trivial documentation
or adjacent-line conflict. No conflict was resolved and no T13 merge commit was
created. Ran `git merge --abort` successfully; the worktree returned clean to
4ff4eaf96f6aeae2259c532646685b3188a868b4. All automatically staged T13 changes,
including inherited model/result/data-array changes, were discarded by the abort.
No T13 code, hypotheses, pending analysis or result files entered integration.

Both sides, preserved for owner review:

- Ours: integration pre-merge `4ff4eaf96f6aeae2259c532646685b3188a868b4`, from T09 a45d1f9; blob `1efb712003f894b7c415ffbc0dfe521abd21781e`.
- Incoming: task/T13-india-adapt `a0d7002cb9a43162e297f98a795694982791acb5`; blob `12470d087be9c38e7b6784ec235e370f1d4a6373`.

Ours retains the original max-sigmoid projection and direct WorldCover loading.
Incoming adds differentiable `coarse_logits`, routes `to_coarse` through that
helper, and introduces `load_wc` (including mmap array loading) for adaptation.
Choosing either side affects the adaptation/model/loader interface. No choice
was guessed. Exact side-to-side diff:

```diff
--- ours: T09 ben/transfer.py
+++ incoming: T13 ben/transfer.py
@@ -59,0 +60,7 @@
+def coarse_logits(logits, members):
+    """Shared max-member pooling, differentiable for torch adaptation logits."""
+    if isinstance(logits, torch.Tensor):
+        return torch.stack([logits[:, idx].max(1).values for idx in members.values()], 1)
+    return np.stack([logits[:, idx].max(1) for idx in members.values()], 1)
+
+
@@ -62 +69,11 @@
-    return np.stack([prob[:, idx].max(1) for idx in members.values()], 1)
+    return coarse_logits(prob, members)
+
+
+def load_wc(pack):
+    """Load the same WorldCover arrays, labels and metadata for transfer and adaptation."""
+    meta = pd.read_parquet(pack / "meta.parquet")
+    info = json.loads((pack / "classes.json").read_text())
+    labels = np.load(pack / "labels_coarse.npy").astype(np.float32)
+    raw = {"s1": np.load(pack / "s1.npy", mmap_mode="r"),
+           "s2": np.load(pack / "s2_10m.npy", mmap_mode="r")}
+    return raw, labels, meta, info
@@ -111,3 +128 @@
-    meta = pd.read_parquet(wc / "meta.parquet")
-    wc_info = json.loads((wc / "classes.json").read_text())
-    y_wc_all = np.load(wc / "labels_coarse.npy").astype(np.float32)
+    raw, y_wc_all, meta, wc_info = load_wc(wc)
@@ -115 +129,0 @@
-    raw = {"s1": np.load(wc / "s1.npy"), "s2": np.load(wc / "s2_10m.npy")}
```

No test run is claimed for the conflicted transient tree. After the abort, code
and result files exactly match the last tested T14/no-op state. The only later
changes are this stop record and tracker bookkeeping. No additional integration
or repair is attempted after the protected conflict.

## Final review state (partial integration; not a clean-baseline claim)

- T04 9bde585 and T14 df4a236 were already ancestors of starting origin/main.
- T09 a45d1f9 was cleanly merged with --no-ff as 54ae066.
- T13 a0d7002 was aborted on protected loader/model conflict; it remains unmerged.
- T15, feature/change-vqa and keep/* were not explicitly merged or selected.
- T06, T07, T08, T09 are marked done with hashes; flood-otsu stays experimental
  with India-test IoU 0.631332 < 0.65; T15 stays open.
- Last full test result: **1,322 passed, 5 failed, 2 skipped, 40 setup errors,
  0 import/collection errors**. All unsuccessful identities and their missing
  authentic-package cause are listed above. GroundingDINO distribution remains
  unavailable; no substitute was installed. No training/data/result source fix
  or fabricated fixture was used to hide a failure.
- RESULTS.md was regenerated with an empty diff. All three >5 MB files and all
  49 data/ non-code paths are inherited main assets. No new large/data/private-key
  file or Sen1Floods11 pack was introduced; no result number changed.
- requirements-lock.txt captures the installed environment. Main and other
  worktrees are untouched. No main push, force push, branch deletion or history
  rewrite occurred. This partial branch is published only for review; further
  integration requires resolution of ben/transfer.py under the owner's rules.

## T05 resumed — authorized resolution and completion

Starting integration commit: 94740dd. The required check
`git diff --exit-code feature/ben-landcover task/T09-ask-api -- ben/transfer.py`
returned exit 0 and an empty diff: T09 copied the BEN helper unchanged.
Soham explicitly authorized taking T13's version in this case.

Retried `git merge --no-ff --no-commit task/T13-india-adapt`. Its only conflict
was ben/transfer.py. Resolved only that file with the incoming version. No other
model/data/result conflict occurred. The scoped two-parent merge commit is
**d6cb91fb45daa932708b5ae60c944e0d9e723ecc**, parents 94740dd and a0d7002.

Only adaptation code, support tests/run cells and hypotheses enter the final
integration tree. Unrelated inherited backend/model/count/B13 changes and result
files were restored to the previous state before committing. Their bytes were
preserved, not deleted, in ignored outputs/.T05-excluded-t13. An initial test
collection discovered those unmerged copies, so the quarantine folder was moved
to that hidden name; no repository test was removed. Two BEN support tests then
identified required run-cell source files. Those exist in T13 and were included
unchanged rather than skipped. No Kaggle cell was executed.

The complete included T13 code/hypothesis set (byte-compared to a0d7002) is:

- ben/adapt.py
- ben/run_adapt_gpu.txt
- ben/run_adapt_seeds_gpu.txt
- ben/run_landcover_gpu.txt
- ben/run_transfer_gpu.txt
- ben/test_adapt.py
- ben/test_landcover.py
- ben/test_transfer.py
- ben/transfer.py
- docs/hypotheses/T13.md
- docs/hypotheses/T13b.md

No T13b analysis, result array, checkpoint, count/B13 result or private pack is
added to the integration tree. The only shared model/loader helper change is
the explicitly authorized ben/transfer.py version. No path under models/, data/,
results/ or scripts/ changed from 94740dd. No tracked file or test was deleted.

### Required compatibility checks

Locked-venv pytest selection: ben/test_adapt.py, ben/test_transfer.py,
ben/test_landcover.py, core/tools/test_landcover.py and the WorldCover case of
backend/test_ask.py::test_real_cpu_end_to_end. Result: **49 passed, 0 failed,
0 errors**, 13.03 s. All T13 adapt tests, the real BEN checkpoints in all three
modes and the real T09 land-cover API test pass with T13's transfer.py.
Raw log: /private/tmp/t05-t13-targeted-final.log.

### Explicit skips without changing tested code

A narrow collection check in root conftest.py checks the named test/fixture
paths that depend on packages/stage1-t4-smoke assets. Missing files skip with
`fixture not in repo: <path>`; available inputs still run the original tests.
The real GroundingDINO adapter test skips because its package is absent;
synthetic GroundingDINO tests continue to run. groundingdino is uninstalled,
with no substitute. No tested training code, package data or test expectations
were edited to manufacture passing results.

JUnit self-checks assert that exactly the previous 45 failing/setup-error nodes
are now fixture skips, every skip has a reason, and unrelated synthetic package
threshold/revision tests plus the BigEarthNet checksum test still PASS. There
is no whole-module skip or xfail masking independent tests.

### Final full-suite gate

`PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 python3.11 -m pytest -q -rs --junitxml=/private/tmp/t05-final.xml`

**1,354 passed, 47 skipped, 0 failed, 0 errors**, exit 0, 52.62 s.
Exact reasons grouped across parametrized cases:

| Count | Reason |
|---:|---|
| 1 | dependency not installed: groundingdino |
| 1 | fixture not in repo: packages/stage1-t4-smoke/config-full.yaml |
| 12 | fixture not in repo: packages/stage1-t4-smoke/config-last4-vision-fp32.yaml |
| 29 | fixture not in repo: packages/stage1-t4-smoke/config.yaml |
| 1 | fixture not in repo: packages/stage1-t4-smoke/package-manifest.json |
| 1 | fixture not in repo: packages/stage1-t4-smoke/revision.json |
| 1 | fixture not in repo: packages/stage1-t4-smoke/src/scripts/run_stage1_smoke.py |
| 1 | qwen2.5vl-3b: CUDA_UNAVAILABLE: Qwen2.5-VL inference requires a CUDA GPU for this baseline; use the Kaggle T4 runner. For local iteration only, set SATQUERY_ALLOW_MPS=1 to permit Apple MPS — its output is not a baseline number and must never be reported as one. |

47 total skips = 45 absent package fixtures + 1 absent GroundingDINO runtime
+ 1 CUDA-only Qwen adapter. Both requested real CPU API tests pass in the full
suite. Raw log: /private/tmp/t05-final.log. JUnit SHA-256:
`5833f4e1e0f208be5e36d0383577d0d04d640a71a36f91b6435e8d211ff4745c`. Machine-readable evidence is preserved locally in
outputs/T05-final-validation.json. requirements-lock.txt is unchanged; no
package installation or substitution occurred during this resumption.

### Results, scope and final tracker

Reran `.venv/bin/python3.11 docs/plan/build_results.py`. RESULTS.md is byte-identical
to 94740dd with an empty diff; no result number or line changed. SHA-256:
`a0d7e3508c9bf687600c8ca728ed2d8e1771e595caf2c1a1a27a1ec22e23b371`. The earlier three inherited >5 MB files and 49
non-code data/ paths remain unchanged. No Sen1Floods11 pack or excluded result
array was added to the integration file tree.

T06/T07/T08/T09 stay done with their source hashes. T13 code/hypotheses are now
merged; **T13b analysis remains pending**. flood-otsu remains experimental_unvalidated,
India-test IoU 0.631332 < 0.65; T15 stays open and unmerged. Only integration/spine-v1
is pushed. No main merge/push, force push, deletion or history rewrite occurred;
protected worktree files were not inspected or modified.

## 18 Nov 2026 — T13b registered paired-bootstrap results

Merged `task/T13-india-adapt` at `eb1bb87` into `integration/spine-v1`
at `de23faa`. Merge commit: `3cec3da`. The only conflict was an add/add
in `docs/plan/DECISIONS.md`. Both decision histories were preserved, with
the T13b entries placed by date and no existing entries removed. The
`results/t13b/` files match `eb1bb87` exactly. The merge introduced no
NPZ files or blobs over 5 MB.

Full suite in the existing Python 3.11 venv (`python3.11 -m pytest -q -rs`):
**1,366 passed, 47 skipped, 0 failed, 0 errors** in 66.32 s. The
baseline had 1,354 passed and 47 skipped. The skip reasons match the
baseline.

## 2026-11-20 — Sentinel-2 BOA offset and regenerated backups

Earth Search reported its Ludhiana optical chip as already offset-corrected, while the asset metadata advertised an additional −0.1 reflectance offset. Fresh S2 acquisition now honours the applied flag, checks the processing baseline when the flag is false or absent, and refuses live acquisition when metadata cannot establish the offset rule. The manifest points to new Ludhiana, Delhi and Mumbai v2 files with the original item IDs and pixel grids; the superseded files and their SHA-256 values are retained. Ludhiana v2 mean reflectance in B02/B03/B04/B08 order is 0.09153165 / 0.10402494 / 0.11323595 / 0.15472317.

Verification: pytest 1,398 passed, 47 skipped, 0 failed. Vitest 49 passed in 11 files. TypeScript no-emit check passed with zero diagnostics. Playwright live rerun 8 passed and 1 offline-only skip. The offline-only case passed separately, so all 9 cases passed across the two modes. The initial live run timed out at the 30-second browser assertion while the S2 fetch was pending.

Ports 3000 and 8000 were free before each launch. Both sessions ran through `bash scripts/run-demo.sh` and were stopped after verification. API results below are from the Europe S2 checkpoint.

| Scene | Source and item | Verbatim API answer | Highest score |
|---|---|---|---|
| Ludhiana | Live Earth Search, `S2B_43REQ_20261028_0_L2A` | Uncalibrated coarse land-cover presence scores. | Built 0.954770 |
| Delhi | Live Earth Search, `S2B_43RGM_20261005_0_L2A` | Uncalibrated coarse land-cover presence scores. | Built 0.949291 |
| Punjab | Local WorldCover demo fixture, patch 0, no STAC item | Uncalibrated coarse land-cover presence scores. Cloud cover not checked (no cloud metadata). | Cropland 0.378211 |

## 2026-11-20 — Live fetch deadline and Mumbai fixture

Live scene acquisition defaults to 20 seconds overall, including place lookup, STAC selection and all band reads. Set `SATQUERY_LIVE_FETCH_TIMEOUT` in seconds to override it. A timeout uses a cache for the requested place or chip footprint, with the backup acquisition date in the message. If no matching cache exists the API returns HTTP 504 with the reason. GDAL can finish the current read in a worker after the deadline. Expired requests do not publish live chips.

The Mumbai fixture was introduced on 2026-11-18 in `78e577e`. Its source, Earth Search item `S2C_43QBB_20250829_0_L2A`, differs from the Mumbai backup item. A fresh read of the same 120 × 120 window confirmed a duplicate −1,000 DN offset. Mean reflectance in B02/B03/B04/B08 order was 0.49092194 / 0.47021519 / 0.46295581 / 0.45865514 before and 0.59092194 / 0.57021519 / 0.56295581 / 0.55865514 after. The new fixture retains the grid and original cloud metadata. The old file is retained.

Verification: pytest **1,401 passed, 47 skipped, 0 failed**. Vitest **50 passed** in 12 files. TypeScript no-emit passed with zero diagnostics. Playwright live run **8 passed, 1 offline-only skip**, and the offline case **1 passed** separately.

The default-budget Ludhiana request first fell back at 20.04 seconds and then returned live on retry in 0.89 seconds. The one-second run returned cached in 1.03 seconds with “Live fetch timed out after 1s — showing cached scene from 2026-10-28”. Both used item `S2B_43REQ_20261028_0_L2A` and answered “Uncalibrated coarse land-cover presence scores.” Built areas ranked first at 0.954770 in both. Ports 3000 and 8000 were free before each launch, and the owned demo sessions were stopped after verification.

## 22 Nov 2026 — Mac MLX B13 VQA

G6 run 2 passed at `8231a76` with 3.227 GiB minimum and 3.493 GiB median available against the 3 GiB threshold. Swap used rose from 0 MiB to 0.25 MiB. Finder, Terminal, and Safari were open. The controlling Codex session and claude-mem helpers were present. The passing gate report and 45 raw samples were pushed in `cca98d0`. G1–G5 use the original run. [G6 report](../../results/vqa-mac-fresh-test/REPORT.md)

The feature-enabled demo used the same app setup. Three cached-image questions and one count question went through the running `/ask` endpoint, with the frontend open in Safari. All four traces verified. Worker latency is the value in the tool trace. HTTP latency includes scene loading, gating, and transport.

| Cached scene | Question | Answer (verbatim) | Tool latency | HTTP latency |
|---|---|---|---:|---:|
| Ludhiana S2 | Are there agricultural fields in the image? | `yes` | 529 ms | 574.0 ms |
| Delhi S2 | Are there more buildings than water areas? | `yes` | 362 ms | 384.8 ms |
| Ludhiana S2 | Is this a rural or an urban area? | `urban` | 366 ms | 388.0 ms |
| Delhi S2 | How many buildings are there? | `counting is not image-based in this model` | — | 20.0 ms |

Pytest passed 1,414 tests with 47 skips, including a pass for the opt-in real MLX worker test. Vitest passed 50 tests across 12 files. TypeScript passed the no-emit check with zero diagnostics. The backend venv reported PyArrow 22.0.0.

The full flag-on Playwright run passed 8 cases, including real Mac VQA and the count refusal. It failed 1 live STAC case and skipped the offline-only case. The offline case passed in a separate run. On the live retries, Ludhiana and Silchar loaded live while Mumbai hit the 20-second fetch deadline, fell back to its cached scene, and correctly refused on cloud cover. The Mumbai timeout also occurred with MLX disabled. The live assertion still requires the Live scene badge.

## 22 Nov 2026 — v0.9-demo evidence import

Cherry-picked the T18 preregistration and baseline results (`86cdf97` and `1d0cbaf`). Copied the India land-cover promotion results and gate documents from `feature/ben-landcover` at `4695cd9`. The gate paths did not overlap existing files. [T18 baseline report](../../results/t18/REPORT.md) · [India land-cover promotion decision](../../results/landcover-india-promotion/DECISION.md)

- Full pytest: **1,414 passed, 48 skipped, 1 failed**. The Stage 1 throughput smoke test completed one step against an expectation of two. Its three isolated reruns were **pass, fail, pass**. The run used the default pytest configuration for the opt-in real MLX worker test.
- Vitest: **50 passed** across 12 files. TypeScript no-emit: **0 diagnostics**.
- The first two MLX-enabled Playwright runs each had **8 passed, 1 skipped, 1 failed**. Ludhiana hit the 20-second live-fetch deadline and displayed its cached scene. The old test required the live badge.
- After `ce6de54`, Playwright passed **9 default cases**, including Mac VQA and live STAC imagery for Ludhiana, Silchar and Mumbai, with **2 opt-in cases skipped**. The separate offline fallback case **passed**. Set `SATQUERY_E2E_STRICT_LIVE=1` to require live imagery for all three scenes in the strict case.
- A 0.01-second deadline forced cached responses. The default STAC case passed and the strict-live case failed at the live badge assertion, as expected.
