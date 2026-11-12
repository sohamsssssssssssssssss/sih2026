# Detection counts with explicit VLM fallback

Branch `feature/count-grounding`, based on `app/b13-adapter` at `bcd0e23`. Implementation lives only in `/Users/soham/sih2026-count`. No files under `training/`, `packages/`, `results/`, or `data/manifests/` were changed. The app worktree was used only for the explicitly requested baseline test run, with bytecode/cache writes disabled. Nothing is pushed or merged.

## Existing and updated app path

`backend/services.py` builds a plan and executes it through `orchestrator/executor.py` and `orchestrator/router.py`. General question intents live in `orchestrator/question.py`; the existing B13-specific count target parser is `models/qwen_vl/b13.py::count_target`. The B13 count route remains enabled by setting `SATQUERY_VQA_ADAPTER` before starting the API. Non-count VQA still uses its original adapter loading, prompt and inference path.

The parser recognizes `How many …`, `What is the number of …`, `What is the amount of …`, bare `number of …` / `amount of …`, and `Count [the] …`. It preserves plural/adjectival object phrases and strips generic `are/is …` and `in the image/scene/picture/photo` suffixes. It does not add lemmatization, a language model parser or a new dependency.

For a single-image count request, the planner selects Grounding DINO when ready; otherwise it selects single-image VQA with rule `b13_unverified_count_fallback`. This availability decision is required before execution because the existing backend refuses unavailable plans. The router checks detector readiness again, invokes the actual selected provider using the existing killable-worker seam, and traces its actual provider/source.

`models/grounding_dino/count.py::count_objects` accepts a **detector callable**, `(image_paths, object_prompt) -> {evidence: [...]}`. The production callable invokes `GroundingDINOModel.infer`; CPU tests supply deterministic callables instead. No mock detector is shipped as a production fallback.

The counting policy:

1. Require normalized `xyxy` boxes and finite scores/coordinates; malformed detector evidence fails instead of inventing a count.
2. Discard zero-area boxes and scores **less than or equal to** the configured box threshold. This matches the detector's strict score-cutoff semantics.
3. Sort by confidence, highest first, with stable ordering for ties. Apply greedy, class-agnostic NMS for the single object prompt. Suppress a box when IoU is **greater than** the configured NMS threshold; equality is retained.
4. Answer with the number of retained boxes. Zero detections is a detector-backed zero, not an unavailable-detector fallback.

## Threshold configuration

`configs/count_grounding.json` is the configuration source, without runtime hard-coded score/NMS cutoffs:

```json
{"box_threshold": 0.35, "text_threshold": 0.25, "nms_iou": 0.5}
```

The box/text defaults of `GroundingDINOModel` now read that file; explicit constructor overrides remain available for the evaluator. The existing CUDA-only lazy loader is unchanged. The count postprocessor uses the same settings, and both scores and NMS boundaries are tested on CPU. Restart the API/worker after changing this file so the loaded detector and router use consistent settings. These defaults are **not calibrated counting settings**; detector counting has not been measured.

## Answer evidence and fallback

Successful detection-count responses preserve the existing UI box contract (`bounding_box`, `normalized_xyxy`, score and coordinates), add an evidence summary, and expose `count` / `count_thresholds`:

```json
{
  "answer": "1",
  "count": 1,
  "answer_source": "grounding_dino_count",
  "count_thresholds": {"box_threshold": 0.35, "text_threshold": 0.25, "nms_iou": 0.5},
  "evidence": [
    {"type": "bounding_box", "coordinates": [0.1, 0.1, 0.3, 0.3],
     "coordinate_space": "normalized_xyxy", "confidence": 0.9},
    {"type": "detection_count", "object": "buildings", "count": 1,
     "thresholds": {"box_threshold": 0.35, "text_threshold": 0.25, "nms_iou": 0.5}}
  ]
}
```

This is an illustrative payload, not a measured output. Existing UI overlay code already reads box evidence. The thresholds and answer source are also recorded in the hash-chained trace.

If Grounding DINO readiness reports unavailable (including absent CUDA, dependencies, checkpoint or text encoder), the router invokes the current VLM with the **original** question. It preserves the VLM's answer, sets `answer_source: vlm_unverified_count`, and adds:

```json
{"unverified_count": "detector_unavailable"}
```

The response notice contains the literal marker `unverified_count: detector_unavailable` and explains that no detection evidence backs the estimate; the marker is traced. Direct B13 VLM count generations carry the same unverified label. No numeric detector count or boxes are fabricated for fallback. The VLM must itself be ready; otherwise the normal unavailable-provider error is returned. Detector execution/validation errors are not silently reclassified as readiness failures. Non-count VLM behavior is unchanged.

## Frozen validation and CUDA evaluation

The manifest was located read-only at:

`/Users/soham/sih2026/packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/data/manifests/rsvqa/pilot.v1/validation.jsonl`

Its SHA256 is `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`. It has 389 validation questions, of which 100 are count questions. `kaggle/eval_count_detection.py` rejects a different manifest digest, row count, duplicate count IDs, unparseable count prompt or noncanonical integer target. It never selects training or official test questions.

Provision the count branch code, B13 V1.1 dataset, Grounding DINO checkpoint/config and cached `bert-base-uncased` text encoder separately. The B13 dataset metadata identifies `sohamssssss/satquery-rgb-vqa-b13-v11`; mount it with its `data/manifests/...` and `images/` tree intact. Dependencies must be installed before enabling offline mode. The script does not install/download models, run training or fall back to a VLM.

Exact command on a Kaggle CUDA runtime, assuming these mounts:

```bash
cd /kaggle/working/sih2026
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SATQUERY_GROUNDING_CHECKPOINT=/kaggle/input/satquery-models/groundingdino_swint_ogc.pth
python kaggle/eval_count_detection.py \
  --manifest /kaggle/input/satquery-rgb-vqa-b13-v11/data/manifests/rsvqa/pilot.v1/validation.jsonl \
  --image-root /kaggle/input/satquery-rgb-vqa-b13-v11/images \
  --config configs/count_grounding.json \
  --thresholds 0.25 0.30 0.35 0.40 \
  --output /kaggle/working/count-detection-validation.json
```

The packaged Grounding DINO config is discovered automatically, or set `SATQUERY_GROUNDING_CONFIG` to its provisioned local `GroundingDINO_SwinT_OGC.py`. Use a Python 3.11-or-newer runtime with the detector dependencies installed. Output must be fresh. The command was **not executed** in this task.

The evaluator runs detection at the lowest requested box threshold once per question, then applies the exact shared count policy for each requested threshold. It reports `n`, exact integer count accuracy and mean absolute error at the configured app threshold plus all four sweep thresholds, with per-question expected/predicted counts, prompts, boxes and threshold evidence. It records the checkpoint and actual input-image hashes for provenance. The original supplied images are required; recording their hashes does not independently prove that they match an earlier evaluation's image bytes.

## What is not yet measured

**No detector count accuracy claim is made.** The CUDA evaluation, threshold calibration, real detector NMS/count behavior, latency/memory and comparison with the existing VLM have not been run. CPU tests validate policy/shape/routing only. The detector is still unavailable on the Mac's MPS/CPU path, so a ready VLM returns labelled estimates there.

Dense small buildings, roads/area counting, overlapping objects and caption-driven detections may not correspond to RSVQA's annotated instance counts. The parser retains some descriptive/relational wording but does not enforce spatial relationships or separate multiple categories. No general language/object ontology or spatial-relation solver was added. Validation threshold sweeps are exploratory; selecting a setting from this validation set is not independent held-out evidence.

## Tests: before and after

Baseline was run **on `app/b13-adapter` before edits**, at `bcd0e23`, with:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/soham/sih2026-app-b13/.venv/bin/python \
  -m pytest -p no:cacheprovider -q
```

The same interpreter/flags and full suite were then run from `/Users/soham/sih2026-count`.

| Run | Passed | Failed | Setup errors | Subtests passed |
|---|---:|---:|---:|---:|
| Base app/b13-adapter | 1,016 | 5 | 40 | 28 |
| feature/count-grounding | 1,047 | 5 | 40 | 28 |

The ordered failure/error node lists are **identical**. Every failure below existed on the base before this change. They reference absent `packages/stage1-t4-smoke` configs, archive, package manifest or packaged source; that protected directory was not generated or modified. Core suite (`backend orchestrator models kaggle/test_eval_count_detection.py`) passes **707 tests**, 12 subtests. There are 31 new CPU tests; the existing count tests were updated for the explicitly requested fallback and richer evidence contract. Pure evaluator scoring/hash tests import helper functions only; neither its CUDA entry point nor detector inference was invoked.

### All pre-existing failures and setup errors

```text
FAILED scripts/test_stage1_full.py::test_config_full_yaml_has_no_placeholder_steps — PRE-EXISTING on app/b13-adapter
FAILED scripts/test_stage1_smoke.py::test_t4_smoke_package — PRE-EXISTING on app/b13-adapter
FAILED scripts/test_stage1_smoke.py::test_t4_package_entry_and_stages — PRE-EXISTING on app/b13-adapter
FAILED scripts/test_stage1_t4_package.py::test_package_manifest_integrity — PRE-EXISTING on app/b13-adapter
FAILED scripts/test_stage1_t4_package.py::test_updated_package_runner_matches_source — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_cpu_resume_restores_all_state_without_gpu_checkpoint_copy — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[vision_fp32] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[vision_trainable_last_n_blocks] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[learning_rates] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[seed] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[steps] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[captions] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[contract] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_resume_rejects_changed_identity[shuffle] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_relocated_identical_config_preserves_identity — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_eval_cpu_load_and_shuffled_arm_keep_training_identity — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_resume_variant.py::test_missing_identity_and_bad_data_position_are_rejected — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_local_dir_metadata_proves_revision — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_missing_source_fails — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_metadata_mismatch_fails — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_weight_mismatch_fails — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_smoke_missing_source_never_reaches_model — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[6666666666666666666666666666666666666666] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[7777777777777777777777777777777777777777] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_config_source_has_priority[None] — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_missing_weight_file_is_recorded_failure — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_empty_expected_weights_never_passes — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_revision_gate.py::test_failed_proof_is_not_overwritten — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_verify_passes_when_hashes_match — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_verify_aborts_on_weight_mismatch — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_verify_aborts_on_revision_mismatch — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_passes_first_attempt_and_records_rate — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_falls_back_to_vision_fp32 — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_smoke_both_attempts_fail — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_resume_stops_and_resumes_with_continuity — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_resume_fails_beyond_threshold — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_overfit_gates_pass_and_steps_sized_from_smoke — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_overfit_shuffle_gate_fails_when_ratio_low — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_stage_extrapolate_uses_measured_rate — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_happy_path_runs_stages_in_registered_order — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_main_stops_at_first_failed_stage — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_memory_matrix_records_control_oom_and_variant — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_explicit_last4_fp32_reaches_resume_and_overfit — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_resume_only_runs_exact_variant_and_keeps_previous_evidence — PRE-EXISTING on app/b13-adapter
ERROR scripts/test_stage1_t4_package.py::test_resume_execution_failure_is_preserved — PRE-EXISTING on app/b13-adapter
```

No frontend files were changed or frontend suite rerun. No full GPU evaluation was run. Existing worker isolation, timeouts, local-only model resolution, input checks and hash-chain validation remain in place.
