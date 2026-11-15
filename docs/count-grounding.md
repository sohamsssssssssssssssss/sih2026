# Honest count resolution and original-DIOR detector evaluation

Branch `feature/count-grounding`, continuing `f28b862`. No changes to `training/`, `packages/`, `results/` or `data/manifests/`. No model evaluation, training, dataset download or large extraction was run on the Mac. Only CPU tests and read-only label/source inspection were performed.

## App count path and GSD

The B13-specific count path remains opt-in through `SATQUERY_VQA_ADAPTER`. `models/qwen_vl/b13.py::count_target` extracts the existing count noun phrase. The backend supplies GSD to `orchestrator/router.py`; `models/grounding_dino/resolution.py` applies the resolution policy before detector/VLM inference. Non-count model prompts, adapter loading and routing are unchanged.

GSD source precedence in `backend/services.py::scene_gsd`:

1. TIFF affine transform: `abs(transform.a)` **only for projected CRS units declared as metres** and affine georeferencing. Geographic degrees, feet, identity/missing transforms and GCP/RPC-only georeferencing do not become metre GSD. The raster metadata stores native `gsd_m`; the existing upload response's formerly-null `gsd` is populated, e.g. `10 m/px`.
2. Stored scene/catalog GSD or a source filename carrying an explicit token such as `_gsd0.3`. For uploaded TIFF previews, the gate multiplies native metre GSD by the maximum native-to-preview downsampling factor, so it does not pretend the detector receives the full native resolution.
3. Scene-ID token such as `tile_gsd0.3`.
4. Otherwise unknown. Sensor name, image dimensions or a dataset name alone are not used to guess an uploaded scene's GSD.

The response/trace records `gsd_m` and `gsd_source`. Scene-ID/metadata GSD is declared metadata, not independent verification. The horizontal affine-spacing rule assumes the usual north-up, approximately square-pixel imagery; unusual rotation/anisotropy and inaccurate metadata remain limitations.

## Resolution policy: assumptions, not facts

`configs/count_grounding.json` contains score thresholds and a `resolution` section. The rule is:

`assumed object_size_m / effective_gsd_m >= min_pixels`

`min_pixels` defaults to **8**. The configured class dimensions are:

| Class | Assumed size (m) | Maximum allowed GSD at 8 pixels (m/px) |
|---|---:|---:|
| building | 8 | 1 |
| vehicle | 2 | 0.25 |
| ship | 10 | 1.25 |
| airplane | 12 | 1.5 |
| storage tank | 8 | 1 |
| tree | 3 | 0.375 |
| default / unknown class | 3 | 0.375 |

**Source: project engineering assumptions in the configuration, as requested for this gate.** These dimensions and the eight-pixel cutoff are not measurements, literature-derived class minima or guarantees of detection. They need calibration. Lexical plurals are recognized, with aliases such as car/truck → vehicle and plane → airplane. The gate is deliberately simple and does not infer actual object size from the image.

Payload statuses:

- `not_resolvable`: refuse before either model runs, even if both providers are unavailable. The answer explains current and required GSD, e.g. “The building count cannot be determined at 10 m/px. Imagery at 1 m/px or finer is required by the resolution policy.” There is **no numeric count field or count estimate**. The policy decision has evidence and a hash-chained trace with provider `count-resolution-gate`.
- `detected`: known GSD passes the policy and a ready detector returns the post-NMS count.
- `gsd_unknown`: a ready detector runs, but the payload, evidence and notice carry `unverified_gsd`. No claim of physical resolvability is made.
- `detector_unavailable`: when the gate did not refuse, preserve the existing VLM fallback labelled `unverified_count: detector_unavailable`. Unknown GSD remains an additional `unverified_gsd` label. If the VLM is also unavailable, the normal unavailable-provider error remains.

Common gate fields include `gsd_m`, `gsd_source`, `object_class`, `object_size_m`, `required_gsd_m` and `min_pixels`. The API notice distinguishes a policy refusal from model inference. The backend handles refusals before its normal unavailable-plan rejection, so missing hardware cannot cause an unresolvable count to fall through to a VLM.

Examples for buildings: **10 m/px → not_resolvable; 0.3 m/px → detector path; no GSD → detector path labelled unverified_gsd**, subject to detector availability. A gate pass is permission to attempt detection, not evidence that the count is correct.

## Detector policy and evidence

The existing lazy CUDA-only Grounding DINO provider receives the object prompt. `models/grounding_dino/count.py::count_objects` accepts an injectable detector callable for CPU tests. Its settings remain box score `0.35`, text score `0.25`, NMS IoU `0.5`, all configured in the JSON file. Restart the API/worker after config changes.

Boxes must be finite normalized xyxy with scores. Discard zero-area boxes and scores at or below the box threshold. Stable greedy NMS suppresses IoU greater than its configured threshold. Return the number of retained boxes, existing UI-compatible `bounding_box` evidence, a `detection_count` summary with thresholds/resolution evidence, and `count_thresholds`. No evidence is fabricated for VLM fallback. Malformed detector evidence is an error rather than silent fallback.

## Why RSVQA-LR is a negative control, not this detector's count benchmark

The specified RSVQA-LR inputs are 256-pixel Sentinel-2 imagery at 10 m/px. The configured gate rejects the frozen count questions at that resolution rather than trying to count subpixel/small instances. The frozen 389-question manifest has **100 count questions; 24 ground-truth answers are exactly `"0"`**, verified from its bytes. This is a label-prior fact, not a detector result or evidence that the adapter always predicts zero. The existing cohort includes building targets as high as **7,116**, not an upper bound of 312. A real-versus-blank tie and the zero-label prior do not establish image-grounded counting.

The frozen LR manifest SHA256 remains `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`. It is accepted only under the explicit `--mode lr_negative_control`. That mode checks that all count questions receive `not_resolvable` at the dataset-declared 10 m/px, records the zero-label count, and reports **no detector accuracy**. The evaluator CLI remains CUDA-guarded even though this mode only checks the policy. Its gate logic was tested only with synthetic CPU examples, not run on the real cohort.

## Original DIOR protocol

Use original DIOR detection annotations, **not DIOR-RSVG referring-expression indices**. [Li et al., 2020](https://arxiv.org/abs/1909.00133) describes DIOR's 23,463 images and 20 classes. The supplied source is the [Zenodo 11213149 third-party mirror](https://zenodo.org/records/11213149), file [DIOR-VOC.zip](https://zenodo.org/records/11213149/files/DIOR-VOC.zip). The record identifies it as an Open-access re-upload after the original host became inaccessible; it is not the original publisher's host.

**Mirror verified on Kaggle (CPU run, 13 Nov 2026), now pinned:** 7,423,852,627 bytes, sha256 `7a386a56ba815333627d89d3b8b372d1af2a07b78955c017530ec62b7f1e0540`. `kaggle/eval_count_detection.py` (`verify_archive`), `kaggle/fetch_dior_cpu.sh` and `kaggle/run_count_dior_gpu.txt` all check this size and hash. Any mismatch is a hard stop.

| | Paper (Li et al. 2020) | This mirror (13 Nov 2026) |
|---|---|---|
| Images | 23,463 | 23,454 JPEG |
| Annotations | n/a | 23,455 XML (1 image missing) |
| train / val / test split lines (`wc -l`) | 5,862 / 5,863 / 11,738 | 5,861 / 5,860 / 11,734 |

The `wc -l` figures can undercount by one when a file lacks a final newline. Paper split sizes are as commonly cited; the evaluator records the exact parsed sizes in `split_sizes`.

**Layout** is required to be `<data-root>/DIOR-VOC/VOC2007/{ImageSets/Main,Annotations,JPEGImages}` (`dataset_root`). If that layout is absent, the evaluator extracts the ZIP into an empty data root. Otherwise it fails with the found tree; it does not search for or guess another root. Split files are parsed robustly: whitespace is stripped, blank and CRLF lines are skipped, and a missing final newline is fine. IDs must be unique, and train/val/test must be disjoint.

**Missing files.** Before sampling, val ids whose XML or JPEG is missing are dropped. Train ids are dropped only for a missing XML, because the majority baseline needs no pixels. The dropped ids and counts (`missing_xml`, `missing_image`, `n_dropped`, `n_requested` per split) are written to both the frozen sample manifest and the output JSON under `dropped_before_sampling`. **Invalid annotations** are also dropped before sampling, in both train and val, and recorded under `invalid_annotation` as `{id, reason, detail}`. The reasons are `parse_error`, `bad_filename` (missing or mismatched `<filename>`), `nonstandard_size` (declared size not 800×800; detail holds the declared width/height), `unknown_class` and `bad_bbox` (missing, non-numeric, out-of-range or inverted box). For example, DIOR `Annotations/00131.xml` declares 800×802. If invalid annotations exceed 1% of a split's requested ids, the run stops with the reason counts and the first 20 ids. Each split also reports `n_kept` and `n_dropped_by_reason`, and all id lists are sorted. Missing layout directories or split files are still a hard stop. Which split the single missing image belongs to is not yet known; the first GPU run records it.

Protocol:

- Official **val image IDs** for evaluation, official **train image IDs** for majority baselines. All train/val/test ID sets must be unique and mutually disjoint.
- Ground truth per `(image, class)` is the number of VOC boxes of that class, including difficult/truncated annotations. Validate class names, bounding boxes and the original 800×800 dimensions.
- **Assumption:** original DIOR VOC annotations exhaustively label instances within its 20 classes, making absent-class zero targets valid. This is not assumed for DIOR-RSVG.
- All 20 classes by default, or an explicit `--classes` subset using the repository's DIOR category names. DIOR does **not** have building/tree categories, so its results would not validate those app classes.
- Default **1,000 unique image/class pairs**, seed **17**, with **500 positive and 500 zero-target pairs**, balanced 1:1 within each class (25+25 per class with all 20 classes). Sampling does not change in response to detector outcomes. An insufficient class pool fails rather than silently shrinking the sample. This balanced benchmark does not represent natural class prevalence.
- The sample manifest includes pair identities, targets, prompts, XML/image hashes, train-annotation identity, split-file hashes, train-only modes, configuration and mirror ZIP hash. It is written and fsynced **before detector construction/inference**. An existing manifest can be reused only when the recomputed selection/provenance/config matches exactly.
- Infer proposals once per pair at the lowest score threshold. Reuse the production score/NMS policy for the `0.25/0.30/0.35/0.40` sweep. Report per-class and overall exact-match accuracy, MAE, within-±1 and n, plus per-pair evidence.
- Output both **always-0** and **per-class train-majority-count** baselines on the identical frozen val pairs. Train modes include zero counts for every train image/class pair; ties choose the smaller count. No val/test label is used to fit a baseline.

DIOR's dataset-level GSD varies roughly **0.5–30 m/px** and per-image GSD is not present in these VOC annotations. **This evaluator measures detector counting only, not the app's resolution gate.** It intentionally invokes the shared detector-count postprocessor without pretending that DIOR resolution is known.

## Kaggle GPU cell (supplied, NOT executed)

The cell is in `kaggle/run_count_dior_gpu.txt`; paste it into one cell and run it as a committed version. It uses `set -euo pipefail` with an ERR trap that prints the failing line, and it ends with `CELL DONE`.

Required:
- **Accelerator:** GPU (T4).
- **Internet:** on.
- **Code:** this branch's code as a Kaggle dataset at `/kaggle/input/satquery-count-grounding/sih2026`. Edit `REPO_INPUT` if your dataset is named differently.
- **Downloads:**
  - the DIOR mirror (pinned above);
  - `ShilongLiu/GroundingDINO/groundingdino_swint_ogc.pth` from Hugging Face. Its sha256 is **recorded, not pinned**, because no verified value exists yet. `configs/model_artifacts.json` has `sha256: null`.
  - `bert-base-uncased` from Hugging Face. The snapshot revision is **recorded, not pinned**.
  - PyPI `groundingdino-py==0.4.0` and `transformers==4.49.0`.

The ZIP, the extracted images (~15 GB together), the code copy and the HF cache all live in `/kaggle/tmp/count-dior`, which is not saved. Only JSON is written to `/kaggle/working`:
- `count-dior-provisioning.json`: checkpoint sha256 and BERT revision;
- `dior-count-val-seed17-1000.json`: the frozen sample;
- `dior-count-val-evaluation.json`: the results.

After the first run, copy the recorded checkpoint sha256 into `configs/model_artifacts.json` to pin it.

LR policy-control command, using the preserved B13 dataset mount:

```bash
python kaggle/eval_count_detection.py --mode lr_negative_control \
  --manifest /kaggle/input/satquery-rgb-vqa-b13-v11/data/manifests/rsvqa/pilot.v1/validation.jsonl \
  --config configs/count_grounding.json \
  --output /kaggle/working/lr-count-gate-negative-control.json
```

## Unmeasured and open risks

No detector-count accuracy, threshold optimum, GPU latency/memory, DIOR mirror byte equivalence to the unavailable original host, or real-image resolution-gate performance is claimed. The mirror download and hash were verified on Kaggle CPU (13 Nov 2026). No CUDA evaluation has run, and the GPU cell is unexecuted. The mirror is 9 images short of the paper. The Grounding DINO checkpoint hash and BERT revision are not pinned yet. Whether `groundingdino-py` builds its CUDA ops on the Kaggle image is untested; without them it falls back to slow CPU ops. A hash records exactly which mirror bytes were used; it is not an independent endorsement of their provenance. Exhaustiveness is an explicit annotation assumption. Class physical sizes, unknown/default classes, metadata quality, spatial qualifiers, dense objects and NMS behavior need further measurement. The DIOR class set and variable unknown per-image GSD limit transfer claims to other app counts.

## Test evidence

Baseline `f28b862`: **1,047 passed, 5 failed, 40 setup errors**, 28 subtests passed. Same interpreter and command in the same worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/soham/sih2026-app-b13/.venv/bin/python \
  -m pytest -p no:cacheprovider -q
```

Synthetic CPU tests cover 10 m refusal, 0.3 m detector routing, unknown-GSD labels, metadata-only tokens, projected metre versus geographic/feet CRS, preview downsampling, DIOR disjoint image splits, exhaustive counts, train-only modes, balanced deterministic selection, immutable sample manifests, missing-layout/annotation errors, and LR policy control. Actual evaluator entry points were not run.

The first full run after this change additionally hit the unchanged timing-sensitive `scripts/test_stage1_full.py::test_sharded_smoke_produces_full_accepted_throughput` (1 step completed vs 2). It passed in isolation immediately afterward. The final run and any recurrence are recorded below; no training infrastructure was edited to hide it.

### Persistent pre-existing fixture failures/errors

The five fixture failures and 40 setup-error node IDs from the earlier count work remain:

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

### Final comparison

Final full suite: **1,074 passed, 5 failed, 40 setup errors**, 28 subtests passed. All 45 persistent failure/error node IDs match the `f28b862` baseline exactly. The isolated timing-sensitive stage1 test passed, and did not fail in the final full run. No protected fixture or training code was altered. `git diff --check` passed. No CUDA evaluator or notebook provisioning cell was executed.
