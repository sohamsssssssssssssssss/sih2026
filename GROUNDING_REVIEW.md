# Grounding Migration Review

Independent, read-only review of the Grounding DINO capability (provider, routing, evidence, fail-closed paths, tests). Findings are prioritized: **P0** = correctness/data integrity, **P1** = important robustness, **P2** = useful improvement.

## Existing architecture

- `models/grounding_dino/model.py` — lazy wrapper over `groundingdino.util.inference` (`load_image`/`load_model`/`predict`). Config resolves once via explicit arg → `SATQUERY_GROUNDING_CONFIG` → packaged `groundingdino/config/GroundingDINO_SwinT_OGC.py`; the resolved path is reused by `readiness()` and `_load()` (tested in `models/grounding_dino/test_model.py::test_readiness_and_load_use_same_resolved_config`). Deliberately CUDA-only; post-load parameter-device verification.
- `models/artifacts.py` — offline artifact lookup from `configs/model_artifacts.json`; enforces nonempty checkpoint with the exact manifest filename; `revision`/`sha256` are currently `null` (unverified).
- `orchestrator/registry.py` — name-keyed singleton registry; `GroundingDINOModel()` registered as `grounding-dino-swint` at import time.
- `orchestrator/capabilities.py` — capability→provider binding (`GROUNDING` → `grounding-dino-swint`), conflict-detecting `register_provider`, `require_provider_ready` gating before any execution.
- `orchestrator/planner.py` — deterministic grounding-intent rules; combined temporal+localization intent produces a `change_vqa → grounding` two-step plan.
- `orchestrator/executor.py` — executes exactly one available step; the two-step chain is representation-only and raises `_SINGLE_STEP_ONLY`.
- `orchestrator/router.py` — validates the model result (`_validate_result`), forbids forged planner/plan metadata, appends the hash-chained trace (`orchestrator/trace.py`).
- `backend/services.py` — live/cached boundary. Live: `analyze_scene` → `execute_plan` → `route`. Cached: pixel-hash scene identification (`backend/scene_pack.py::cached_scene`) → `_curated_cached_response`, which strictly validates grounding evidence against `selected_prediction`. Unavailable providers are recorded truthfully via `_record_unavailable` (`model_name: "not-executed"`).
- `eval/suites/grounding_dior_rsvg.py` — DIOR-RSVG eval (seed 26167, stratified), committed artifact `results/grounding-dino-swint__dior-rsvg__20260911T084941Z.json`; replayed via `data/demo/manifest.json` (`dior-rsvg-07272`).
- `scripts/gpu_smoke.py` — GPU-only live smoke with the strictest evidence validator in the repo (`validate_result`).

## Risks

- **P0 — Live path never validates grounding evidence.** `orchestrator/router.py::_validate_result` checks only that `answer` is a non-empty string and `evidence` is a list. Item structure (type, coordinate range/order, confidence range/finiteness) is unchecked on the live route. The cached path (`backend/services.py::_curated_cached_response`) and the smoke path (`scripts/gpu_smoke.py::validate_result`) both validate deeply — the live route is the weakest, and the only user-facing, link. Any provider bug surfaces as authoritative evidence in the API response and trace.
- **P0 — NaN/Inf scores and boxes are silently clamped into plausible-looking output.** `models/grounding_dino/model.py::infer` clamps via `max(0.0, min(1.0, ...))`. In Python, `min(1.0, nan) == 1.0`, so a NaN tensor (corrupt/poisoned checkpoint — plausible because the checkpoint checksum is unverified) becomes `[1.0, 1.0, 1.0, 1.0]` with `confidence: 1.0` — a full-image box at maximum confidence. The clamping converts corruption into confident fabrication instead of failing closed.
- **P1 — `groundingdino-py` is not declared in `requirements.txt`** (only documented in `docs/gpu-smoke.md` and installed ad hoc by `kaggle/run_grounding_dior_rsvg.py`). An environment built from requirements fails readiness (`DEPENDENCY_UNAVAILABLE` — fail-closed is correct, but the capability silently vanishes).
- **P1 — Failure classification by exception-string matching.** `backend/services.py::analyze_scene` maps exceptions to `ModelUnavailable` vs `ModelExecutionError` by substrings (`"CUDA GPU"`, `"requires groundingdino"`, …). E.g. `torch.cuda.OutOfMemoryError` ("CUDA out of memory") matches none → 502 instead of 503. Any wording change during migration silently reclassifies failures.
- **P1 — Package/API drift in `groundingdino-py` forks.** `models/grounding_dino/model.py` calls `predict(..., device="cuda")` and expects string phrase labels. Readiness cannot detect a signature mismatch across fork versions; only a live GPU smoke run can. Migration must pin the exact version verified on Kaggle (`groundingdino-py==0.4.0`).
- **P2 — Manifest provider-set tripwire is brittle for migration.** `models/artifacts.py::load_manifest` raises unless providers are exactly `{"qwen2.5vl-3b", "grounding-dino-swint"}`. Adding any future provider without updating this in lockstep makes *all* artifact validation fail → every capability unavailable.
- **P2 — Sole inference worker is shared and blocking.** `orchestrator/router.py` uses a `max_workers=1` executor; a timed-out (abandoned, not cancellable) grounding job keeps the GPU busy and blocks all subsequent providers.
- **P2 — Singleton `_load` is not thread-safe.** Safe through the router's single worker, but direct concurrent callers (e.g., shared with eval scripts) could double-load.

## Missing tests

All in `models/grounding_dino/test_model.py` unless noted; all are CPU-safe.

- **P0/P1 — No live-path evidence validation test.** `backend/test_api.py::test_grounding_analysis_returns_normalized_bounding_box_evidence` uses valid evidence. Add a test asserting the router rejects malformed grounding evidence (non-finite confidence, coords outside [0,1], `x1 ≥ x2`, wrong `coordinate_space`) once validation exists — and a test documenting today's behavior otherwise.
- **P1 — Input-validation branches of `infer` are untested:** more than one image path (`ValueError: requires exactly one image path`), empty/whitespace question, nonexistent image path (`FileNotFoundError`).
- **P1 — NaN/Inf handling is untested** (and currently wrong; see P0 above). A test would have caught the clamp-to-1.0 behavior.
- **P1 — `zip(..., strict=True)` mismatch** (boxes/scores/labels length divergence) raises `ValueError` — untested.
- **P2 — `ARTIFACT_CONFIG_INVALID` branch for grounding** (`models/artifacts.py`: checkpoint filename differing from manifest) has no test.
- **P2 — `load_manifest` provider-set tripwire** (extra/missing provider) untested in `models/test_artifacts.py`.
- **P2 — Multi-box curated replay** — `backend/services.py::_curated_cached_response` only accepts exactly one evidence box for grounding (`evidence[0] if len(evidence) == 1 else None`); a curated row with 2+ boxes fails closed. Undocumented and untested.
- **P2 — Label typing:** `str(label)` would happily stringify a tuple from a fork's `predict` (` "('the', 'ship')"`) and `gpu_smoke.validate_result` only checks `isinstance(str)`. No guard/test.

## Edge cases

- **P2 — The full user question is passed as the DINO caption.** `models/grounding_dino/model.py::infer` does not normalize the caption (lowercase, trailing period, strip imperative words). "Locate the buildings in this image." becomes the phrase "locate the buildings in this image", degrading grounding quality silently. Note the cached demo distinguishes `question` ("Locate a yellow ship.") from `evaluated_expression` ("A yellow ship") — a live run of the same question uses a different caption than the measured artifact, so results can legitimately differ.
- **P2 — Zero detections** are handled truthfully (`"No match found for '…'."`, empty evidence) and covered by tests — good.
- **P2 — Post-clamp degenerate boxes** (`[x, y, x, y]` or all-1.0) are never checked for `x1 < x2` on the live path; the smoke validator and cached-path validator both require strict ordering — three validators, three standards.
- **P2 — Buffer (non-parameter) tensors are not device-verified** in `_load` (`models/grounding_dino/model.py` verifies only `parameters()`); a misplaced buffer surfaces as an opaque device-mismatch error at predict time.
- **P2 — Stale config resolution is deliberate**: `SATQUERY_GROUNDING_CONFIG` is read once per instance (tested); callers reusing the singleton across env changes keep the first resolution.
- **P2 — `evaluate_compatibility` for `grounding`** (`data/pairing.py`) appends `verified_checks: ["source_format_supported"]` without checking anything — a claimed-but-unperformed check.
- **P2 — `scripts/verify_phase0.py` hard-codes grounding → 503**; on any future GPU-provisioned host this golden check flips, confusingly.

## Truthfulness / provenance concerns

- **P1 — Model identity is asserted, not verified.** `configs/model_artifacts.json` has `revision: null, sha256: null`; `models/artifacts.py` issues `unverified-revision` identity, yet traces and cached replays record `ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth` as the version. Any checkpoint that loads passes; the trace claim is stronger than the evidence. (Same gap exists for Qwen; `docs/gpu-smoke.md` already acknowledges it.)
- **P0 — Clamping converts corruption into confident output** (see Risks) — the most direct "fabricated output" pathway found.
- **P2 — Grounding `confidence` is an uncalibrated phrase-similarity logit**, and `box_threshold=0.35` filters low-confidence boxes before output (survivorship bias): surfaced evidence is by construction above-threshold. Not documented near the provider.
- **P2 — Live traces don't record `box_threshold`/`text_threshold`.** The eval artifact records them (`configuration` block); a live result cannot be tied to the thresholds that produced it. Changing constructor defaults during migration would silently change live behavior with no provenance trail.
- **P2 — Qwen-side (adjacent, focus item 7):** `models/qwen_vl/model.py` returns `confidence: None` always — honest, and tests assert `"confidence" not in payload`; keep grounding's per-box confidence from leaking to a top-level field. Separately, `max_new_tokens=50` truncation is undetected (no `finish_reason` check), so a truncated answer is presented as complete.
- **Good:** cached replay requires exact scene pixel-hash + exact question, exact `selected_prediction` equality, and labels the response `CACHED REAL … No live model ran.`; `_record_unavailable` writes `model_name: "not-executed"` rather than pretending execution.

## GPU/runtime concerns

- **Deliberate CUDA-only policy is well implemented**: readiness gate (`CUDA_UNAVAILABLE`), `_load` re-check ("CPU fallback is disabled"), post-`load_model` parameter-device verification, and a CPU-safe test for each. Keep all of these intact in the migration.
- **P1 — Verify `load_model`/`predict` signatures against the pinned fork version** (see Risks); readiness cannot catch drift, and the only guard is the GPU smoke run.
- **P2 — fp32 Swin-T on T4** is comfortably small; no OOM handling beyond the bounded, single-line `load_model` failure message (well tested in `test_load_preserves_bounded_checkpoint_failure_reason`).
- **P2 — Timed-out inference is abandoned, not cancelled** — the GPU stays busy until the model finishes; documented "ponytail" in `orchestrator/router.py`.

## Recommended follow-up work

1. **P0** — Validate grounding evidence items on the live route: mirror `scripts/gpu_smoke.py::validate_result` strictness (finite, `0 ≤ v ≤ 1`, `x1 < x2`, `y1 < y2`, string label, `coordinate_space`) in `orchestrator/router.py::_validate_result` or in a shared validator used by router and smoke.
2. **P0** — Fail closed on non-finite scores/coordinates in `models/grounding_dino/model.py::infer` (raise before clamping) instead of clamping NaN → 1.0.
3. **P1** — Declare `groundingdino-py==0.4.0` in `requirements.txt` (or a pinned extras group) so grounding availability is reproducible.
4. **P1** — Replace exception-string classification in `backend/services.py` with typed exceptions from the model wrappers (e.g., `ProviderEnvironmentError`), keeping the `ModelUnavailable`/`ModelExecutionError` semantics.
5. **P1** — Pin `revision` + `sha256` for the checkpoint in `configs/model_artifacts.json` and surface the verified artifact identity in traces.
6. **P1** — Add CPU-safe tests for: `infer` input validation (multi-image, empty question, missing file), strict-zip mismatch, NaN/Inf rejection, and the router-level malformed-evidence rejection (`models/grounding_dino/test_model.py`, `backend/test_api.py`).
7. **P2** — Record `box_threshold`/`text_threshold` in live trace params; document that grounding confidence is uncalibrated and threshold-filtered.
8. **P2** — Normalize or document caption handling (leading imperatives, casing) in `models/grounding_dino/model.py`, and document the `question` vs `evaluated_expression` distinction for the cached demo scene.
9. **P2** — Add tests for the `load_manifest` provider-set tripwire and the grounding `ARTIFACT_CONFIG_INVALID` filename branch (`models/test_artifacts.py`).
10. **P2** — Decide and document the multi-box curated-replay policy in `backend/services.py::_curated_cached_response` (currently fail-closed on 2+ boxes).
11. **P2** — Check buffer device placement alongside parameters in `_load`; rename the `source_format_supported` "verified check" in `data/pairing.py` to reflect that nothing is verified.
12. **P2** — During migration, keep the executor's one-step-only guard (`orchestrator/executor.py`) so the `change_vqa → grounding` plan stays representation-only; do not enable partial chain execution.
