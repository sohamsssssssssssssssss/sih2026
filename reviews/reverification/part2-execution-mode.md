# Part 2 — Execution-mode badge re-verification

Reviewed branch `part1-golden-demo-review`, HEAD `a62c16931cbf72874b0c52263a16e25e6aa12055`, against main baseline `345b4a7`. App code was not changed during this audit. Line numbers below refer to current `demo_gui/app.py`.

## Quoted requirements

From the latest prompt pack:

> After a response is produced, read response["trace"]["params"]["execution_mode"] and render an explicit, visually distinct badge: LIVE vs CACHED.

> When in cached mode, also surface which results artifact was used (trace["params"]["results_artifact"] already carries this — just display it, don't invent new fields).

> If execution_mode is missing or an unrecognized value, fail loudly in the UI (st.error with a clear message) — do not silently default to either LIVE or CACHED display.

> Do not change execute_query()'s or cached_response()'s actual logic for deciding live vs cached — only change how the UI reads and displays the mode that was already decided.

The earlier scoped Part 2 additionally required: “Add execution_mode and results_artifact (when present) to the dict returned by evidence_fields()”; “Keep the existing notice”; and “response can be None (error path) — guard the badge rendering accordingly.”

## Line-by-line requirement check

| Lines | Current operation | Finding |
|---|---|---|
| 72–103 | Cached response records `cached_result` and a relative artifact path. | Satisfied. AST-identical to baseline; no invented new mode. |
| 106–137 | Live routing passes `execution_mode: live`; existing fallback decides the cached path. | Satisfied. AST-identical to baseline. |
| 140–159 | Extracts both mode and artifact alongside existing evidence fields; omits absent/empty optional fields. | Satisfied, additive return keys. |
| 240–243 | Reads session response and obtains mode from the response trace params, guarding a missing response. | Satisfied. It does not derive state from notice text. |
| 244–252 | Explicit error notice remains an error; cached notices are warnings; live notices are successes; unknown-mode notices are errors. | Satisfied. The former prefix heuristic is absent from app code. Tests still use a prefix to locate the existing warning, which is a test selector, not production state inference. |
| 253–258 | Response guard; Answer heading; green success LIVE banner or blue information CACHED banner. | Satisfied at element/placement level. The banner appears before the answer and outside the evidence expander. |
| 259–263 | Shows relative artifact in a code element; reports missing cached provenance as an error. | Satisfied. The missing-provenance check is additional display validation, not execution selection. |
| 264–265 | Explicit error for unknown/missing mode. | Satisfied. Neither known badge is guessed. |
| 266–268 | Model label, answer, existing confidence-calibration caption. | Preserved presentation. Unknown mode still allows the answer text alongside the error; the spec requires a loud error, not suppression of the answer. |

## Tests run separately for this part

All are in `demo_gui.test_golden_path.GoldenPathTests`:

- `test_live_success_renders_live_badge`
- `test_no_gpu_golden_path_uses_verified_cached_result`
- `test_missing_local_pixels_uses_verified_cached_result`
- `test_missing_or_unknown_execution_mode_renders_error` (missing and unknown subcases)
- `test_no_response_renders_error_notice_without_badge`

These five test IDs were supplied explicitly to `python3 -m unittest ... -v`.

Result: **5 run, 5 passed, 0 failures, 0 errors, 1.256 s** on Streamlit 1.52.2 / Python 3.14.

## Limits and deviations

No functional Part 2 deviation found. The live-success test mocks `QwenVLModel.infer`: it verifies route/trace/UI plumbing, not real GPU inference. Cached fallback and mode extraction are exercised.

Coverage gaps: no saved test uses deliberately misleading notice text to independently reject a reintroduced prefix heuristic, and no saved test targets a cached response lacking provenance. Inspection confirms correct current behavior; passing tests alone do not prove these cases.

No browser was available for a fresh pixel-level visual review. Distinct primitives and placement were checked through source/AppTest; they should not be reported as screenshot-verified.
