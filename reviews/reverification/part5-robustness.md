# Part 5 — Robustness presentation re-verification

Branch `part1-golden-demo-review`, HEAD `a62c169`; baseline `345b4a7`. App and scientific artifact unchanged during this audit.

## Quoted requirements

Prompt pack:

> Chart: plot open_accuracy as the primary, visually prominent line. Plot accuracy as a secondary/reference line ... on the same chart.

> Degenerate rungs (from ladder_report["degenerate_rungs"], read dynamically — never hardcoded) get a distinct marker style on the chart itself ... in addition to the existing st.warning() text below.

> Table: add an "Open accuracy" column, keep "Accuracy" and "Binary predicted-yes rate", and flag degenerate rows visibly in the table too ...

> Do not change the underlying results JSON, the degeneracy-detection logic in eval/, or any other file outside demo_gui/app.py.

Its TESTS paragraph separately authorizes `demo_gui/test_robustness_tab.py` and requires a synthetic zero-degenerate-rungs check. The earlier scoped prompt also requires the visible answer-collapse explanation.

## Line-by-line requirement check

| Lines in app.py | Operation | Finding |
|---|---|---|
| 334–339 | Header and artifact model/sample/run metadata. | Satisfied, data-derived. |
| 340–343 | Sorts original string keys by numerical GSD. | Satisfied; preserves exact string keys for flag membership. |
| 344–347 | Reads stored flags, GSD, open_accuracy and accuracy. | Satisfied; no recomputed threshold or hardcoded flagged rungs. |
| 348–351 | Explanation of misleading aggregate accuracy near chart. | Satisfied. No claim of coarse-resolution recovery. |
| 352–356 | Blue open-question line, width 3, circle markers and clear label. | Satisfied. |
| 357–360 | Gray dashed aggregate reference, width 1.2, answer-collapse-sensitive label. | Satisfied. |
| 361–369 | Orange X overlays on both series at stored flagged keys only. | Satisfied. Empty scatter objects are still constructed when no flags exist. |
| 370–378 | Axes, starred flagged ticks, fixed accuracy axis, legend. | Satisfied. No smoothing, error bars or invented intervals. |
| 379–381 | Chart and marker explainer. | Satisfied for current artifact; static flag explainer remains for empty flags. |
| 382–392 | Table includes GSD, status, open accuracy, aggregate accuracy and binary yes rate. | Satisfied semantically. Column is named “Open-question accuracy”, not the pack's shorter “Open accuracy”. |
| 393–399 | Four-decimal display configuration and full-width table. | Current values represented exactly to their stored precision. The code would round display for future data with more than four decimals; underlying DataFrame and plot retain original floats. |
| 401–404 | Original per-rung stored warnings. | Satisfied and unchanged. |

## Actual artifact values re-read

| GSD | open_accuracy | accuracy | pred_yes_rate_on_binary | Flag |
|---|---:|---:|---:|---|
| 0.3 | 0.335 | 0.5625 | 0.64 | No |
| 1.0 | 0.29 | 0.505 | 0.5 | No |
| 2.0 | 0.275 | 0.425 | 0.315 | No |
| 5.0 | 0.245 | 0.465 | 0.975 | Yes |
| 10.0 | 0.27 | 0.485 | 1.0 | Yes |

These are measured stored scores, not new experiments. The final open value is slightly above the 5 m value; no UI claim generalizes that small change into restored robustness.

## Focused test run

`python3 -m unittest demo_gui.test_robustness_tab -v`

Test: `RobustnessTabTests.test_chart_and_table_show_exact_metrics_and_artifact_flags`.

Result: **1 run, 1 passed, 0 failures, 0 errors, 0.386 s**.

It renders the real artifact, compares all table metric lists, flag status and warning strings, captures the actual Matplotlib figure, compares both line datasets/widths/styles, ticks and both sets of marker coordinates.

Supplemental isolated AppTest probe replaced only the in-memory artifact read with the same data and `degenerate_rungs=[]`. It passed assertions for zero flagged rows, zero flagged points/ticks and zero warnings. The file was not modified.

## Deviations / coverage gaps

The synthetic empty-flags regression required by the latest pack is not saved in the repository test suite. This audit ran it as a diagnostic; it is not counted as a new saved unittest.

Minor presentation issue: even with zero flagged rungs, the legend still contains “Flagged degenerate rung” and the static marker explainer remains. No actual rung is falsely flagged. The real-artifact test does not catch this because its fixture always has flags.

Verdict: current scientific series, values and dynamic flagging satisfy the functional requirements. Saved zero-flags coverage and the empty legend's presentation remain gaps; no changes made during the review.
