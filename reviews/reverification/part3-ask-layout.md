# Part 3 — Ask-tab layout re-verification

Branch `part1-golden-demo-review`, HEAD `a62c169`; full scope baseline `345b4a7`. App code unchanged during this audit.

## Quoted requirements

From the prompt pack:

> LEFT column: scene image, scene ID (st.code), sensor, GSD.

> RIGHT column: question, Ask button, answer, execution-mode badge (from Part 2 ...), model name.

> BELOW both columns: evidence section, trace verification button.

> Golden scene: display the real GSD from the matched golden row (0.3 m in the current artifact, but read it from the data — do not hardcode "0.3").

> Uploaded scene with unknown resolution: display literally "unknown". Do not guess, do not default to any numeric value.

> Keep all existing functionality (radio between golden/upload, file uploader, Ask button behavior, trace verify button) working exactly as today — this is a layout change, not a behavior change.

## Line-by-line requirement check

| Lines in app.py | Operation | Finding |
|---|---|---|
| 176–181 | Source selector above the columns. | Satisfied; shared input-mode selector remains full width. |
| 182–184 | Creates two equal columns; initializes existing temporary-path variable. | Satisfied; no new session-state key. |
| 185–191 | Gets the pinned row, scene/sensor, disabled question in right column, GSD from row, and local image path. | Satisfied. GSD is data-derived. |
| 192–195 | Golden image fills left-column width, or existing missing-image info is shown. | Satisfied. Width changed from fixed 520 px to column width. |
| 196–207 | Upload widget and image preview in left; question in right; sensor selector; upload filename; unknown GSD. | Satisfied, with added upload preview rather than only relocation. Actual upload selection is mocked in the saved test. |
| 208–211 | Scene ID code, sensor caption, GSD caption. | Satisfied. Minor duplication/order difference: the upload sensor selector appears before these captions, so sensor appears twice and once before scene ID. |
| 213–238 | Right-column Ask validation, routing, session-state assignment and temp-file cleanup. | Satisfied. Moving the block did not alter its control flow. |
| 240–268 | Right-column notice, Answer heading, badge/provenance, model name and answer. | Satisfied placement. The badge precedes the answer, preserving the earlier Part 2 requirement; the pack's list puts answer first. This is an ordering difference, not missing content. |
| 270–320 | Full-width evidence section after leaving both outer columns. | Satisfied Part 3 placement. Its own inner columns do not change the outer layout. |
| 322–332 | Full-width explainer and Verify trace button/results. | Satisfied. |

## Focused test run

`python3 -m unittest demo_gui.test_golden_path.GoldenPathTests.test_scene_metadata_and_uploaded_preview_use_input_column -v`

Result: **1 run, 1 passed, 0 failures, 0 errors, 0.508 s**.

This checks golden GSD/sensor, golden image presence, right-column question, unknown upload GSD, absent preview before upload, filename and preview after a mocked upload. The image accessor was repaired this audit using singular `image` with support for this installed runtime's legacy `imgs` elements.

Supplemental read-only AppTest check: after a real cached-path Ask, model caption was in the right column; evidence JSON and Verify trace were outside both columns. Assertions passed.

## Reproduced defect and coverage limits

After a cached answer, switching the source selector to Upload leaves:
- left scene label: `No scene uploaded`;
- right Answer still visible;
- answer trace scene: `loveda_LoveDA_images_png_0_gsd0.3`.

The source-mode switch does not clear the previous response. This behavior also exists in the baseline's session-state handling; it is not newly introduced by columns, but the juxtaposed layout can make the old answer look associated with the new input.

The saved layout test switches before asking, so it cannot catch this defect. This is a coverage gap, not evidence that its image assertions never run. No app fix was authorized or applied during this audit.

The test also does not exercise the real file-uploader interaction, an uploaded Ask execution, full-width proof placement, or model placement. Those last two were checked separately in this audit, not added as saved tests. Its golden `0.3` expectation is a fixed regression expectation rather than a dynamic artifact comparison.

Browser screenshots were unavailable; layout was checked through the Streamlit element tree.
