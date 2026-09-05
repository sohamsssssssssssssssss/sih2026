# Part 6 — SAR presentation re-verification

Branch `part1-golden-demo-review`, HEAD `a62c169`; baseline `345b4a7`. App/annotation/loader unchanged during this audit; the image assertion received a compatibility fix and positive control.

## Quoted original requirements

From the detailed scoped Part 6 request:

> Parse the Mumbai section into the four category buckets: Water, Built-up, Vegetation, Terrain ... Skip "Why it looks this way:" from the short-summary view ...

> Render four short summary cards/sections (st.columns or stacked, your call), each showing a CONCISE excerpt for that category ... Do not paraphrase or rewrite the analyst's actual words ... e.g. the Region/Class line(s) without the full Reasoning explanation, or the first sentence of each bucket.

> Place the COMPLETE, unmodified Mumbai section (current full markdown dump, byte-identical to what load_mumbai_interpretation() already returns) inside an expander labeled "View full analyst annotation".

> Add a clearly visible label, prominently, above or alongside the SAR image ... "HUMAN SAR VALIDATION — NOT AI MODEL OUTPUT" ... visible without expanding anything.

> Keep the existing missing-image fallback ... and the existing missing-annotation error handling ... If your new per-category parsing can fail ... fail safe with a clear message, not a stack trace.

The later pack's Part 6 describes “four category-excerpt columns” and five tests as supposed existing facts. Those facts do not match this branch: it has stacked cards and four tests. The original task explicitly allowed stacked cards. There is no supplied standalone first-sentence-only implementation in the `345b4a7` ancestor to compare byte-for-byte: that ancestor simply rendered the complete section.

## Line-by-line requirement check

| Lines in app.py | Operation | Finding |
|---|---|---|
| 40–47 | Cached loader extracts Mumbai section and strips its image-Markdown line. | Unchanged from baseline. “Full annotation” means this existing loader output, not literal entire Markdown source including scene heading/image. |
| 406–408 | Header and exact uppercase human-validation label in `st.info`. | Exact text and visible-before-image placement satisfied. |
| 409–416 | Existing local image or missing-image info. | Satisfied; fallback behavior preserved. |
| 417–421 | Loads annotation and catches OSError/IndexError. | Satisfied; no uncaught traceback. Missing Mumbai heading still produces the low-quality suffix “list index out of range”. |
| 423–428 | Explicit mapping for four categories. | Satisfied, Mumbai only. |
| 429–437 | Requires each category heading and synthesis heading exactly once and in order. | Intentional strictness beyond a permissive first-sentence policy. See tradeoff below. |
| 438–446 | For each bucket, selects Region/Class lines in source order. | Satisfied for current Water/Built-up. Built-up is a concatenation of verbatim source lines, not one contiguous substring because intervening Reasoning text is omitted. |
| 447–452 | Otherwise uses the first period-terminated excerpt and rejects empty categories. | Current Vegetation/Terrain excerpts are verbatim first sentences. General sentence splitting at the first period is simplistic for future abbreviations/numeric prose. |
| 453–454 | Reports category parsing failure and points to full annotation. | Safe, but all four summaries disappear if any one bucket fails. |
| 455–459 | Four stacked bordered cards with category subheaders. | Satisfies original “columns or stacked” choice; differs from later pack's claimed column layout. |
| 460–461 | Full original loader string in the named expander, even after summary parsing failure. | Satisfied; synthesis remains here. |

## Prominence assessment

A full-width colored information banner immediately below the tab header and above the image is an appropriate prominent scientific-origin label. `st.warning`/`st.error` would imply a warning/error condition; neither is required by the detailed spec. No type change is recommended solely to make the message honest.

This is source/AppTest placement assessment, not screenshot proof. Browser runtime discovery returned no available browser, so responsive clipping/contrast were not visually verified. The label is more prominent than a caption and exists outside any expander.

## Strict parsing tradeoff

The implementation deliberately rejects missing/duplicate/reordered headings instead of guessing which category owns the text. That reduces silent misclassification, but is brittle to harmless editorial changes and requires the synthesis heading even though synthesis is omitted from summaries.

It satisfies the detailed task's explicit safe-error allowance: full text remains accessible. It is NOT equivalent to “first sentence, verbatim, never fails,” and this audit does not retroactively declare that alternative behavior approved. If the latter is the authoritative policy, this is a deviation requiring a separately approved parser change.

## Focused tests run

`python3 -m unittest demo_gui.test_sar_tab -v`

- `test_real_annotation_has_verbatim_summaries_and_full_text`
- `test_missing_image_keeps_annotation_available`
- `test_missing_annotation_shows_error`
- `test_missing_category_fails_safely_and_preserves_full_annotation`

Result: **4 run, 4 passed, 0 failures, 0 errors, 0.622 s**.

The missing-image test now first asserts one real image exists, then mocks that image path missing and asserts zero. This prevents an unsupported image selector from passing the absence-only assertion vacuously.

Supplemental read-only probes passed for duplicate headings, reordered headings, empty Vegetation, and a missing Mumbai scene heading. All had zero AppTest exceptions and a UI error. Category failures kept the full expander; the missing scene header could not produce a full section.

## Test limits and scope

The excerpt test validates each displayed line against its source bucket, not contiguous-substring identity or complete semantic equivalence. It would not prove a first-sentence-only policy. There are no saved tests for duplicate/reordered/empty categories or missing Mumbai heading; this audit's probes do not count as durable coverage.

The annotation file, Ask tab, robustness tab, and extraction helper are unchanged relative to the start of this audit. The branch still contains the entire new Part 6 renderer versus main's original raw-text view; its scope must not be called Part 1-only.
