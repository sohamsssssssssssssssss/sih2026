# Part 4 — Evidence presentation re-verification

Branch `part1-golden-demo-review`, HEAD `a62c169`, baseline `345b4a7`. No app changes in this audit.

## Quoted requirements and the spec conflict

Latest prompt pack:

> Group the same underlying data into four labeled sections: Identity (scene_id, sensor), Execution (model_name, model_version, timestamp), Question (the question text), Integrity (record_hash, prev_hash).

> Truncate the hash values visually ... but keep the full value accessible — either on hover/tooltip if Streamlit supports it cleanly, or in the still-present raw JSON view.

> Keep the raw JSON available, but demote it into its own collapsed expander titled "Raw evidence JSON" (separate from the four grouped sections, which should be visible without expanding).

> Add a short, one-or-two-sentence visible explanation of what the hash-chain integrity means.

The earlier scoped Part 4 said:

> Inside the "Evidence and execution trace" expander, replace the single st.json(evidence_fields(...)) call with a card/metric-style layout ...

It also suggested “Identity row: model_name, model_version, scene_id, sensor” and explicitly supplied the static explainer now used.

Therefore the two requests conflict on default visibility and model grouping. This report does not treat compliance with the earlier request as compliance with the latest one.

## Line-by-line requirement check

| Lines in app.py | Operation | Finding |
|---|---|---|
| 140–159 | Evidence extraction retains all ten supported keys when present. | Satisfied; no loss of underlying evidence. |
| 270–272 | Cards live inside an outer expander, collapsed by default. | **Deviation from latest pack:** four groups are not visible without expanding. Satisfies the earlier scoped request. Real AppTest confirmed `expanded=False`. |
| 273–282 | Identity contains model, version, sensor and scene ID. | **Grouping deviation:** model/version belong to Execution in latest pack. Earlier grouping explicitly allowed this. |
| 284–293 | Execution contains mode, timestamp and optional artifact. | Data preserved; model/version absent from this group but shown elsewhere. Mode/artifact are real trace fields, not invented values. |
| 295–296 | Question caption and full question in `st.text`. | Satisfied. |
| 298–317 | Integrity caption; two bordered metrics; long hashes abbreviated first 8 + last 8; full values in help text. | Satisfied. Missing hashes read “Not recorded”; empty previous hash reads “First record”. An empty record_hash is filtered out by evidence_fields, so it does not normally get mislabeled as a first record. |
| 319–320 | Nested Raw evidence JSON expander, explicitly collapsed, untruncated dictionary. | Satisfied raw-data access. It requires opening the outer evidence section too. |
| 322–325 | Exact earlier explainer outside all evidence expanders. | Satisfied visibility and requested meaning. Literal latest-pack phrasing “Use only these terms” could call for an explicit approved title such as Execution Trace; current sentence has no such title but contains none of the forbidden labels. |
| 326–332 | Existing verification button and result display. | Unchanged. |

## Focused test run

`python3 -m unittest demo_gui.test_golden_path.GoldenPathTests.test_evidence_cards_preserve_full_hashes_and_raw_fields -v`

Result: **1 run, 1 passed, 0 failures, 0 errors, 0.727 s**.

The test exercises three synthetic hash cases, metric values, full tooltips, full code values, question text, exact raw JSON equality, raw expander collapse, and explainer outside the evidence section.

Supplemental real cached-Ask probe:
- outer evidence expander: collapsed;
- Raw evidence JSON: collapsed;
- groups present;
- no “chain-of-thought”, “model reasoning”, or “hidden reasoning” found in rendered markdown, text or captions.

## False-positive risk / limits

The saved test passes while the latest visibility/grouping requirements are violated. It never asserts that groups are visible without expanding or that model/version are specifically under Execution. Its synthetic session response also means it is not itself testing a real invocation or cryptographic validity.

The live and cached tests additionally check evidence JSON mode/provenance; they do not close those presentation gaps.

Integrity is metadata-chain verification, not proof of answer correctness. Inspection of the unchanged trace implementation shows Verify trace checks the in-memory chain; the record does not include the answer payload. The mandated explainer should not be interpreted as signed provenance or verification of persisted-file tampering. No trace changes were made.

Verdict: data availability and hash presentation pass; latest-pack default visibility and field grouping do not. These deviations are reported, not silently fixed.
