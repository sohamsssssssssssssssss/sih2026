# Follow-up fixes and reproducibility

The five part2-through-part6 reports in this directory are unchanged historical
audit reports for a62c169, not claims about the fixed revision.

This follow-up fixes:

- Part 3: switching query source clears the old answer, notice and provenance.
  Previously, golden-to-upload after Ask displayed the golden answer beside
  "No scene uploaded", despite the displayed source no longer matching it.
- Part 4: Identity (scene ID/sensor), Execution (model/version/mode/timestamp/
  artifact), Question and Integrity appear outside any expander. Only Raw evidence
  JSON is collapsed. The evidence regression checks default visibility and field
  grouping as well as full hashes and raw values.
- Part 5: zero flagged rungs produce no flag scatter, legend entry, starred ticks,
  marker explanation or warnings. A synthetic empty-flags regression is saved.
- Part 6: a missing Mumbai heading produces a clear annotation-section error,
  with a saved regression rejecting the raw Python index error.

The image-element compatibility helper and positive image controls are included.

Run from the repository root:

```sh
python3 -m unittest discover -s demo_gui -t . -v
```

Validation: 24 tests passed on Python 3.14 / Streamlit 1.52.2. The four defect
regressions also ran against a62c169's app source via AppTest.from_string, with
ROOT adjusted to this checkout: all four failed by assertion (six failures
including the evidence test's three hash subcases), with zero test errors.
An initial negative-control harness omitted that ROOT adjustment and failed
to render the app; it was corrected before collecting the above result.

No fresh browser screenshot verification or real GPU inference was performed.
Other historical coverage limits in the original reports remain unless explicitly
addressed above. This is not completion of the later Part 8 failure-path review.
