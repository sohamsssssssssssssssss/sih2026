# Re-score note — Stage 0 RSVQA-LR baseline

`qwen2.5vl-3b__rsvqa__20260903T175900Z.json` was scored under the matcher as
it stood **before** `daab09b`. The figures inside that file are the figures
that were measured on the day, and they have deliberately not been edited:
the file records what was run, not what we currently believe.

This note records the difference between the two, so a reader who finds
0.5131 in the JSON and 0.5142 in the README knows why without re-deriving it.

## What changed in the scorer

`daab09b "fix: rescore ladder with shared answer matcher"` made two changes:

1. `answer_matches()` gained a rule: a raw prediction of `"1"` now counts as a
   match for gold `"yes"`. A model answering "1" to a yes/no question does
   mean yes, and the previous matcher refused it credit.
2. The `ladder` suite stopped being scored by exact string equality and now
   routes through `answer_matches()` like every other suite. This affects
   ladder runs only and has no bearing on the RSVQA baseline below.

## Effect on the committed baseline

Re-scoring the 10,004 stored predictions from that JSON under the current
matcher:

| metric | as measured (pre-`daab09b`) | current matcher |
|---|---|---|
| accuracy | 0.513095 | **0.514194** |
| binary_accuracy | 0.665517 | **0.667098** |
| open_accuracy | 0.165080 | **0.165080** (unchanged) |
| pred_yes_rate_on_binary | 0.3635 | 0.3675 |

**11 samples flipped.** Every one was gold `yes` with prediction `"1"`, and
every flip was `False -> True` — the new rule only grants credit, it never
revokes it. All 11 sit in the binary stratum, which is why `open_accuracy`
did not move at all.

`pred_yes_rate_on_binary` shifts too, because the degeneracy guard delegates
yes-detection to `answer_matches()` rather than keeping a second normaliser.
At 0.3675 it remains far below the 0.85 threshold, so the run is a valid
measurement under either matcher.

## No re-inference was performed

These are deterministic re-scores of stored predictions. The model was not
re-run, no GPU time was spent, and the predictions themselves are byte-identical
to what came off the T4. Only the function deciding correct-vs-incorrect changed.

## Keeping this honest

`eval/test_baseline_rescore.py` asserts that re-scoring this JSON with the
current matcher still yields 0.514194. If someone revises scoring again, that
test fails and names this note — the baseline gets restated deliberately
instead of drifting until somebody notices.
