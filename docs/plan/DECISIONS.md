# Decisions

## T13b CI deviation — approved by Soham before analysis

> T13b CI rule (T13b.md:31) needs per-patch predictions that weren't saved. Substitute: a CI-based criterion PASSES only if it passes in EACH of seeds 1, 2, 3 using that seed's own patch-bootstrap CI95 from its JSON. It FAILS if it fails in all three. Otherwise it is INCONCLUSIVE. This is stricter than the registered rule, so it cannot inflate a pass. Seed means use the plain mean of the seed-level values.

The preregistration in docs/hypotheses/T13b.md is unchanged. No rerun is
approved. Report the original CI rule and this deviation explicitly; never
average CI endpoints or invent a missing paired interval. Any inconclusive
registered criterion is referred back to Soham after the report. Saving future
prediction artifacts is a separate code-only task, not permission to run them.

## T13b registered bootstrap supersedes the substitute report

The substitute-rule results in commit `9672b78` are superseded by the
registered-bootstrap results in [the T13b report](../../results/t13b/REPORT.md).
The saved seed 1–3 outputs from `adapt-run-3` (run code `0b581f7`) include
per-patch predictions. The analysis uses the preregistered bootstrap on those
predictions, including paired resamples for arm comparisons. P1, P2 and the
overall gate pass, and the bootstrap resolves rule (d). The decision above
remains the historical record for the earlier analysis.
