# T08 router validation — 2026-11-17

The 40 precommitted examples match their expected route/refusal: **40/40**;
14 refusals, no off-diagonal errors. These are synthetic regression cases with
fake tools, not estimates of model or open-ended language accuracy.

Spec-only commit: `ac5e0b23c0b3166b65cecf961ed8c6222e604512`.
T07 base: `7bfb1790ad92a8e63225d6475f1d2975ef0e43c2`.

## Route confusion table

Rows are expected, columns are actual. LC=land-cover; OS=optical-sar-deterministic;
RF=flood-otsu; CH=change-deterministic; CT=grounding-dino-swint; VL=qwen2.5vl-3b.
Refusal reasons are retained separately in the machine-readable report.

| Expected / actual | LC | OS | RF | CH | CT | VL | Refusal |
|---|---:|---:|---:|---:|---:|---:|---:|
| LC | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| OS | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| RF | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| CH | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| CT | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| VL | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Refusal | 0 | 0 | 0 | 0 | 0 | 0 | 14 |

## PROVISIONAL thresholds

Every use is recorded in the hash-chained trace. These are conservative guards,
not validated accuracy thresholds. Equality passes for cloud, nodata and OOD.

| Guard | Value | Source / limitation |
|---|---|---|
| India rectangle | [68, 6, 98, 38], EPSG:4326 | PROVISIONAL; spec ac5e0b2; not a country boundary; outside is allowed and flagged |
| Optical cloud | >50% | PROVISIONAL; spec ac5e0b2; majority-cloud guard; lower cloud does not establish reliability |
| Nodata | >50% | PROVISIONAL; spec ac5e0b2; majority-missing guard |
| Per-band mean shift | absolute shift >2 training standard deviations | PROVISIONAL; spec ac5e0b2; Rajasthan +3.18 motivates warning only; no empirical cutoff is claimed |

The GSD limit comes only from Query.needs_gsd_m. There is no universal GSD or
confidence threshold. No calibration map is fitted or shipped.

## Verification

- T07 precondition: `python3.11 -m pytest core -q -p no:cacheprovider`, 139 passed, 2 skipped.
- Final gate: `python3.11 -m pytest core -q`, **243 passed, 2 skipped**.
- Full suite, before and after: `python3.11 -m pytest -q -p no:cacheprovider --continue-on-collection-errors --junitxml=...`.
- Baseline: **1,069 passed, 7 failed, 12 collection errors, 3 skipped**.
- Final: **1,173 passed, 7 failed, 12 collection errors, 3 skipped**.
- Failure/error identities match exactly; no new failures. Both full runs exit 1.
- Hypothesis properties cover refusal invariants, unvalidated status/confidence,
  and determinism/no input mutation. Additional tests cover threshold boundaries,
  both temporal inputs, cloud rerouting and absent radar pixels, sensor/band
  checks, OOD uncertainty, malformed/error results, trace failure before execution,
  calibration identity/range/monotonicity, and schema compatibility.

[Machine-readable results](../../results/t08-router-validation.json) include all
40 expected/actual outcomes, warning labels, unchanged failure identities and
SHA-256 hashes of the baseline/final JUnit reports (local reports in /private/tmp).

Unchanged baseline failures/errors:

- error: `::data.test_bigearthnet`
- error: `::data.test_bigearthnet_pilot`
- error: `::demo_gui.test_capability_status`
- error: `::demo_gui.test_failure_paths`
- error: `::demo_gui.test_golden_path`
- error: `::demo_gui.test_offline`
- error: `::demo_gui.test_robustness_tab`
- error: `::demo_gui.test_sar_tab`
- error: `::scripts.test_stage1_full`
- error: `::scripts.test_stage1_resume_variant`
- error: `::scripts.test_stage1_revision_gate`
- error: `::scripts.test_stage1_t4_package`
- failure: `data.test_bigearthnet_shards::test_verifier_checks_source_pixels_captions_and_all_row_checksums`
- failure: `scripts.test_stage1_smoke::test_fp32_trainable_masters_and_complete_checkpoint[False-4]`
- failure: `scripts.test_stage1_smoke::test_fp32_trainable_masters_and_complete_checkpoint[False-None]`
- failure: `scripts.test_stage1_smoke::test_fp32_trainable_masters_and_complete_checkpoint[True-4]`
- failure: `scripts.test_stage1_smoke::test_fp32_trainable_masters_and_complete_checkpoint[True-None]`
- failure: `scripts.test_stage1_smoke::test_t4_package_entry_and_stages`
- failure: `scripts.test_stage1_smoke::test_t4_smoke_package`

## Integration and contract notes

`core.router.route(query, scene)` returns intent, ranked tool names and reason.
`core.gate.evaluate(query, scene, route, registry, ...)` is pure and returns a
GateDecision; it never runs a tool or writes a trace. `core.answer.decide(...)`
writes every route candidate and gate event through orchestrator.trace.
`core.answer.run_query(...)` uses that decision before invoking any tool and
assembles the Answer. Existing backend/orchestrator endpoints are not rewired by
this task; callers adopt this new core API explicitly.

run_query requires `trace_id`, a ToolRegistry, and `pixels_by_sensor`. Keys are
S1, S2, RGB_HR or S1S2; values are the existing adapter pixel inputs. A temporal
value is a date-ordered pair; an optical-SAR value is an optical/radar pair.
Cloud rerouting selects S1 pixels and refuses if those inputs are absent. Radar
targets must declare S1 support in ToolNeeds. Callers/loaders remain responsible
for pixel provenance, units, and the actual band/grid metadata; adapters retain
their pixel-level validation. Metadata alone does not certify those properties.

For OOD checks, pass training_stats={band: (mean, std)} and band_means=(means_t1,
means_t2) in the same units, from the selected tool's training distribution.
Missing/invalid learned-tool stats or means produce ood_unknown and low_confidence.
No distribution is invented for deterministic tools. A joint scene defaults to
S2 mode for land cover/change; monsoon mode switches to S1. This is an English
pattern router with finite coverage; the 40 cases are not a language benchmark.

Contracts are additively bumped to **1.2.0** for Evidence.needs_sensor,
Answer.evidence and explicit fixed refusal reasons; all v1.1.0 values remain
accepted. Generated JSON schemas are updated. Install the application/backend
requirements and `pip3 install -r requirements-test.txt` for property tests.

Only a validated tool with a clean gate can return answered. All experimental
results carry the experimental_unvalidated label, no numerical confidence and
no raw_confidence. Confidence-bearing keys are recursively removed from result
payloads/evidence; measurements are preserved. Free text is not a calibration
channel and is not semantically certified by this router. Gate warnings give
low_confidence with confidence=None, including validated tools.

Optional per-tool calibration_path files use this structure (illustrative only):

```json
{"tool":"tool-name","tool_version":"version","checkpoint_sha":null,
 "raw":[0.0,1.0],"calibrated":[0.1,0.7]}
```

The map must match the result identity, have strictly increasing raw knots and
nondecreasing calibrated values in [0,1]. Interpolation is allowed only within
its range. Invalid/missing maps yield confidence=None. Trace events record the
map SHA-256, result identity and calibration outcome. Real map fitting and
validation are outside T08.

The production T07 registry still has no land-cover or flood-otsu tool and no
validated tools. Fake-tool successes do not change those capability claims.
No VLM/training/annotation code, datasets, other worktrees or Kaggle were changed.
