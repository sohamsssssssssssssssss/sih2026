# T07c tools — final report

Branch: task/T07b-tools, fast-forwarded to T08 06adbc9. Contracts v1.2.0.
Part A was committed and pushed independently as d5f7296. Part B was
preregistered as f5bd22d before implementation/evaluation.

| Tool / mode | Status | CPU | One real-patch latency |
|---|---|---|---:|
| landcover / s1 | experimental_unvalidated | yes | 3 ms warm (1243 ms cold) |
| landcover / s2 | experimental_unvalidated | yes | 3 ms warm (106 ms cold) |
| landcover / s1s2 | experimental_unvalidated | yes | 4 ms warm (102 ms cold) |
| flood-otsu | experimental_unvalidated | yes | 232.2 ms including event-window threshold |

Land-cover latencies use one BEN test patch (120x120), one cold and one warm
call per mode. Flood latency returns one real 512x512 India test patch, including
threshold computation over all 68 India-event chips. These are individual CPU
samples, not a throughput benchmark. [Land-cover evidence](T07c-landcover.md).

Part B verdict: **FAIL**. India-test IoU **0.631332**, CI95 **[0.440758, 0.723997]**,
below the fixed >=0.65 rule. B1 shipped: **0.704408**; B2 per-chip: **0.379412**.
[All-split comparison and train-only audit](T07c-flood.md). No post-verdict
changes to the method, clamp, histogram or mask policy; no rerun to seek a pass.
The new tool remains registered experimental despite failure, as requested.

Router: landcover intent -> landcover; flood intent -> flood-otsu. Cloudy joint
land-cover scenes select S1; clear joint scenes select S1S2. Router integration
tests invoke both actual adapters. Existing T08 synthetic examples were updated
only for the requested concrete tool names and direct radar flood route; all
40 still pass. The historical T08 report records its earlier placeholder table.

Five coarse classes are trees, shrub_grass, cropland, built, water, excluding
wetland. This follows the provisional mapping stated during the task because the
original T07b five-class list was unavailable. No class scores are advertised as
calibrated confidence or area fractions. The only copied BEN files were
ben/landcover.py and ben/transfer.py from feature/ben-landcover (4707d4c).
Their normalizer and architecture are reused without alteration. Their existing
scikit-learn import is declared in requirements.txt; no package installation or
training was needed. Checkpoints remain external, configured by path.

For flood input, provide a CHW scene or NCHW whole-event stack in dB. Optional
structured input: {s1: {pixels, bands, units, chip_index}}; chip_index returns one
patch while computing its threshold over the entire event stack. Joint scenes
require this per-sensor metadata because Scene.units cannot describe mixed
optical DN and radar dB. Callers must supply the whole scene/event window, not
silently split it into independently thresholded chips. The water mask denotes
open water; flood causality and inundation under vegetation are not established.

Verification:

- python3.11 -m pytest core orchestrator -q: **500 passed, 2 skipped, 7 subtests passed**.
- All three real-checkpoint land-cover tests executed and passed; no weight test skipped.
- Full baseline: **1,173 passed, 7 failed, 12 collection errors, 3 skipped**.
- Full final: **1,203 passed, same 7 failures, same 12 collection errors, 3 skipped**.
- Full suites use --continue-on-collection-errors to compare all runnable tests;
  both exit 1. JUnit failure/error identities match exactly.
- [Validation evidence](T07c-validation.json) records identities and JUnit SHA-256.

No weights or datasets were committed. No other worktree, VLM training code,
annotation files or Kaggle account was changed. The only merge operation was the
explicitly requested successful fast-forward; no merge commit or force push.
