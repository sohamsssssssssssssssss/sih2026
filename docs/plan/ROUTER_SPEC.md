# T08 router specification — committed before implementation

Base: task/T07-registry at 7bfb1790ad92a8e63225d6475f1d2975ef0e43c2.
Precondition: core/orchestrator 366 passed, 2 skipped, 7 subtests passed on a
committed snapshot. Untouched full suite: 7 failed, 12 errors, 1069 passed,
3 skipped. Final comparison uses failure identities, not just counts.

## Routes

Deterministic English word-boundary patterns, no model call. Precedence:
non-imagery refusal > temporal change > count/detect/locate > flood/water >
land cover/forest/cropland/vegetation > general. Unknown text uses the VLM if
registered, otherwise tool_missing. Text determines intent; explicit qtype=change
also conservatively requires a second scene. No multilingual coverage is claimed.

| Intent | Ranked tool names |
|---|---|
| landcover | landcover |
| flood with optical | flood-otsu |
| flood radar-only | flood-otsu |
| change | change-deterministic |
| count/detect/locate | grounding-dino-swint |
| general/unknown | qwen2.5vl-3b |

The selected disabled/unregistered tool refuses; no silent classifier replacement.
T07 lacks landcover and flood-otsu. Fake tests do not claim these exist in the
production registry. Joint optical-SAR is generic evidence, not a validated flood
map, and its adapter still requires actual co-gridded optical and SAR inputs.

## Gate rules and threshold provenance

Every rule decision and used threshold produces its own event. Gate functions are
pure; a separate writer sends events through existing orchestrator.trace, whose
hash chain uses core.canonical. No model is loaded or called in route/gate logic.

1. Prices/costs, names/ownership, people and future/prediction -> not_answerable.
2. Change without a second scene -> second_scene_missing. Temporal CRS/bounds
   mismatch -> checks_disagree (exact metadata contract comparison).
3. Transform bounds to EPSG:4326. Outside [68,6,98,38] -> out_of_region, allowed
   with a warning. **PROVISIONAL** conservative rectangle, not a legal boundary.
   Failed transformation -> region_unknown and low_confidence, not a guess.
4. Nodata >50% on either required scene -> nodata. **PROVISIONAL** majority-missing
   guard, not a scientifically validated accuracy cutoff. Equality passes.
5. Optical cloud >50% -> reroute only if S1 exists at all required dates and an
   appropriate registered/enabled radar tool exists: landcover -> landcover in
   S1 mode; flood -> flood-otsu; change -> change-deterministic in S1 mode.
   Otherwise refuse cloud, Evidence.needs_sensor=(S1,). Count/general have no
   authorized radar replacement. **PROVISIONAL** majority-cloud guard; <=50% is
   not claimed reliable. Unknown optical cloud -> cloud_unknown/low_confidence.
6. Required sensors absent -> sensor_missing. Missing bands -> bands_missing,
   with exact band names and needed sensors in evidence. Validate both dates after
   any reroute. Combined optical-SAR requires S1+S2. Known mode-dependent bands:
   landcover S1 VV/VH or S2 B02/B03/B04/B08; flood-otsu VH; change S1 VV/VH/
   dataMask, S2 B02/B03/B04/B08/dataMask, RGB red/green/blue. These groups follow
   provider interfaces/code at 7bfb179 and TOOLS_INVENTORY.md, not accuracy
   thresholds. T07's change union list does not require radar bands on optical
   change images. Other tools honor declared bands. Units/pixel validation stays
   with adapters; no unit conversion is invented here.
7. Tool missing -> tool_missing; disabled -> tool_disabled.
8. gsd_m > explicit Query.needs_gsd_m -> gsd_too_coarse. The threshold is the
   user's stated requirement, not a universal learned object-size claim.
9. abs(observed_band_mean-training_mean)/training_std >2 -> out_of_distribution,
   low_confidence, never refusal. **PROVISIONAL** two-std warning, not calibrated.
   The task's Rajasthan +3.18 std motivates this check but does not establish a
   validated threshold. Observed means are supplied by the pixel loader in the
   same units as supplied training stats. Missing/non-finite means/stats or
   nonpositive std -> ood_unknown/low_confidence for learned tools. Deterministic
   tools without a configured training distribution report not_applicable.

Threshold config records PROVISIONAL provenance for geographic bounds, cloud,
nodata and OOD shift. No unsupported numerical confidence cutoff is introduced.

## Contract and answer policy

Add explicit Evidence.needs_sensor and optional Answer.evidence; extend the fixed
refusal enum for the precise rules above. Preserve existing enum values/records,
regenerate schemas and bump minor version to 1.2.0 so new wire fields/reasons are
not falsely labeled v1.1.0. Old contract data remain accepted.

Refusals always have a reason and no confidence. Answered requires a validated
tool and a clean gate; warnings or experimental status -> low_confidence.
Experimental tool output has confidence=None, raw_confidence=None, an experimental
label in evidence, and recursively removed confidence keys in free-form result
payloads/evidence. Other measurements are retained. Free text is not claimed to
be calibrated. Mismatched ToolResult identity -> checks_disagree; error payloads
become refusals, never confident answers.

Only a per-tool calibration map may produce Answer.confidence. Bind it to tool
name/version/checkpoint; increasing raw-score knots and calibrated values in
[0,1]; interpolate only inside its range. Missing/invalid/mismatched map -> None
and a trace event. No map is fitted here. Every ranked candidate, each gate
rule/threshold/outcome, reroute, status policy and calibration decision is traced
separately with a supplied trace_id, outside the pure decision function.

## Forty precommitted examples

Synthetic fixtures, not a language benchmark. Six named fake tools are enabled
experimental by default. Default S2: 10m, EPSG:4326, bounds [75,20,75.1,20.1], cloud
10%, nodata 0%, B02/B03/B04/B08/dataMask. s1: VV/VH/dataMask. joint: both groups.
rgb: red/green/blue. change profiles add matching second scene. Cloud=80%;
nodata=60%; outside=[-2,45,-1,46]; ood B04 mean=131.8, training=100/std=10 (3.18std).
Those stats are synthetic fixture values, not empirical claims. no_radar removes
flood-otsu; no_vlm/no_landcover remove that tool; disabled marks landcover disabled;
missing_band removes B08; coarse_count sets Query.needs_gsd_m=1.

| ID | Question | Profile | Expected route or refusal |
|---|---|---|---|
| E01 | What is the land cover? | s2 | landcover |
| E02 | How much cropland is visible? | s2 | landcover |
| E03 | Is this forest cover? | s2 | landcover |
| E04 | Estimate the built-up area. | s2 | landcover |
| E05 | What land cover does the radar show? | s1 | landcover |
| E06 | What fraction is cropland? | s2 | landcover |
| E07 | Describe the vegetation cover. | s2 | landcover |
| E08 | Is there forest here? | s2 | landcover |
| E09 | Is there flood water? | joint | flood-otsu |
| E10 | Map the water extent. | s1 | flood-otsu |
| E11 | Which area is inundated? | s1 | flood-otsu |
| E12 | Map flooding through the clouds. | joint_cloud | flood-otsu |
| E13 | Describe the land cover through clouds. | joint_cloud | landcover |
| E14 | Map flood water outside India. | s1_outside | flood-otsu |
| E15 | What changed between these dates? | change_s2 | change-deterministic |
| E16 | Have the buildings changed? | change_s2 | change-deterministic |
| E17 | Has the water extent expanded? | change_s1 | change-deterministic |
| E18 | Compare before and after the flooding. | change_joint_cloud | change-deterministic |
| E19 | How many cars are visible? | rgb | grounding-dino-swint |
| E20 | Detect ships. | s2 | grounding-dino-swint |
| E21 | Locate buildings. | rgb | grounding-dino-swint |
| E22 | How many trees are visible? | s2 | grounding-dino-swint |
| E23 | Describe this scene. | s2 | qwen2.5vl-3b |
| E24 | What is visible in this image? | s2 | qwen2.5vl-3b |
| E25 | Explain this pattern. | s2 | qwen2.5vl-3b |
| E26 | What is Rajasthan's land cover? | ood | landcover |
| E27 | What is the price of this land? | s2 | refuse:not_answerable |
| E28 | Who owns these fields? | s2 | refuse:not_answerable |
| E29 | How many people live here? | rgb | refuse:not_answerable |
| E30 | Predict tomorrow's flood. | joint | refuse:not_answerable |
| E31 | What is this village's name? | s2 | refuse:not_answerable |
| E32 | What changed since last year? | s2 | refuse:second_scene_missing |
| E33 | What is the land cover? | s2_cloud | refuse:cloud |
| E34 | Map flooding through clouds. | joint_cloud_no_radar | refuse:tool_missing |
| E35 | Describe this scene. | s2_nodata | refuse:nodata |
| E36 | What is the land cover? | s2_missing_band | refuse:bands_missing |
| E37 | What is the land cover? | tool_disabled | refuse:tool_disabled |
| E38 | Explain this pattern. | no_vlm | refuse:tool_missing |
| E39 | What is the land cover? | no_landcover | refuse:tool_missing |
| E40 | Count cars at one-metre detail. | coarse_count | refuse:gsd_too_coarse |

Warnings: E12/E13/E18 monsoon_mode, E14 out_of_region, E26 out_of_distribution.
These remain routes, not silent refusals. Report the complete route confusion
table and every off-diagonal error. Preserve missing-tool reality in production;
no registry validation or model availability is inferred from fake tests.

T07c supersedes the placeholder table: landcover -> landcover; every flood route ->
flood-otsu. Missing/disabled radar now refuses tool_missing/tool_disabled directly.
Both tools are registered experimental; joint landcover uses s1s2 unless cloudy.
The original T08 report remains a historical measurement of its pre-T07c table.
