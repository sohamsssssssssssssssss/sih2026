# T14 flood pack decisions

## 2026-11-16 — approved alignment tolerance

Require identical CRS and shape on all five layers for every chip. Compare the
full affine transform against S1Hand in S1 pixel coordinates: maximum absolute
origin difference and maximum absolute pixel-step difference must each be at
most 1e-6 pixel. Record both maxima per chip in index.parquet and MANIFEST.json;
fail loudly above this threshold. Preserve the original transforms and arrays.
Never resample or overwrite source metadata. The initial exact-equality failure
on Ghana_103272 was approximately 2e-11 pixel in origin and floating-point
roundoff in pixel size, not evidence warranting resampling.

## 2026-11-16 — licence and redistribution

The official hand-labeled source and label STAC collection.json files declare
`proprietary`. The repository README supplies no reuse licence, and the GitHub
repository API returned a null licence. Reuse terms: unknown. Record this exactly
in the manifest and report. Redistribution: not allowed until licence is confirmed.
The pack stays local; do not publish or upload it. Git contains code, tests and
our verification report only, never source or packed data.

## 2026-11-16 — S2 raw diagnostics

Physical saturation is unknown. Separately record raw spatial-pixel counts with
any band equal to the Int16 maximum (32767), and any band >=10000. Also retain
per-band counts so overlapping bands are not confused with spatial pixels.
Neither threshold is interpreted as physical saturation or a rejection rule.

## Existing pack definitions

Preserve official train/valid/test/Bolivia rows and order. Retain all chips, even
flagged ones. nodata_pct is the spatial union of invalid pixels across all five
layers (any invalid band); record individual layer percentages too. water_pct is
water / valid LabelHand pixels, excluding -1. LabelHand's nodata header is unset,
so treat its explicitly encoded -1 values as invalid. Train normalization uses
per-band raster masks and finite values only, without filtering on label values.

Observed S2 source dtype is Int16 despite the README's UInt16 statement; preserve
Int16. Band values are not rescaled in the pack; record the README's TOA physical
scale 1/10000 and L1C product claim separately from raster observations. Source
band names establish the canonical B01..B12 order, including B8A. S1 is Float32
VV/VH, documented as dB, with no unit conversion. Source angular resolution and
per-chip local-UTM approximate ground pixel dimensions are both recorded; the
README's nominal 10 m is not asserted to be uniform ground spacing.

## 2026-11-16 — full-pack observation after build

The initial sampled LabelHand header was unset. The full audit found one exception:
Ghana_24858 declares header nodata 0 and contains values 0 and 1. The builder's
existing source-mask rule honors that header (in addition to explicit -1), flags
this chip, preserves all values and records the header per chip in the manifest.
This conflicts with the README's no-water meaning for 0; no reinterpretation or
relabeling is made. Five other chips have no valid label pixels, and their water
percentage is unknown/null. See REPORT.md for measured counts and all flagged IDs.

## 2026-11-16 — T15 approved prediction coverage and sensitivity masks

Primary loss/metric mask is exactly target label != -1, including zero-valued
labels in Ghana_24858 despite its source header. This explicit T15 contract takes
precedence over the T14 source-header quality-mask summary; pack bytes are unchanged.
Missing shipped Otsu predictions (-1) are predicted dry, with coverage reported.
Recomputed per-chip VH Otsu fits only finite target-valid VH; non-finite VH is
predicted dry. Model S1 non-finite values and S2 encoded nodata (0) or non-finite
values become zero AFTER train-stat normalization (mean imputation).

Also report every baseline/model IoU on a common strict sensitivity mask:
target-valid AND all S1/S2 input bands finite/nonmissing before imputation AND
shipped Otsu valid. S2 encoded nodata is treated as missing before imputation.
The same strict mask applies to S1, S1+S2 and all baselines, so comparisons do not
change pixels by model mode. Show both masks side by side; flag if any H1–H3
verdict changes. Report per-split target-valid counts and affected counts/percent
for missing Otsu, non-finite VH, and any S2 nodata band (overlaps allowed).
No label-valid pixel is removed from the primary result because of input quality.
