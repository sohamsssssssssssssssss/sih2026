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
