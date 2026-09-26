# Deterministic bi-temporal change baseline

SatQuery's `change_vqa` capability uses `change-deterministic`, version `bitemporal-difference-v1`. It compares two already co-registered rasters and reports spatial and distribution evidence. It is not a trained change model and does not establish CDVQA accuracy.

## Input and temporal contract

The first scene is T1 and must have an acquisition timestamp earlier than the second scene, T2. Pair validation requires compatible declared modalities, at least 90% footprint overlap, valid affine georeferencing, identical dimensions, identical CRS, and an exactly identical affine transform. Any pair requiring reprojection, resampling, or co-registration fails closed. The provider repeats the raster dimension, CRS, affine, band-count, finite-data, and co-valid-pixel checks before computation.

The provider accepts either three-band RGB or five-band B02/B03/B04/B08/dataMask GeoTIFFs. Five-band rasters with band descriptions must use that exact order; files without descriptions use the same positional contract.

## Computation

For multispectral inputs, the provider computes NDVI `(B08-B04)/(B08+B04)` and McFeeters-style NDWI `(B03-B08)/(B03+B08)` at T1 and T2. It reports signed delta summaries and defines normalized change magnitude as `(abs(delta_NDVI) + abs(delta_NDWI)) / 4`, clipped to `[0,1]`. Division by zero, nodata, NaN, and infinite values are excluded.

For RGB inputs, each band is normalized against the shared T1/T2 2nd and 98th percentiles, clipped to `[0,1]`, and the three absolute band differences are averaged. This is a visual-difference heuristic. It is not semantic change understanding.

For both methods, a co-valid pixel is marked changed only when normalized change magnitude is strictly greater than `0.1`. This fixed threshold is heuristic and uncalibrated. Evidence records the threshold and rule, changed-pixel count and fraction, magnitude mean/maximum and 5th/50th/95th percentiles, co-valid coverage, T1/T2 paths and SHA-256 values, and the normalized bounding extent of all changed pixels when any exist. Confidence remains `null`.

## Routing and limitations

An appropriate two-scene query routes through `change_vqa`, pair validation, and the deterministic provider. The persisted live trace records both image paths, both acquisition timestamps, T1/T2 order, provider identity, planner rule, and execution step.

The baseline reports no land-cover class, object class, causal explanation, or claims such as built-up-area increase. The one enclosing extent can include unchanged pixels between disconnected changed regions. Percentile normalization can reduce sensitivity to radiometric differences or outliers, and no atmospheric, seasonal, illumination, or sensor calibration correction is performed. Generated GeoTIFF fixtures verify the contract. A small real Sentinel-2 L2A pair pack (research evaluation, not an accuracy benchmark) is evaluated in `docs/research/change-real-pair-evaluation.md`. It documents where this baseline detects change and where it fails: bare-to-built transitions, phenology, and unmasked cloud or haze.
