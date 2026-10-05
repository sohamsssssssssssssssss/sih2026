# Deterministic bi-temporal change baseline

SatQuery's `change_vqa` capability uses `change-deterministic`, version `bitemporal-difference-v2`. It compares two already co-registered rasters and reports spatial and distribution evidence. It is not a trained change model and does not establish CDVQA accuracy.

## Input and temporal contract

The first scene is T1 and must have an acquisition timestamp earlier than the second scene, T2. Pair validation requires compatible declared modalities, at least 90% footprint overlap, valid affine georeferencing, identical dimensions, identical CRS, and an exactly identical affine transform. Any pair requiring reprojection, resampling, or co-registration fails closed. The provider repeats the raster dimension, CRS, affine, band-count, finite-data, and co-valid-pixel checks before computation.

The provider accepts either three-band RGB or five-band B02/B03/B04/B08/dataMask GeoTIFFs. Five-band rasters with band descriptions must use that exact order; files without descriptions use the same positional contract.

A pair whose band descriptions are both exactly `VV`, `VH`, `dataMask` (linear Sentinel-1 backscatter) takes the SAR water-change path below instead. A pair with only one such raster is rejected. SAR rasters must have a CRS and a north-up grid.

## Computation

For multispectral inputs, the provider computes NDVI `(B08-B04)/(B08+B04)` and McFeeters-style NDWI `(B03-B08)/(B03+B08)` at T1 and T2. It reports signed delta summaries and defines normalized change magnitude as `(abs(delta_NDVI) + abs(delta_NDWI)) / 4`, clipped to `[0,1]`. Division by zero, nodata, NaN, and infinite values are excluded.

For RGB inputs, each band is normalized against the shared T1/T2 2nd and 98th percentiles, clipped to `[0,1]`, and the three absolute band differences are averaged. This is a visual-difference heuristic. It is not semantic change understanding.

For both methods, a co-valid pixel is marked changed only when normalized change magnitude is strictly greater than `0.1`. This fixed threshold is heuristic and uncalibrated. Evidence records the threshold and rule, changed-pixel count and fraction, magnitude mean/maximum and 5th/50th/95th percentiles, co-valid coverage, T1/T2 paths and SHA-256 values, and the normalized bounding extent of all changed pixels when any exist. Confidence remains `null`.

## Sentinel-1 water change

`models/change/sar_water.py` converts VV to dB and segments open water with one Otsu threshold pooled over both dates, so "water" means the same thing at T1 and T2. Values are clipped to their 0.1st–99.9th percentiles first; unclipped, a single extreme pixel can capture the split on its own. If that threshold is above `-15` dB, the scene has no open-water mode: the provider abstains (`status: abstained`, `reason_code: NO_OPEN_WATER_MODE`) and reports no hectares. Otherwise it reports water at T1 and T2, newly water-covered area and receded area. Change regions smaller than 10 connected pixels are removed as speckle; small gaps inside a change region are never filled. Areas come from pixel corners projected into EPSG:6933 (equal area), so lat/lon and UTM grids both give true hectares. Change polygons are returned as an RFC 7946 GeoJSON FeatureCollection in EPSG:4326, largest first, capped at 500 with the full count reported. VH is not used. The `-15` dB ceiling is a physical prior, not a calibrated value, and confidence remains `null`. A trained segmenter replaces `water_masks()` and has to beat this baseline on held-out IoU.

## Routing and limitations

An appropriate two-scene query routes through `change_vqa`, pair validation, and the deterministic provider. The persisted live trace records both image paths, both acquisition timestamps, T1/T2 order, provider identity, planner rule, and execution step.

The baseline reports no land-cover class, object class, causal explanation, or claims such as built-up-area increase. The one enclosing extent can include unchanged pixels between disconnected changed regions. Percentile normalization can reduce sensitivity to radiometric differences or outliers, and no atmospheric, seasonal, illumination, or sensor calibration correction is performed. No suitable repository-controlled real or benchmark bi-temporal pair was present for an operational smoke run; generated GeoTIFF fixtures verify the contract.
