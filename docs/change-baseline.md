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

`models/change/sar_water.py` converts VV to dB and sets one water threshold for both dates, so "water" means the same thing at T1 and T2. The threshold is a simplified split-based Otsu after Chini et al. (2017, IEEE TGRS 55(12), 6975-6988). Each date is cut into 64 px tiles. A tile is selected only when its own Otsu split is clearly bimodal: Ashman's D above 2, each class at least 10% of the tile, a low-class mean below `-15` dB and a high-class mean above it. The tile must hold both water-like and land-like pixels, so two dark land classes such as sand and smooth soil cannot pass as water. A final tile flush with each far edge covers the strip left after the last full tile. Otsu over the pooled pixels of the selected tiles sets the threshold. Unlike Chini et al., the tiles are fixed rather than split hierarchically, and class statistics come from the Otsu split rather than a fitted Gaussian mixture. Otsu inputs are clipped to their 0.1st–99.9th percentiles, so a single extreme pixel cannot capture the split.

A scene-wide split fails when water is a few percent of the scene. On the real Kosi 2024 pair it landed at -12.3 dB, between two land classes, and the provider abstained. If no tile qualifies, the provider abstains (`status: abstained`, `reason_code: NO_OPEN_WATER_MODE`) and reports no hectares.

Otherwise it reports water at T1 and T2, newly water-covered area and receded area. It also reports new-water hectares with the threshold moved down and up by 1 dB (`new_water_ha_sensitivity`), and the answer quotes that range. Change regions smaller than 10 connected pixels are removed as speckle; small gaps inside a change region are never filled. Areas come from pixel corners projected into EPSG:6933 (equal area), so lat/lon and UTM grids both give true hectares. Change polygons are returned as an RFC 7946 GeoJSON FeatureCollection in EPSG:4326, largest first, capped at 500 with the full count reported. VH is not used. The `-15` dB water level and the tile size are fixed choices, not calibrated values, and confidence remains `null`. A trained segmenter replaces `split_threshold()` and `water_masks()` and has to beat this baseline on held-out IoU.

### First real run: Kosi 2024

`python -m backend.sentinel1 --event kosi-2024` paired Sentinel-1A acquisitions from 24 September and 6 October 2024 on ascending track 85. The output was a 749 x 1649 grid at 10 m with 100% valid pixels. The split-based threshold was -13.2 dB from 192 tiles. It found 1,405 ha of new open water (1,243–1,584 ha at ±1 dB), 55 ha receded, and water rising from 967 to 2,355 ha. 46 of 51 villages gained some open water; the most were Jamalpur (253 ha), Jhagarua (136 ha) and Dhangha (132 ha). Bhobhaul, the likely Census spelling of the breach village Bhubhol, gained 40.7 ha. None of this has been checked against an independent flood map.

## Village overlay

When water change is measured, `models/change/villages.py` burns each village polygon in `data/boundaries/villages.v1.geojson` onto its own pixel window of the scene grid, so overlapping or nested polygons each keep their full area, and sums new-water area per village. A village's area counts all its pixels in the scene, including unobserved ones; `observed_fraction` says how much was seen. The `village_flooding` evidence lists every village with new water, largest first: name, sub-district, district, state, Census 2001 code, flooded hectares, the village's area inside the scene, the flooded and observed (valid-pixel) fractions, and whether the village extends past the scene edge. Villages without new water are counted but not listed. The answer names the five most flooded. If no boundaries cover the scene, the evidence says `no_boundaries` and the answer says villages are not named. The boundaries are DataMeet's community-digitised Census 2001 polygons (ODbL-1.0), subset to the catalogued flood events; see `data/boundaries/README.md`.

## Routing and limitations

An appropriate two-scene query routes through `change_vqa`, pair validation, and the deterministic provider. The persisted live trace records both image paths, both acquisition timestamps, T1/T2 order, provider identity, planner rule, and execution step.

The baseline reports no land-cover class, object class, causal explanation, or claims such as built-up-area increase. The one enclosing extent can include unchanged pixels between disconnected changed regions. Percentile normalization can reduce sensitivity to radiometric differences or outliers, and no atmospheric, seasonal, illumination, or sensor calibration correction is performed. No suitable repository-controlled real or benchmark bi-temporal pair was present for an operational smoke run; generated GeoTIFF fixtures verify the contract.
