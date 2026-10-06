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

### First real runs

Each catalogued event was fetched once with `python -m backend.sentinel1 --event <event_id>` and analysed by this provider directly, all from commit `959a014` on 6 October 2026. Every pair is Sentinel-1A on one track, passed `change_vqa` pair validation, and is at least 99.99% co-valid on a 10 m UTM grid. Product ids, grids, full statistics and the top villages are in [`data/manifests/cdse/flood-runs.v1.json`](../data/manifests/cdse/flood-runs.v1.json).

| Event | Pair (track) | Threshold | New water (±1 dB) | Receded | Most new water |
|---|---|---|---|---|---|
| `kosi-2024` | 24 Sep → 6 Oct 2024 (ascending 85) | -13.2 dB, 192 tiles | 1,404.9 ha (1,242.7–1,583.5) | 55.0 ha | Jamalpur 252.9 ha, Jhagarua 135.8 ha, Dhangha 131.9 ha; 46 of 51 villages |
| `silchar-2022` | 16 → 28 Jun 2022 (ascending 41) | -13.2 dB, 1,034 tiles | 10,077.1 ha (9,964.8–10,082.4) | 39.4 ha | not named: no boundaries |
| `kerala-2018-periyar` | 16 Jul → 21 Aug 2018 (descending 165) | -11.5 dB, 193 tiles | 2,092.5 ha (2,033.8–2,102.0) | 262.7 ha | Varappuzha 198.8 ha, Kunnukara 195.5 ha, Thirumukkulam 175.3 ha; 37 of 38 villages and towns |

- **Kosi:** the post-event image is 7 days after the 29 September breach. Water rose from 967.2 to 2,354.6 ha. Bhobhaul, the likely Census spelling of the breach village Bhubhol, gained 40.7 ha (11th).
- **Silchar:** the post-event image falls inside the 19–30 June flood. New water covers about a quarter of the scene, and total water rose from 5,690.7 to 15,923.8 ha. DataMeet has no Assam boundaries, so no village is named. Water among buildings in the town itself can be missed, because buildings brighten the radar return rather than darken it.
- **Kerala:** only Sentinel-1A acquisitions exist over this AOI in the catalogued windows, so the pair is 36 days apart. The post-event image is two days after the 14–19 August spell, so it shows water still standing after the peak, not the peak extent. Its threshold of -11.5 dB is high for open water: the selected tiles' water classes average below -15 dB, but the split between them and land lands higher. Pixels between about -15 and -11.5 dB, such as wet soil or flooded vegetation, therefore count as water here. Lowering the threshold by 1 dB slightly raises new water rather than lowering it, which is plausible when both dates move together.
- **All three:** each pre-event image falls in the monsoon, and water already present then (river, wetland or earlier flooding) is not counted as new. The numbers are change between two dates, not total flood extent. None of them has been compared with an independent flood map.

## Village overlay

When water change is measured, `models/change/villages.py` burns each village polygon in `data/boundaries/villages.v1.geojson` onto its own pixel window of the scene grid, so overlapping or nested polygons each keep their full area, and sums new-water area per village. A village's area counts all its pixels in the scene, including unobserved ones; `observed_fraction` says how much was seen. The `village_flooding` evidence lists every village with new water, largest first: name, sub-district, district, state, Census 2001 code, flooded hectares, the village's area inside the scene, the flooded and observed (valid-pixel) fractions, and whether the village extends past the scene edge. Villages without new water are counted but not listed. The answer names the five most flooded. If no boundaries cover the scene, the evidence says `no_boundaries` and the answer says villages are not named. The boundaries are DataMeet's community-digitised Census 2001 polygons (ODbL-1.0), subset to the catalogued flood events; see `data/boundaries/README.md`.

## Routing and limitations

An appropriate two-scene query routes through `change_vqa`, pair validation, and the deterministic provider. The persisted live trace records both image paths, both acquisition timestamps, T1/T2 order, provider identity, planner rule, and execution step.

The baseline reports no land-cover class, object class, causal explanation, or claims such as built-up-area increase. The one enclosing extent can include unchanged pixels between disconnected changed regions. Percentile normalization can reduce sensitivity to radiometric differences or outliers, and no atmospheric, seasonal, illumination, or sensor calibration correction is performed. No suitable repository-controlled real or benchmark bi-temporal pair was present for an operational smoke run; generated GeoTIFF fixtures verify the contract.
