# SatQuery acceptance fixtures

## Mumbai Sentinel-2 cloud refusal

`mumbai-s2-cloudy.tif` comes from the [Earth Search Sentinel-2 L2A item S2C_43QBB_20250829_0_L2A](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2C_43QBB_20250829_0_L2A), acquired at `2025-08-29T05:53:59.828000Z`. The `cloud_pct` tag copies the source `eo:cloud_cover` of **99.980801%**. This is a **scene-level value, not a chip-level cloud fraction**. Visual inspection of [the RGB preview](mumbai-s2-cloudy.png) confirmed visible cloud across the chip. The preview is not used to calculate the cloud value.

The Earth Search query used collection `sentinel-2-l2a`, bbox `[72.78,18.98,72.88,19.08]`, July–August 2025 (UTC), and `eo:cloud_cover > 80`. Rasterio read a 256 × 256 window (column 7086, row 9374) from each public `blue`, `green`, `red`, and `nir` COG at 10 m in EPSG:32643. The fixture keeps the central 120 × 120 window (column 7154, row 9442), centered near 72.835° E, 19.025° N. No pixels were resampled.

The source COGs store uint16 DN with scale 0.0001 and offset −0.1. The fixture applies those published values and stores signed int16 reflectance × 10,000 (source DN minus 1,000). Bands are B02, B03, B04, B08, with the same shape, order, and units as the Punjab fixture. Unlike Punjab’s multitemporal composite, Mumbai is a single L2A acquisition. The preview shows B04/B03/B02 with a fixed display-only stretch and gamma. It does not change the GeoTIFF.

## Punjab Sentinel-2 land cover

`punjab-s2.tif` is the optical-only export of WorldCover pack patch `0` (EPSG:32643) used in the HTTP acceptance report. It preserves B02/B03/B04/B08 and their native 120 × 120, 10 m grid. The source composite has no cloud percentage tag. [Acceptance report](../../docs/plan/E2E_ACCEPTANCE.md)
