# Real bi-temporal change smoke plan

## Preferred bounded source

Use two official Sentinel-2 L2A observations from Copernicus Data Space for one small, unchanged area of interest. Select the products through the CDSE Catalog, then request the same output CRS, bounding box, width, height, and band order (`B02`, `B03`, `B04`, `B08`, `dataMask`) from the Process API. This reuses the repository's established CDSE provenance pattern while keeping the download bounded. The two timestamps must come from product metadata and T1 must precede T2.

Do not use screenshots, rescaled browser exports, or manually assigned timestamps. Do not claim semantic change from the output.

## Required retained metadata

For each observation retain product ID, collection, acquisition timestamp, processing level, request body, response content type, byte size, SHA-256, CRS, affine transform, bounds, dimensions, band descriptions, nodata/mask behavior, and retrieval time. Record the AOI separately. Treat API request parameters as declared provenance and raster inspection as observed metadata.

Suggested ignored local layout:

```text
data/external/change-smoke/<pair-id>/
  t1.tif
  t2.tif
  provenance.json
```

## Validation and smoke

1. Inspect both rasters with `rasterio`: five bands, finite samples, nonempty masks, identical CRS, dimensions, affine transform, and bounds.
2. Verify both SHA-256 values and the catalog timestamps in `provenance.json`.
3. Start the canonical API, upload T1 and T2 with `modality=multispectral`, truthful timestamps, the same `pair_group`, and their sensor/product declarations.
4. Submit `change_vqa` with T1 as `scene_id` and T2 as `scene_id_2`.

```bash
curl -sS -F file=@data/external/change-smoke/<pair-id>/t1.tif \
  -F modality=multispectral -F sensor=Sentinel-2-L2A \
  -F acquisition_timestamp=<T1-RFC3339> -F pair_group=<pair-id> \
  http://127.0.0.1:8000/api/scenes

curl -sS -F file=@data/external/change-smoke/<pair-id>/t2.tif \
  -F modality=multispectral -F sensor=Sentinel-2-L2A \
  -F acquisition_timestamp=<T2-RFC3339> -F pair_group=<pair-id> \
  http://127.0.0.1:8000/api/scenes

curl -sS -H 'content-type: application/json' \
  -d '{"scene_id":"<T1_SCENE_ID>","scene_id_2":"<T2_SCENE_ID>","question":"Measure visual change between these observations.","capability":"change_vqa","execution_mode":"live"}' \
  http://127.0.0.1:8000/api/analyze \
  -o data/external/change-smoke/<pair-id>/smoke-response.json
```

Confirm `execution_mode=live`, provider `change-deterministic`, version `bitemporal-difference-v2`, both input identities in the trace/evidence, `confidence` absent from the API response, finite change statistics, and hash-chain verification. Record wall latency separately as an operational observation.

This smoke would prove that a real provenance-complete pair passes canonical ingestion, compatibility gating, deterministic change execution, evidence generation, and trace persistence. It would not prove semantic understanding, CDVQA accuracy, threshold calibration, geographic generalization, or model improvement.
