# Real bi-temporal change evaluation (v1)

Status: evaluation of the existing `change-deterministic` / `bitemporal-difference-v1` provider on real Sentinel-2 L2A pairs. Nothing was trained or tuned; the provider's fixed `0.1` threshold was not changed. Registry records: `SQ-20260926-301` … `-314` (`eval/registry/`). Full per-pair output: `eval/results/change-real-pairs-v1.json`.

**This measures change *detection* only.** The provider flags pixels whose NDVI/NDWI moved; it says nothing about what changed or why. Every "interpretation" below (construction, water loss, landslide) comes from dated public event records and visual inspection, never from the provider.

## Data and provenance

| Item | Choice |
|---|---|
| Source | Element84 Earth Search STAC, collection `sentinel-2-c1-l2a` (ESA Collection 1 reprocessing; public COGs, anonymous HTTPS range reads) |
| Why C1 | One processing-baseline convention across years (PB ≥ 05.00, BOA offset −0.1). The legacy `sentinel-2-l2a` collection mixes PB 02.xx and 05.xx products for the same dates. C1 assets also carry a `file:checksum`. |
| Bands | B02, B03, B04, B08 at native 10 m, plus a `dataMask` band. The 20 m SCL layer is kept as a separate diagnostic raster. |
| Radiometry | Reflectance = DN × scale + offset from each asset's STAC `raster:bands`. Checked empirically: C1 red DN p01 ≈ 1100–1400, confirming DN carries the +1000 offset. |
| Grid | Both dates come from the **same MGRS tile**, and T2 reuses T1's exact native pixel window. Nothing is resampled or reprojected. Windows are snapped to even offsets so the 20 m SCL nests exactly (SCL is nearest-replicated 2×2 for diagnostics only). |
| Masks | `dataMask` = 1 where all four bands have non-zero DN (L2A nodata). **Clouds are deliberately not masked**, to match the semantics of the CDSE `dataMask` in the smoke plan and to measure cloud false positives. No GeoTIFF nodata tag is written, because 0.0 is a valid reflectance. |
| Size | AOIs of 3.9–7.8 km (390–778 px per side); every raster stays under the 20 MiB API upload cap. |
| Transfer | ~307 MB estimated (sum of native COG blocks touched). Rasters and renders live outside git in `~/satquery-data/change/`. |

Per-pair provenance (STAC item IDs, acquisition times, platform, processing baseline, product URI, granule ID, tile cloud cover, sun angles, asset hrefs and checksums, native window, output CRS/transform/bounds/size, SHA-256 of every written raster, SCL histogram, retrieval time) is in `data/manifests/change/<pair_id>.provenance.json`. The pair spec, including the selection rule, expected outcomes and reference boxes, is `data/manifests/change/pairs.v1.json`.

### Deviations from `docs/research/change-smoke-plan.md`

- **Source:** CDSE requires credentials, which are not available here (a user blocker), so pairs come from Earth Search C1 instead. The provider contract is unchanged: five bands, B02/B03/B04/B08/dataMask.
- **dataMask:** derived from DN ≠ 0 rather than CDSE's `dataMask`. Both mean data presence, not cloud. SCL is added as a separate diagnostic layer that CDSE's 5-band request would not return.
- **API smoke:** the canonical API was exercised in-process through FastAPI `TestClient` (`/api/scenes` upload with truthful metadata, then `/api/analyze` `change_vqa` live). This is the same route → service → pair gate → provider → trace code path, but not a running uvicorn server, so wall latency was not measured. The scene store and trace were redirected to `~/satquery-data/change/runtime/`.

## Method

1. **Selection (declared before pixels were seen):** from STAC metadata only, the lowest tile cloud cover in the target window, preferring the same platform and nearest day-of-year. AOI-level cloud is reported from SCL afterwards and never used to re-select.
2. **Expectations (declared before the provider ran):** each pair has an `expected_outcome` from dated public records. For event cases, coarse analyst `reference_boxes_norm` were drawn on true-colour previews (`--rgb-only`) before any change output existed; git commits `feea1ad` and `901bd59` precede the evaluation commit `3ec55e5`. No box was drawn where the event could not be localized visually.
3. **Pair gate first:** every pair is uploaded and submitted through the API. Incompatible pairs must be refused with the expected reason code.
4. **Provider output** (changed fraction, magnitude, ΔNDVI/ΔNDWI) comes from the API evidence. A per-pixel replica of the rule (`eval/change_real_pairs.py::provider_rule`) is asserted to reproduce the provider's changed and valid pixel counts exactly on every pair, so the diagnostics below describe the same pixels the provider counted.
5. **Diagnostics:**
   - detections inside vs outside the reference boxes;
   - detections on SCL clear / cloud (8,9,10) / shadow (2,3) pixels;
   - detections on pixels whose SCL water class flips between dates. ESA's scene classification is a different algorithm but also spectral, so it is a partially independent reference;
   - NIR phase-correlation residual shift, with its correlation peak;
   - a re-run of the provider with `dataMask` restricted to SCL-clear pixels at both dates.
6. **Registry status** is pre-declared in code (`classify`). It means whether the case produced valid evidence, **not** whether the baseline was right:
   - FAILED: unexpected API outcome, wrong rejection code, or an invalidated pair;
   - INCONCLUSIVE: co-valid fraction < 0.5, SCL clear-both < 0.9, or a cloud control with no cloud;
   - PASSED: otherwise.

## Results

Changed fraction = fraction of co-valid pixels with magnitude > 0.1. "Inside/outside" = changed rate inside/outside the reference boxes. All accepted pairs had co-valid fraction 1.000.

### Event cases

| ID | Pair (dates) | Status | Changed | In / out (ratio) | Share of detections in boxes (box area) | Notes |
|---|---|---|---|---|---|---|
| 301 | Jewar airport construction (2021-03-05 → 2025-03-04, S2B/S2B) | PASSED | 0.221 | 0.349 / 0.116 (3.0×) | 0.71 (0.45) | ΔNDVI mean −0.020. Only 35 % of the airport parcel flagged. |
| 304 | Puzhal reservoir loss (2018-06-05 → 2019-06-15, S2A/S2B, tile 44PMV) | PASSED | 0.393 | 0.689 / 0.133 (5.2×) | 0.82 (0.47) | ΔNDWI mean −0.120; 86.9 % of SCL water-flip pixels flagged |
| 306 | Puzhal reservoir gain (2019-06-15 → 2021-05-30, S2B/S2A) | INCONCLUSIVE | 0.538 | 0.908 / 0.255 (3.6×) | 0.73 (0.43) | 99.2 % of SCL water-flip pixels flagged. SCL clear-both 0.888 < 0.9 (T2 SCL "unclassified" 10.8 %), so the pre-declared rule marks it INCONCLUSIVE. |
| 307 | Kancha Gachibowli clearing (2025-03-28 → 2025-04-07, S2B/S2B) | PASSED | 0.028 | — (not localized) | — | Indistinguishable from the control floor (0.024–0.031); scene-wide ΔNDVI +0.041 |
| 308 | Wayanad debris flow (2024-02-10 → 2025-02-09, S2A/S2B) | PASSED | 0.022 | 0.113 / 0.008 (14.5×) | 0.69 (0.13) | The scar is detected as a continuous strip. 16.3 % of all detections fall on SCL topographic/cloud-shadow pixels. |

### Controls

| ID | Control | Status | Changed | Reading |
|---|---|---|---|---|
| 305 | Same datatake, two tile products (44PMV vs 44PLV, 2019-06-15, 3 s apart) | PASSED | **0.0000** (magnitude p95 0.004, max 0.044) | Processing floor is zero: per-tile L2A processing adds no detections. NIR shift 0.000 px, correlation peak 0.988. |
| 309 | Chandigarh sectors, 7 days apart (S2B → S2A), both clear | PASSED | 0.024 | Near-simultaneous false-positive floor. Detections are thin lines along street and building edges; global shift 0.035/0.006 px (peak 0.79). |
| 310 | Chandigarh sectors, one year apart, same platform | PASSED | 0.031 | Inter-annual floor over established urban fabric |
| 311 | Chandigarh, T2 under haze/thin cloud (SCL cloud 100 % of AOI) | PASSED | 0.200 | Every detection is on cloud pixels: ΔNDVI mean −0.126, ΔNDWI +0.148. With SCL masking, the provider **refuses** (no co-valid pixels). |
| 312 | Ludhiana farmland, Feb → May same year (wheat → harvested) | PASSED | 0.825 | Phenology alone produces the largest "change" in the pack (ΔNDVI mean −0.500) |

### Gate rejections (all real products)

| ID | Case | Result |
|---|---|---|
| 302 | Jewar pair uploaded with T2 declared first | 422 `acquisition_order_invalid` ✓ |
| 313 | Hyderabad: T1 from UTM-43 tile 43QHV, T2 from UTM-44 tile 44QKE (78°E zone boundary) | 422 `reprojection_required`, `dimensions_incompatible`, `grid_alignment_incompatible` ✓ |
| 314 | Jewar with T2 window offset by 2 columns (20 m) | 422 `grid_alignment_incompatible` ✓ |

### Failed / invalidated (kept)

| ID | Case | Result |
|---|---|---|
| 303 | `water-puzhal-lake-2018-2019` (tile 44PLV) | **FAILED: acquisition defect.** The AOI ran ~2.4 km past the east edge of 44PLV. rasterio silently clipped the read to 410 of 654 columns, and the writer stretched those columns across the declared window, so the rasters were geometrically wrong while their metadata claimed the full AOI. It was caught when a same-datatake 44PLV-vs-44PMV comparison correlated only 0.11. Acquisition now fails closed on out-of-tile windows and clipped reads (`scripts/acquire_s2_pair.py`, with a test using this exact window). The pair was replaced by 304. Its original provenance stays committed in `feea1ad`. The first, flawed version of the cross-tile control built on these rasters was discarded before registration. |

## Where the baseline succeeds and fails

**Succeeds (detection of spectrally large change):**
- Water extent change. 87–99 % of pixels whose SCL water class flipped are flagged, and detections concentrate 3.6–5.2× inside the water boxes (304, 306).
- A vegetated → bare debris-flow scar. It is detected as a clean continuous strip with a 14.5× inside/outside ratio (308).
- The processing path is clean: a same-datatake cross-tile pair gives exactly zero detections (305).
- The pair gate rejects all three real incompatible pairs with the correct reason codes (302, 313, 314).

**Fails or is confounded:**
- **Bare soil → built surface is largely invisible.** At Jewar the acquired parcel was already fallow and bare at T1, so only 35 % of it is flagged, even though the whole parcel became runway, apron and terminal. Built and bare surfaces both have low NDVI and similar NDWI. The spectral-index rule is structurally blind to this transition, so a genuine construction event was under-detected.
- **Phenology dominates.** Same-year crop harvest flags 82.5 % of the scene (312), and cropland outside the Jewar airport flags 11.6 % between two years at the same date. By this measure, seasonal/agricultural change is "larger" than the construction event.
- **Clouds and haze produce silent false positives.** Thin cloud over an unchanged city yields 20 % "changed" with a confident-looking answer (311). The production path has no cloud mask: `dataMask` means data presence. With SCL masking, the provider fails closed, which is the safer behaviour.
- **Topographic shadow:** in the Wayanad hills, 16.3 % of all detections lie on SCL shadow pixels (308).
- **Subtle vegetation loss is below the noise floor.** The reported Kancha Gachibowli clearing yields 2.8 % changed, inside the 2.4–3.1 % range of the urban controls. The ΔNDVI render shows a coherent negative patch in the scene centre that *may* be the cleared parcel. Because it was not localized before the run, it is recorded as an unverified observation, not a detection.
- **Edge effects set a ~2.4 % floor even for a 7-day clear pair.** The global residual shift is < 0.05 px, so whole-image misregistration is not the cause. The linear street/building-edge pattern is consistent with view-geometry (different relative orbits, S2B → S2A) and sub-pixel edge effects. That is a hypothesis; it was not isolated.

**Illumination/season:** within-pair sun-elevation differences are < 1° for 301/304/305/308, 1.2°–5.6° for 306/307/309/310/311, and 27° for the seasonal control 312 (42° → 70°). The seasonal control's result therefore mixes phenology with illumination change and should not be attributed to either alone.

**Sensor effects:** in 309, 3.35 % of T2 (S2A) pixels have negative B02 reflectance, a blue-band dark-pixel artefact of the L2A offset convention. B02 is not used by the indices, so the provider is unaffected. The mixed-platform 7-day control (309, S2B → S2A, 0.024) is not above the same-platform inter-annual control (310, 0.031). The two differ in interval as well as platform, so this does not isolate an inter-sensor effect.

## Observations for other tracks

- **Validation gap:** the gate accepted two tile products of one datatake, 3 s apart, as a bi-temporal pair (305). That is correct under the current contract (T1 < T2), but no minimum interval or "same acquisition" check exists.
- **Upload cap:** a 5-band float32 window of ~800 × 800 px is close to the 20 MiB `/api/scenes` limit (the largest here was 5.6 MB compressed). Larger AOIs would need tiling or int16 reflectance.

## Limits

- Ten accepted real pairs and three rejection cases, all in India, cannot establish accuracy or generalization.
- There are no pixel labels. Reference boxes are coarse rectangles that include unchanged pixels, so inside rates understate and outside rates overstate performance.
- SCL is itself a spectral classifier. It serves as a partially independent diagnostic, not ground truth.
- The registry status thresholds (0.5 valid, 0.9 clear, 0.05 cloud) were fixed in code before the first evaluation run and not tuned afterwards. Case 306 stays INCONCLUSIVE even though its result looks informative.

## Reproduce

```bash
# acquisition (public, no credentials; ~307 MB transfer; rasters to ~/satquery-data/change/)
python3 scripts/acquire_s2_pair.py urban-jewar-airport-2021-2025 water-puzhal-lake-2018-2019-pmv \
  control-puzhal-same-datatake-cross-tile-2019 water-puzhal-lake-2019-2021 vegetation-kancha-gachibowli-2025 \
  disaster-wayanad-landslide-2024-2025 control-chandigarh-5day-2025 control-chandigarh-interannual-2024-2025 \
  control-chandigarh-cloud-2025 control-ludhiana-seasonal-2024 reject-hyderabad-cross-utm-2025 \
  reject-jewar-2px-shift-2021-2025
# previews for reference boxes (before running the provider)
python3 eval/change_real_pairs.py --rgb-only
# evaluation (add --register <first_id> only for a new, never-used ID block)
python3 eval/change_real_pairs.py
python3 -m pytest -q eval/test_change_real_pairs.py scripts/test_acquire_s2_pair.py
```

Re-acquisition should reproduce the recorded raster SHA-256 values as long as the public C1 COGs are unchanged. Compare them against the provenance files. The evaluation environment was Python 3.13.0, rasterio 1.5.1 / GDAL 3.12.4, numpy 2.4.2, on an Apple M3 Pro (CPU only), as captured in each registry record.
