# Optical/SAR sensor necessity — Hyderabad reservoirs, 2020

Status: **pre-registered** (this section was committed before any metric was computed).

## Question

Where Sentinel-2 is optically obstructed, does Sentinel-1 recover open-water / land separation that optical alone cannot, when both are judged against a reference that is independent of both sensors?

## Case design

- AOI: lon/lat bbox `78.28, 17.30, 78.50, 17.45` (Osman Sagar, Himayat Sagar, Hussain Sagar and surrounding urban/rural land, Hyderabad, Telangana).
- Grid: native S2 10 m grid of MGRS tile 44QKE (EPSG:32644), snapped outward around the bbox: 2363 × 1694 pixels.
- Source: Microsoft Planetary Computer STAC, anonymous access. `sentinel-2-l2a` (Copernicus Sentinel data terms), `sentinel-1-rtc` (CC-BY-4.0), `jrc-gsw` (JRC Global Surface Water, Pekel et al. 2016; attribution required).
- Dates were selected by S2 tile cloud cover and S1–S2 time separation only, before any metric was computed. All three share S1 relative orbit 165 (descending) and S2 relative orbit 19.

| Case | Role | S2 item | S1 item | S1–S2 separation |
|---|---|---|---|---|
| hyderabad-20200717 | cloudy test (tile cloud 48.4%) | `S2A_MSIL2A_20200717T050701_R019_T44QKE_20200816T104934` | `S1A_IW_GRDH_1SDV_20200717T003854_20200717T003919_033487_03E166_rtc` | ~4.5 h |
| hyderabad-20200915 | cloudy test (tile cloud 91.5%) | `S2A_MSIL2A_20200915T050701_R019_T44QKE_20200918T110717` | `S1A_IW_GRDH_1SDV_20200915T003857_20200915T003922_034362_03FEE7_rtc` | ~4.5 h |
| hyderabad-20201114 | clear-sky control (tile cloud 2.5%) | `S2A_MSIL2A_20201114T051051_R019_T44QKE_20201116T113004` | `S1A_IW_GRDH_1SDV_20201114T003858_20201114T003923_035237_041D68_rtc` | ~4.5 h |

S2 B02/B03/B04/B08 are read on their native grid without resampling (processing baseline 02.12, so no −1000 DN offset). The S1 RTC grid of every case was verified to be pixel-aligned with the S2 grid, so S1 was read without resampling. SCL (20 m) and GSW (EPSG:4326, 0.00025°) were mapped to the grid with nearest neighbour. Per-file SHA-256, unsigned asset URLs, and retrieval times are in `data/manifests/optical-sar/hyderabad-2020.v1.json`. Rasters stay outside git.

## Reference (independent of S1 and S2)

- Reference water: GSW occurrence ≥ 90 %, eroded 3 px (30 m).
- Reference land: GSW occurrence = 0 %, eroded 3 px.
- Every other pixel (transitional 1–89 %, nodata, boundary) is excluded.

GSW is Landsat-derived (1984–2020). Pixels that were water in ≥ 90 % of 37 years of observations, or never water, are assumed to hold that state on the test date. This assumption is weakest at reservoir margins, which the thresholds and erosion exclude.

## Methods (thresholds frozen, not retuned)

Thresholds are read at run time from the frozen CDSE rule in `data/manifests/cdse/sensor-necessity-results.v1.json`. That rule was constructed on Jiangsu and replicated on Rotterdam; it has never seen this AOI.

- optical-only: NDWI `(B03−B08)/(B03+B08)` > 0.048095703125
- SAR-only: VV ≤ 0.053388 **and** VH ≤ 0.009292 (linear γ⁰)
- fusion (frozen AND): optical-only ∧ SAR-only, pixel level (component filtering and boundary targets of the proxy benchmark are not applied)
- fusion (cloud-gated): SAR-only where S2 SCL ∈ {3 cloud shadow, 8, 9 cloud, 10 thin cirrus}, optical-only elsewhere

## Metrics

On joint-valid pixels, stratified into all / optically clear / optically obstructed:

- water recall = TP/(TP+FN)
- land specificity = TN/(TN+FP)
- balanced accuracy = mean of the two
- water IoU = TP/(TP+FN+FP)

A stratum is evaluable only with ≥ 500 reference pixels of each class. Pixels are spatially autocorrelated, so pixel counts are not independent samples and no pixel-level confidence intervals are reported. The unit of replication is the case: 3 dates, 1 AOI.

## Pre-declared hypotheses

- **H1 (necessity):** on both cloudy test dates, in the optically obstructed stratum, SAR-only balanced accuracy exceeds optical-only by ≥ 0.10.
- **H2 (fusion value):** on both cloudy test dates, cloud-gated fusion balanced accuracy on all pixels is ≥ both optical-only and SAR-only.
- **H3 (control, no necessity when clear):** on the clear control date, in the optically clear stratum, optical-only balanced accuracy is ≥ SAR-only. If H3 fails, SAR would be adding value even without cloud, and the cloudy-date gain cannot be attributed to obstruction alone.

Each experiment's status records whether it executed with complete evidence (PASSED / FAILED / INCONCLUSIVE). Hypothesis outcomes are reported separately.

---

## Results (added after execution; the sections above are unchanged from pre-registration commit `259a147`)

The primary results were produced by the evaluator exactly as committed in `259a147`. Afterwards, an optional `--exclude-cols` post-hoc sensitivity mode was added to `eval/sensor_necessity.py`. It defaults to off and changes no threshold, reference, stratum, metric or hypothesis. The registry records were written from a verification re-run with that edited evaluator, and they reproduced every primary metric exactly (the re-run asserts equality). This is why those records show `repository.dirty = true` at `259a147`.

### Pair validation and provider

All three pairs went through canonical ingestion (`backend.services.ingest_scene`) and `scene_compatibility(..., "optical_sar")`. Each was `eligible` with 9 verified checks, footprint overlap 1.0, resolution ratio 1.0, S1–S2 interval ≈ 16,070–16,300 s, and no warnings or unknown metadata. `optical-sar-deterministic` ran on each pair with `confidence: null` and joint-valid fraction 1.0.

The provider's optical summary includes cloud pixels, because `dataMask` does not encode cloud (by design; see `docs/optical-sar-baseline.md`). Its NDWI mean was −0.122 (07-17), −0.002 (09-15) and −0.339 (11-14). On the cloudy dates it therefore describes cloud tops, not the surface. **Finding:** the provider's optical evidence is not cloud-aware. This was documented only; the provider was not changed.

### Primary metrics — balanced accuracy (water IoU)

| Case | Stratum | Ref water / land px | Optical-only | SAR-only | Fusion AND (frozen) | Fusion cloud-gated |
|---|---|---|---|---|---|---|
| 07-17 cloudy (82.6 % of valid px obstructed) | all | 25,615 / 3,524,693 | 0.480 (0.002) | **0.834** (0.460) | 0.507 | 0.820 (0.451) |
| | clear | 1,024 / 616,497 | 0.641 (0.097) | **0.994** (0.261) | 0.643 | 0.641 (0.097) |
| | obstructed | 24,591 / 2,908,196 | 0.469 (0.000) | **0.827** (0.483) | 0.502 | 0.827 (0.483) |
| 09-15 cloudy (95.8 %) | all | 25,615 / 3,524,693 | 0.605 (0.011) | 0.963 (0.780) | 0.768 | **0.964** (0.768) |
| | clear | 2,170 / 132,007 | **0.996** (0.732) | 0.988 (0.878) | 0.988 | **0.996** (0.732) |
| | obstructed | 23,445 / 3,392,686 | 0.578 (0.010) | **0.961** (0.772) | 0.748 | **0.961** (0.772) |
| 11-14 control (8.9 %) | all | 25,615 / 3,524,693 | **0.998** (0.843) | 0.991 (0.574) | 0.992 | 0.997 (0.815) |
| | clear | 16,253 / 3,304,901 | **0.999** (0.808) | 0.991 (0.487) | 0.993 | **0.999** (0.808) |
| | obstructed | 9,362 / 219,792 | **0.995** (0.911) | 0.990 (0.829) | 0.990 | 0.990 (0.829) |

Registry: `SQ-20260926-401` (07-17), `-402` (09-15), `-403` (11-14). Result files: `eval/results/sensor-necessity/hyderabad-<date>.json`.

### Hypothesis outcomes

- **H1 holds on both cloudy dates.** In the obstructed stratum, SAR-only minus optical-only balanced accuracy is +0.358 (07-17) and +0.383 (09-15).
- **H2 fails.** On 07-17, cloud-gated fusion (0.820) is below SAR-only (0.834); on 09-15 it is +0.0008 above. Cause on 07-17: in pixels SCL marks clear, NDWI with the frozen threshold detects only 28.6 % of reference water (median NDWI on clear reference water is 0.040, below the 0.048 threshold). Routing those pixels to optical therefore loses water that SAR finds. Undetected haze, adjacency to cloud, and turbid monsoon water are possible causes, but none was established.
- **H3 holds.** On the clear control date, optical-only (0.999) ≥ SAR-only (0.991) on clear pixels; SAR's water IoU is lower (0.487 vs 0.808) because of more land false positives.

### Reference validity audit (the reference assumption failed in part of the AOI)

`scripts/check_reference_water_state.py` measured NDWI on clear-sky S2 scenes independent of the test dates. It checked three column bands of the grid: west = Osman Sagar (cols 0–700), centre = Himayat Sagar (700–1500), east = Hussain Sagar (1500–2363).

| Check | Scene | West | Centre | East |
|---|---|---|---|---|
| `SQ-20260926-404` | 2020-05-03 / 05-23 / 05-28 | **0.000 / 0.000 / 0.000** | 1.000 / 0.846 / 0.702 | 0.761 / 0.997 / 0.986 |
| `SQ-20260926-405` | 2020-08-31 | 0.601 | 0.985 | 0.956 |

(Values are the fraction of unobstructed GSW reference-water pixels with NDWI > threshold.)

Before the July test date, Osman Sagar's GSW "permanent water" pixels were not water: the reservoir was drawn down. They hold 9,362 of the 25,615 reference-water pixels. On 07-17, SAR-only recall there was 0.123 (VV median −9.7 dB, land-like), consistent with a dry bed. The 07-17 primary SAR recall is therefore a lower bound, and `SQ-20260926-401` is marked **INCONCLUSIVE**. In September the west band had partly refilled (60 %), so part of the 09-15 west-band reference is also uncertain.

### Post-hoc sensitivity (not pre-registered)

Excluding the west band (`--exclude-cols 0 700`; records `SQ-20260926-406/407/408`):

| Case | Stratum | Optical-only | SAR-only | Fusion cloud-gated |
|---|---|---|---|---|
| 07-17 | obstructed | 0.468 | **0.992** | **0.992** |
| 07-17 | all | 0.482 | **0.992** | 0.970 |
| 09-15 | obstructed | 0.617 | **0.984** | **0.984** |
| 09-15 | all | 0.644 | 0.985 | **0.986** |
| 11-14 | clear | **0.999** | 0.991 | **0.999** |

The direction of every conclusion is unchanged. H2 still fails on 07-17 (0.970 < 0.992).

## What this does and does not support

Supported, for this AOI and these dates: when S2 is cloud-obstructed, S1 VV/VH (frozen thresholds from another continent, not retuned) separates stable open water from stable land far better than S2 NDWI. On the two monsoon dates, balanced accuracy in obstructed pixels was 0.83–0.99 vs 0.47–0.62. When the sky is clear, optical is at least as good (control). **SAR necessity is conditional on obstruction.**

Not supported:
- **Fusion value beyond SAR alone.** The cloud-gated fusion never beat SAR-only by a material margin, and lost to it on 07-17.
- **Semantic land cover, flood mapping, or change.** The task is binary stable-water vs stable-land on pre-selected reference pixels.
- **Generalisation.** One AOI, one S1 relative orbit (165 descending), three dates, one reference product.
- **Calibrated confidence.** None is produced.

## Limitations

- The frozen SAR thresholds were built on CDSE Sentinel Hub γ⁰ (terrain) and applied to Planetary Computer RTC γ⁰ without cross-calibration. The processors differ; agreement here is not a guarantee elsewhere.
- GSW ends in 2020 and is Landsat 30 m. Its "permanent water" failed for a drawn-down reservoir (see audit). Reference pixels are heavily imbalanced (≈ 0.7 % water), so balanced accuracy and IoU are both reported.
- SCL is itself an optical classifier. Stratum membership depends on it, and it appears to miss some contamination (07-17 clear-stratum NDWI).
- Pixels are spatially autocorrelated; no confidence intervals are claimed. The case is the replication unit (n = 3).
- Wind-roughened water and radar layover or shadow did not dominate here but can in other AOIs.

## Reproduction

```bash
python -m venv --system-site-packages $SATQUERY_DATA/venvs/optical-sar
$SATQUERY_DATA/venvs/optical-sar/bin/pip install pystac-client planetary-computer
# per case (IDs in data/manifests/optical-sar/hyderabad-2020.v1.json)
python scripts/acquire_pc_s1s2_case.py --s2-id <S2> --s1-id <S1> --bbox 78.28 17.30 78.50 17.45 --out $SATQUERY_DATA/optical-sar/<case>
python -m eval.sensor_necessity $SATQUERY_DATA/optical-sar/<case> --out eval/results/sensor-necessity/<case>.json
python -m eval.sensor_necessity $SATQUERY_DATA/optical-sar/<case> --exclude-cols 0 700 --out eval/results/sensor-necessity/<case>.posthoc-exclude-west.json
python -m scripts.check_reference_water_state $SATQUERY_DATA/optical-sar/hyderabad-20200717 --datetime 2020-05-01/2020-07-10 --bands '{"west": [0, 700], "centre": [700, 1500], "east": [1500, 2363]}'
```

The acquisition hashes in the manifest are for the rasters written on 2026-09-26. Re-acquisition may yield different bytes if Planetary Computer reprocesses a product; compare hashes before claiming a reproduction.
