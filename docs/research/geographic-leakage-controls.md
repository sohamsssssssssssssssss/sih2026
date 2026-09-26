# Geographic leakage controls

Remote-sensing samples that differ in filename can share ground, landscape, or even the exact footprint. SatQuery therefore audits splits by distance, not by identifier.

## Utility: `data/spatial_blocking.py`

- `block_ids(lons, lats, block_size_m)`: projects to EPSG:6933 (equal-area metres) and returns a square-block ID per sample.
- `assign_blocks(blocks, fractions, seed)`: assigns **whole blocks** to splits. Blocks are ordered by a seeded SHA-256 and filled greedily to the target fractions by sample count. It is deterministic, and no block straddles two splits.
- `nearest_distances_m(query, reference)`: great-circle distance (spherical Earth, R = 6,371,008.8 m; k-d tree on unit vectors) from each query point to its nearest reference point.
- `leakage_audit(lons, lats, splits, train_split, radii_m)`: for each non-train split, gives the nearest-train distance summary (min / p05 / median) and counts within each radius.

Tests: `data/test_spatial_blocking.py`.

### How to split a new dataset

1. Get sample-centre lon/lat. If only projected footprints exist, use their centres.
2. Choose `block_size_m` well above the task's spatial autocorrelation range. For 10 m Sentinel scenes, start at ≥ 10 km; land cover and water extent autocorrelate over kilometres.
3. `splits = assign_blocks(block_ids(lons, lats, block_size_m), {"train": .7, "validation": .15, "test": .15}, seed)`.
4. **Always run `leakage_audit` and publish it.** Blocking alone does not guarantee separation: samples in adjacent blocks can still be metres apart. The audit, not the block size, is the evidence. Where edge contact matters, drop samples of non-train splits within a buffer of any train sample (e.g. `nearest_distances_m < buffer`). Report how many were dropped.
5. Also check identical footprints at different dates, e.g. `same_footprint_across_splits` in `scripts/audit_bigearthnet_split_leakage.py`, which keys on tile + x/y index.

Never change an existing official test split to fix leakage. Report it, and report results on the leak-free subset **in addition to** the official split.

## Applied: BigEarthNet v2 official split and SatQuery pilot

`python -m scripts.audit_bigearthnet_split_leakage --metadata <metadata.parquet>` → `data/manifests/bigearthnet/split-leakage-audit.v1.json`, registry `SQ-20260926-410`. The re-run output is byte-identical.

Patch footprints come from patch IDs without imagery. `…_T<tile>_<x>_<y>` is the 1200 m patch at x·1200 m east and y·1200 m south of the Sentinel-2 tile's upper-left corner. Tile corners come from the 10 m B02 geotransform of one Planetary Computer S2 L2A product per tile (`data/manifests/bigearthnet/s2-tile-origins.v1.json`). They are **not** MGRS 100 km corners: an MGRS-corner derivation was tried first and failed the reference-map check by 20–40 m.

The derivation reproduces **all 100 tracked reference-map footprints exactly**, covering 28 of the 54 tiles. The other 26 tiles rely on the fixed S2 tiling grid. Scope: the 480,038 patches in `metadata.parquet`; the 69,450 snow/cloud/shadow patches are not included.

Distances are between patch centres. 1,200 m means edge-adjacent (touching) patches, 1,700 m adds diagonal neighbours, and 600 m would mean overlapping footprints.

| Split vs official train | Patches | Min nearest train | Median | ≤ 1.2 km (touching) | ≤ 1.7 km | ≤ 10 km |
|---|---:|---:|---:|---:|---:|---:|
| test | 119,825 | **11.95 km** | 20.4 km | 0 | 0 | 0 |
| validation | 122,342 | 1.20 km | 6.0 km | **11,444 (9.4 %)** | 14,844 (12.1 %) | 105,036 (85.9 %) |

The same footprint (tile + x/y) appears in more than one split **0 times** (233,966 unique footprints). No two patches in different splits overlap.

**Interpretation.** The official **test** split is geographically buffered from train (≥ ~12 km everywhere), consistent with a spatially blocked design. The official **validation** split is **not**: about one validation patch in ten touches a train patch, and most lie within 10 km. Validation accuracy will be optimistic relative to test. Use validation only for model selection, and never quote it as held-out performance.

### SatQuery 100-pair pilot (`pilot-candidates.v1.json`)

- **Within the pilot:** the nearest pilot-train patch is ≥ 10.3 km (validation), ≥ 15.7 km (test) and ≥ 15.6 km (bench). Nothing is within 10 km, so the pilot split **does not leak internally**.
- **Against the full official train split:** pilot test and bench are ≥ 11.97 km away (0 within 10 km). **4 of the 25 pilot validation patches touch an official train patch**, and 19 of 25 are within 10 km. This is inherited from the official validation split.
- The declared pilot lat/lon matches the derived patch centres to within 2 × 10⁻⁹ m.

The 4 edge-touching pilot validation patches, with the centre distance to the nearest official train patch (computed with `centres_lonlat` + `nearest_distances_m` against official train):

- `S2A_MSIL2A_20170813T112121_N9999_R037_T29SNC_13_43`: 1,197.6 m
- `S2B_MSIL2A_20180421T114349_N9999_R123_T29UPU_77_56`: 1,196.1 m
- `S2A_MSIL2A_20180508T104031_N9999_R008_T31UGR_13_44`: 1,196.1 m
- `S2A_MSIL2A_20170818T103021_N9999_R108_T32TMT_13_74`: 1,196.9 m

(Edge neighbours are 1,200 m apart in UTM; the spherical distance is slightly shorter.)

Recommendation: keep the pilot's test and bench roles as they are. Exclude these 4 validation patches from any calibration or selection experiment where adjacency to training data matters.

## Also applies to

- **RSVQA-LR** and any other dataset with coordinates: run `leakage_audit` on the official split before quoting held-out numbers. Image-ID disjointness (already verified for RSVQA-LR) does not imply geographic separation.
- **Sensor-necessity cases** (`docs/research/optical-sar-sensor-necessity.md`): all three cases share one AOI by design (same place, different conditions). They are an evaluation of conditions, not of geographic generalisation, and must not be pooled with any training data from that AOI.
