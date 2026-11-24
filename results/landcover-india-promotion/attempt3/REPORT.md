# Attempt 3 data preparation

The G4 amendment was pushed as `09f91e7` before data preparation. The WorldCover builder and original pack hashes match the registration. All 10 new regions produced 300 patches and passed the shape, label and nodata checks. G4 uses the same label grids and the integration fetch at `2f66a93`. The collector reads a 128-pixel source window with the app code and bilinearly aligns it to the 120-pixel label grid. No attempt-3 weights were trained.

## Composite patches

Class positive counts follow the 5% WorldCover rule. Download GB covers accepted source bytes, including bytes retained from an interrupted transfer. Minutes cover the whole regional operation.

| Region | Patches | Trees | Shrub/grass | Cropland | Built | Water | Wetland | Download GB | Minutes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bastar | 300 | 292 | 264 | 216 | 2 | 6 | 0 | 0.348 | 7.4 |
| cauvery | 300 | 222 | 150 | 257 | 29 | 47 | 20 | 0.327 | 6.6 |
| chilika | 300 | 61 | 75 | 83 | 3 | 275 | 37 | 0.164 | 3.7 |
| gangetic | 300 | 183 | 79 | 295 | 101 | 29 | 11 | 0.320 | 6.7 |
| garhwal | 300 | 296 | 230 | 75 | 10 | 18 | 0 | 0.326 | 9.4 |
| hyderabad | 300 | 207 | 286 | 240 | 188 | 13 | 0 | 1.343 | 22.0 |
| kodagu | 300 | 300 | 76 | 8 | 6 | 1 | 0 | 0.326 | 8.1 |
| kutch | 300 | 19 | 273 | 261 | 15 | 4 | 5 | 0.339 | 10.7 |
| marathwada | 300 | 94 | 160 | 287 | 26 | 29 | 0 | 0.326 | 11.2 |
| mizoram | 300 | 300 | 46 | 4 | 5 | 0 | 0 | 0.325 | 6.2 |

## Group assignment

Worst absolute prevalence gap  **27.4375 percentage points**.

| Group | Regions | Trees | Shrub/grass | Cropland | Built | Water |
|---|---|---:|---:|---:|---:|---:|
| A | delhi, rajasthan, sundarbans | 458 | 276 | 400 | 253 | 216 |
| B | assam, kerala, punjab | 607 | 138 | 530 | 279 | 105 |
| C | bastar, chilika, hyderabad, mizoram | 860 | 671 | 543 | 198 | 294 |
| D | cauvery, kodagu, kutch | 541 | 499 | 526 | 50 | 52 |
| E | gangetic, garhwal, marathwada | 573 | 469 | 657 | 137 | 76 |

## G4 collection

| Region | Fold | Paired patches | Skips | Unattempted after target |
|---|---|---:|---:|---:|
| delhi | A | 100 | 16 | 184 |
| rajasthan | A | 100 | 16 | 184 |
| sundarbans | A | 100 | 91 | 109 |
| assam | B | 100 | 17 | 183 |
| kerala | B | 100 | 6 | 194 |
| punjab | B | 100 | 0 | 200 |
| bastar | C | 100 | 0 | 200 |
| chilika | C | 100 | 0 | 200 |
| hyderabad | C | 100 | 1 | 199 |
| mizoram | C | 100 | 8 | 192 |
| cauvery | D | 100 | 200 | 0 |
| kodagu | D | 100 | 12 | 188 |
| kutch | D | 100 | 9 | 191 |
| gangetic | E | 100 | 118 | 82 |
| garhwal | E | 100 | 52 | 148 |
| marathwada | E | 100 | 2 | 198 |

Cauvery had 3 pairs on its first pass. One retry on the same 157 patch IDs that had provider errors recovered 97 pairs. Its region metadata records the first-pass and retry outcomes.

Each region's metadata records patch IDs, both scene item IDs, hashes, and skip reasons. [G4 manifest](g4/MANIFEST.json)

All ten fold/mode combinations passed a CPU inference smoke on real held-out composite patches using the shipped weights. The synthetic tests separately exercise the fold runner and the app normalization path. [CPU smoke receipt](CPU-FOLD-SMOKE.json)

The Kaggle cell pins the code ZIP, new composite ZIP, immutable old pack and Europe checkpoints. The G4 data remain local for CPU scoring. [Artifact hashes](KAGGLE-ARTIFACTS.json)

## Attempt 3 scored results (24 Nov 2026)

**Decision: NOT PROMOTED.** Europe BigEarthNet checkpoints continue to serve land cover in Ask.

The run was scored under the [registered protocol](../../../docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT3.md) at `4695cd9` with the pre-data G4 amendment at `09f91e7`. The scoring used the fixed metrics and thresholds without any further amendment.

| Gate | Registered bar | Measured | Result |
|---|---|---|---|
| G1 · Composite accuracy | Pooled held-out macro mAP ≥ 0.689087241566672 for S1 and ≥ 0.6588597201349471 for S2 | S1 0.883402, S2 0.882793 | **PASS** |
| G2 · Calibration transfer | Per-class logit offsets strictly lower macro ECE in all 10 group-mode comparisons | S1 0/5, S2 2/5 (D and E), total 2/10 | **FAIL** |
| G3 · Scene sanity | After calibration, built on Ludhiana v2 and cropland on Punjab each rank in the top two | Built #1 on Ludhiana v2, cropland #1 on Punjab | **PASS** |
| G4 · Single-date transfer | Pooled held-out macro mAP difference (India minus Europe) and the lower bound of its paired 95% CI are both > 0 in each mode | S1 -0.0406 [-0.0678, -0.0170]; S2 +0.1471 [+0.1262, +0.1680] | **FAIL** |

### G2 per-group macro ECE

Each row evaluates one held-out group using offsets fitted on the other four groups’ held-out predictions.

#### S1

| Group | Raw macro ECE | Calibrated macro ECE | Lower after calibration? |
|---|---:|---:|---|
| A | 0.110068 | 0.120248 | No |
| B | 0.065307 | 0.067060 | No |
| C | 0.042207 | 0.073886 | No |
| D | 0.079220 | 0.096900 | No |
| E | 0.079369 | 0.093654 | No |

#### S2

| Group | Raw macro ECE | Calibrated macro ECE | Lower after calibration? |
|---|---:|---:|---|
| A | 0.082578 | 0.095591 | No |
| B | 0.136841 | 0.142741 | No |
| C | 0.045844 | 0.089076 | No |
| D | 0.071035 | 0.062869 | Yes |
| E | 0.098719 | 0.095367 | Yes |

### G4 single-date macro mAP

India and Europe were scored on the same held-out patches in each mode. The pooled interval is the two-sided 95% percentile interval from the registered paired bootstrap over patches (2,000 draws, seed 17). Group scores are descriptive.

| Group | Paired patches per mode | S1 India | S1 Europe | S1 delta | S2 India | S2 Europe | S2 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 300 | 0.5606 | 0.6757 | -0.1151 | 0.7687 | 0.5339 | +0.2348 |
| B | 300 | 0.5127 | 0.5876 | -0.0749 | 0.7155 | 0.6265 | +0.0890 |
| C | 400 | 0.7292 | 0.6285 | +0.1007 | 0.8831 | 0.8034 | +0.0797 |
| D | 300 | 0.7382 | 0.6805 | +0.0577 | 0.7954 | 0.6371 | +0.1583 |
| E | 300 | 0.6144 | 0.6428 | -0.0284 | 0.6616 | 0.7216 | -0.0600 |
| Pooled | 1,600 | 0.5795 | 0.6201 | -0.0406 | 0.7851 | 0.6380 | +0.1471 |

### Plain reading

- The India recipe cleared both accuracy floors on held-out annual composite patches.
- Both scene sanity checks put the expected class first.
- The calibration offsets lowered error in only two of ten comparisons across held-out groups.
- On the registered single-date patches, India S2 gained about 0.147 mAP and India S1 lost about 0.041 mAP against Europe.
- The two failed gates block promotion. Ask continues to use the Europe BigEarthNet checkpoints.

Sources: [Kaggle G1](G1.json), [CPU G2 and G4](satquery-attempt3-scores.json), [G3 scene scores](satquery-attempt3-g3.json), [Kaggle SHA256SUMS](SHA256SUMS), and [evidence manifest](MANIFEST.json). The manifest includes hashes and sizes for the Kaggle weights and held-out predictions, which are retained in the download.
