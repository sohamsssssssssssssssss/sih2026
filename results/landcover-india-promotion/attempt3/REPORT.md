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
