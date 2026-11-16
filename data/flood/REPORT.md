# T14 verified flood-pack report

Builder commit: `1b48caf9182791ea76e745c31c048b206dc1d635`.
Local pack: `/Users/soham/satquery-data/flood/packs/sen1floods11-hand-v1`. Pack data have not been published or uploaded.
MANIFEST SHA-256: `250c230b4698ac4b7cab33e2c1666ebed2b1699455aafca5e56fb429ac6c60ad`.

Gate: `10 passed, 260 warnings in 6.31s`. Warnings are rasterio/affine pending deprecations. The separate real-pack test verified every file hash, array shape/dtype, index identity, all 446 chip/layer hashes, and all alignment measurements.

## Official splits

| Split | Chips | India chips |
|---|---:|---:|
| train | 252 | 40 |
| valid | 89 | 14 |
| test | 90 | 14 |
| bolivia | 15 | 0 |

Zero overlap; official CSV row order retained. No weak labels or permanent-water dataset downloaded.

## Events

| Event | Chips |
|---|---:|
| Bolivia | 15 |
| Ghana | 53 |
| India | 68 |
| Mekong | 30 |
| Nigeria | 18 |
| Pakistan | 28 |
| Paraguay | 67 |
| Somalia | 26 |
| Spain | 30 |
| Sri-Lanka | 42 |
| USA | 69 |

## Quality

| Measurement | Value |
|---|---:|
| chips_with_valid_labels | 441 |
| min | 0.0 |
| median | 2.654266357421875 |
| mean | 11.097016298355461 |
| max | 100.0 |
| pooled_valid_pixel_water_pct | 10.628901109983758 |
| Chips flagged (>50% nodata union) | 40 |
| S1 NaN band samples | 5023410 |
| S1 Inf band samples | 0 |
| s2_int16_max_pixels (raw spatial count; any band) | 0 |
| s2_ge_10000_pixels (raw spatial count; any band) | 9055 |

Water percentages exclude invalid LabelHand pixels; five chips have no valid label pixels, so their water percentage is unknown (JSON null), never zero-imputed. All chips are retained. Nodata is the union of any invalid band across all five layers; per-layer values and per-band S2 raw counts are in the manifest. Neither S2 raw count is interpreted as physical saturation; physical meaning is unknown.

Label validity exception: Ghana_24858 has a source LabelHand nodata header of 0, while the documented label encoding also calls 0 no-water. Its native mask is honored and it is flagged; zero-valued pixels in this chip are not silently reinterpreted as valid labels. The first sampled label had no nodata header; full per-chip metadata expose this exception.

### Flagged chips (retained)

- Bolivia: `Bolivia_103757`, `Bolivia_294583`
- Ghana: `Ghana_24858`, `Ghana_154838`, `Ghana_134751`, `Ghana_144050`, `Ghana_168875`, `Ghana_161233`, `Ghana_247288`, `Ghana_194723`, `Ghana_26376`, `Ghana_45911`, `Ghana_83200`, `Ghana_135389`, `Ghana_5079`, `Ghana_234935`, `Ghana_132163`, `Ghana_277`, `Ghana_97059`, `Ghana_866994`, `Ghana_83483`
- India: `India_979278`, `India_287642`, `India_135434`, `India_533192`
- Nigeria: `Nigeria_417184`
- Pakistan: `Pakistan_909806`, `Pakistan_712873`, `Pakistan_35915`, `Pakistan_366265`, `Pakistan_1036366`, `Pakistan_9684`, `Pakistan_664885`, `Pakistan_528249`
- Paraguay: `Paraguay_224845`, `Paraguay_34417`
- Somalia: `Somalia_989553`
- Sri-Lanka: `Sri-Lanka_450918`
- USA: `USA_231124`, `USA_198411`

## Verified encoding and geometry

All chips are 512 x 512, all five layers are EPSG:4326. The per-chip source CRS, native transform, band tags, dtype and nodata values are preserved in MANIFEST.json. Label encoding observed: -1 invalid, 0 no-water, 1 water, with the header exception above.

| Layer | Source dtype | Packed dtype | Source nodata headers |
|---|---|---|---|
| S1Hand | float32 | float32 | NaN |
| S2Hand | int16 | int16 | 0.0 |
| LabelHand | int16 | int8 | 0.0, None |
| S1OtsuLabelHand | int16 | int8 | -1.0 |
| JRCWaterHand | uint8 | uint8 | None |

S1 band order: VV, VH; units dB according to the official README, consistent with sampled values. No raster unit tag independently establishes units; no conversion applied.
S2 order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12. README: MSI L1C TOA reflectance scaled by 10000 (physical scale 0.0001); raw DN preserved. All observed S2 files are Int16, differing from README UInt16.

README nominal resolution: 10 m. Native grids are angular; approximate per-chip local-UTM pixel dimensions, rather than uniform ground spacing, are stored.
- Approximate column ground pixel dimension: 7.640960930 to 9.986231738 m.
- Approximate row ground pixel dimension: 9.929342119 to 9.974768531 m.
- Largest max_origin_difference_pixels: 4.745835331900433e-10.
- Largest max_pixel_size_difference_pixels: 7.151050400164303e-14.

Alignment limit: 1e-6 pixel for origin and pixel-step differences, exact CRS/shape equality. No resampling. Both maxima are recorded in every index row and independently recomputed by the real-pack test.
Random build re-open: 23 chips (5% rounded up), seed 17. Real-pack verification additionally checks every chip.

## Normalization

Train-only, streaming per-band population mean/std, ignoring source mask nodata and non-finite pixels. S1 stats are dB; S2 stats are raw scaled DN. Validation/test/Bolivia do not enter stats. Exact counts, means and standard deviations are in MANIFEST.json.

## Licence and disk size

Official hand-labeled source and label STAC collections both declare `proprietary`. Reuse terms: **unknown**. README supplies no reuse licence; the GitHub API returned a null licence. Manifest policy: **redistribution: not allowed until licence is confirmed**. The pack remains local; only code, tests and this aggregate verification report are pushed.

- Packed file bytes (including manifest): 4,330,049,185 (4.032673 GiB).
- Raw file bytes (including fetch log and source metadata): 1,754,536,731 (1.634040 GiB).
- Total allocated disk size (`du -sk`, raw + pack): 6,408,200,192 bytes (5.968102 GiB).
- Total logical file bytes: 6,084,585,916 (5.666712 GiB).

Every downloaded file URL, SHA-256 and byte count is recorded in raw/FETCH_LOG.json and its copy inside the pack. No model training, Kaggle upload or data redistribution occurred.
