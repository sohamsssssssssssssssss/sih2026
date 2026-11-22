# Mac VQA parity

**FAIL**

Gate commit `705e5050e3c83b46236bdc3fef867878c090b738`. Validation manifest SHA-256 `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`.
CUDA reference SHA-256 `93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd`. Conversion MANIFEST SHA-256 `e4ed41344ad53c218047253edf61d0e60a2c3d151651977ef8aff18094c99e6f`.

The CUDA reference and image bytes were verified before both runs. The MLX venv is separate because the integration dry run would replace PyArrow 22.0.0 with 26.0.1.

| Condition | Macro MLX / CUDA | Presence | Comparison | Count | Rural/urban | Agreement | Malformed MLX / CUDA |
|---|---|---|---|---|---|---:|---:|
| base | 0.4604 / 0.4683 | 0.540 / 0.550 | 0.680 / 0.690 | 0.060 / 0.060 | 0.562 / 0.573 | 99.23% | 5 / 4 |
| b13 | 0.7038 / 0.6957 | 0.870 / 0.880 | 0.930 / 0.910 | 0.240 / 0.240 | 0.775 / 0.753 | 95.63% | 0 / 0 |

| Condition | Load (s) | First answer (s) | Median (s) | p95 (s) | Peak process footprint (GiB) | MLX active peak (GiB) | Crashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 2.890 | 0.907 | 0.382 | 0.428 | 5.403 | 4.728 | 0 |
| b13 | 2.045 | 0.862 | 0.389 | 0.444 | 5.371 | 4.742 | 0 |

Paired B13 minus base macro gain +0.2434. 95% image-cluster bootstrap CI [+0.1982, +0.2889].
Thirty-question B13 soak completed in the same process with zero crashes. Peak growth was 0.000 GiB in MLX active plus cached memory and 0.001 GiB in process RSS.

| Check | Result |
|---|---|
| macro within three points | PASS |
| paired ci excludes zero | PASS |
| latency | PASS |
| crashes | PASS |
| malformed | FAIL |
| soak | PASS |

The converted checkpoint and its MANIFEST are at `~/models/qwen2.5-vl-3b-mlx-8bit/`. The separate runtime is at `~/models/.venv-satquery-mac-vqa/`. The main integration environment was only inspected by a dry run.

Additional malformed MLX answers, by sample (MLX / CUDA):

- base, `validation-56181`: '1' / 'No'.
