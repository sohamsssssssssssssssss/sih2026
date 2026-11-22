# Mac VQA parity, attempt 2 (MLX bf16)

**FAIL**

Gate: [attempt 2](../../../docs/gates/VQA-MAC-PARITY-ATTEMPT2.md), commit `3b5c02ea5b11bd1490e22cf27eb5725a4f1b25b3`. Attempt 1 ([report](../REPORT.md)) stays **FAIL**.
Validation manifest SHA-256 `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`. CUDA reference SHA-256 `93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd`. bf16 conversion MANIFEST SHA-256 `5b9f7c8457622f1aa4a9ef59e19f010c7337ee313999aa2b1942dd794ec8be2e`.

## Accuracy (MLX / CUDA)

| Attempt | Condition | Macro | Presence | Comparison | Count | Rural/urban | Agreement | Malformed |
|---|---|---|---|---|---|---|---:|---:|
| 1 (8-bit) | base | 0.4604 / 0.4683 | 0.540 / 0.550 | 0.680 / 0.690 | 0.060 / 0.060 | 0.562 / 0.573 | 99.23% | 5 / 4 |
| 1 (8-bit) | b13 | 0.7038 / 0.6957 | 0.870 / 0.880 | 0.930 / 0.910 | 0.240 / 0.240 | 0.775 / 0.753 | 95.63% | 0 / 0 |
| 2 (bf16) | base | 0.4683 / 0.4683 | 0.540 / 0.550 | 0.700 / 0.690 | 0.060 / 0.060 | 0.573 / 0.573 | 95.89% | 7 / 4 |
| 2 (bf16) | b13 | 0.6926 / 0.6957 | 0.870 / 0.880 | 0.920 / 0.910 | 0.250 / 0.240 | 0.730 / 0.753 | 94.09% | 0 / 0 |

## Latency and memory

| Attempt | Condition | Load (s) | First (s) | Median (s) | p95 (s) | Peak footprint (GiB) | MLX peak (GiB) | Crashes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (8-bit) | base | 2.890 | 0.907 | 0.382 | 0.428 | 5.403 | 4.728 | 0 |
| 1 (8-bit) | b13 | 2.045 | 0.862 | 0.389 | 0.444 | 5.371 | 4.742 | 0 |
| 2 (bf16) | base | 2.439 | 1.314 | 0.419 | 0.489 | 7.921 | 7.423 | 0 |
| 2 (bf16) | b13 | 2.071 | 2.206 | 0.445 | 0.546 | 8.014 | 7.436 | 0 |

Paired B13 minus base macro: attempt 1 +0.2434 CI [+0.1982, +0.2889]; attempt 2 +0.2243 CI [+0.1805, +0.2688].
Thirty-question B13 soak: zero crashes; growth 0.000 GiB MLX active plus cached, 0.001 GiB process RSS (attempt 1: 0.000 / 0.001).

## Memory headroom (attempt 2 only; pass needs at least 3 GiB available)

| Measurement | Demo app | Minimum available (GiB) | MLX peak (GiB) | Samples |
|---|---|---:|---:|---:|
| with app | running | 1.81 | 7.436 | 66 |
| alone (context) | stopped | 1.84 | 7.436 | 65 |

Available = (free + inactive + speculative pages) × page size from `vm_stat`, sampled every 0.25 s from before the model load until after the 30th answer.

## Checks

| Check | Attempt 1 | Attempt 2 |
|---|---|---|
| macro within three points | PASS | PASS |
| paired ci excludes zero | PASS | PASS |
| latency | PASS | PASS |
| crashes | PASS | PASS |
| malformed | FAIL | FAIL |
| soak | PASS | PASS |
| memory headroom with app | — | FAIL |

## Malformed base answers (attempt 2)

| Sample | Type | MLX bf16 | MLX 8-bit (attempt 1) | CUDA | Class |
|---|---|---|---|---|---|
| validation-48846 | presence | `Yes, 1` | `Yes, 1` | `Yes, 1` | both |
| validation-55136 | presence | `Yes, 1` | `No` | `No` | MLX only |
| validation-53879 | count | `90%` | `90%` | `90%` | both |
| validation-49157 | presence | `Yes, 1` | `Yes, 1` | `Yes, 1` | both |
| validation-56187 | comp | `1` | `1` | `1` | both |
| validation-53016 | presence | `0` | `No` | `No` | MLX only |
| validation-52080 | presence | `Yes, 1` | `Yes` | `Yes` | MLX only |
