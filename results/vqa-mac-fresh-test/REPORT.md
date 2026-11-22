# Mac VQA fresh test

**PASS**

The official RSVQA-LR test sample comes from [Zenodo record 6344334](https://zenodo.org/records/6344334).
There are 400 questions on 100 image clusters. Annotation and archive hashes are recorded in the frozen sources file.

- gate commit `78cf3fcd8b2d205ea22a07315396d622d85177fe`
- manifest sha256 `0ffd6d711f3b3061bcd4485671532ba689d18655f02b88a8d9e5efee94f430f6`
- sources sha256 `803f3822ecd551fa72ee6c512256f9c6e01b47e3e53330ba2ba06bf9549ac204`
- adapter sha256 `7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a`
- conversion manifest sha256 `e4ed41344ad53c218047253edf61d0e60a2c3d151651977ef8aff18094c99e6f`
- base revision `66285546d2b821cf421d4f5eb2576359d3770cd3`

| Condition | Macro [95% CI] | Presence [95% CI] | Comparison [95% CI] | Count [95% CI] | Rural/urban [95% CI] |
|---|---:|---:|---:|---:|---:|
| base | 54.50% [50.50%, 59.00%] | 57.00% [47.00%, 67.00%] | 81.00% [73.00%, 89.00%] | 14.00% [8.00%, 21.00%] | 66.00% [57.00%, 75.00%] |
| b13 | 73.25% [69.99%, 76.50%] | 90.00% [84.00%, 96.00%] | 94.00% [89.00%, 98.00%] | 30.00% [22.00%, 39.00%] | 79.00% [71.00%, 86.00%] |
| constant | 61.50% [56.75%, 66.50%] | 75.00% [66.00%, 84.00%] | 82.00% [74.00%, 89.00%] | 33.00% [24.00%, 43.00%] | 56.00% [46.00%, 66.00%] |
| majority | 57.25% [53.25%, 61.50%] | 76.00% [68.00%, 84.00%] | 66.00% [57.00%, 75.00%] | 31.00% [22.98%, 40.00%] | 56.00% [46.00%, 66.00%] |

| Gate | Decision | Measurement |
|---|---|---|
| G1 | PASS | B13 minus base +0.1875, 95% CI [+0.1424, +0.2325] |
| G2 | PASS | B13 minus majority +0.1600, 95% CI [+0.1225, +0.1975] |
| G3 | PASS | B13 minus constant +0.1175, 95% CI [+0.0750, +0.1600] |
| G4 | PASS | 0/400 malformed (limit 4) |
| G5 | PASS | Median 0.393 s, 0 recorded crashes, 30/30 soak questions, growth GiB {'mlx_bytes': 0.0, 'rss_bytes': 0.0} |
| G6 | PASS | Run 2 reached a 3.227 GiB minimum under the verified setup (3 GiB required). |

| Condition | Load (s) | First (s) | Median (s) | p95 (s) | Peak MLX (GiB) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| base | 2.308 | 0.884 | 0.385 | 0.424 | 4.728 | 1.214 |
| b13 | 1.804 | 0.861 | 0.393 | 0.456 | 4.742 | 2.173 |
| constant | 1.581 | 0.747 | 0.398 | 0.468 | 4.742 | 2.546 |

For context, the prior 389-question validation run reported 46.04% macro for 8-bit base and 70.38% for 8-bit B13. The test draws use new image clusters. Count accuracy is reported separately and has no release threshold. [Validation report](../vqa-mac-parity/REPORT.md)

Swap before the soak worker loaded: `total = 6144.00M  used = 4875.62M  free = 1268.38M  (encrypted)`.
Swap during the soak, first to last sample: `total = 6144.00M  used = 4875.62M  free = 1268.38M  (encrypted)` to `total = 6144.00M  used = 4879.81M  free = 1264.19M  (encrypted)`.
Full vm_stat and swap snapshots are in results.json.

The frozen evaluator initially averaged across images only, leaving four scores per bootstrap draw. The reported intervals use one macro score per draw, averaging across images and question types as registered. Model outputs and gate criteria are unchanged.

## G6 run 1 invalid (registered setup not met: Antigravity + Claude open; 1.463 GiB)

Run 1 remains diagnostic. Its original measurements are in `results.json` and the swap snapshots above.

## G6 run 2

The demo ran with Finder, Terminal, and Safari. The controlling Codex terminal session and its claude-mem helpers remained active, including the worker, Python helper and launcher, and the worker's Claude CLI child. The Chrome DevTools MCP helper was active without a Chrome browser. Both demo ports were free before launch, and both demo servers responded before and after the soak.

All 30 selected questions completed on one loaded 8-bit B13 worker. Across 45 samples at the registered 0.25-second polling interval, available memory was **3.227 GiB minimum** and **3.493 GiB median**. Swap used went from **0 MiB** before the worker loaded to **0.25 MiB** after the soak. See [G6 run 2](g6-run2.json) for snapshots, process inventories, and pinned inputs.
