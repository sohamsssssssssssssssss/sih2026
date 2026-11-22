# SatQuery AI status (22 Nov 2026)

Every line names its source. A number with no recorded source is not listed.

## Validated (engineering checks, not model accuracy)

- **Live scene fetch**. Earth Search Sentinel-2 L2A and Planetary Computer Sentinel-1 RTC use 120 × 120 windows. S2 has a 60-second per-read timeout (`ecb651e`). The 20-second overall deadline includes place lookup, STAC search and band reads (`5a778ba`). On expiry Ask displays “Live fetch timed out after Ns — showing cached scene from <date>” when a matching backup exists, or returns a clear refusal. [Integration log](plan/INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture)
- **Optical inputs and backups**. Earth Search inputs could be 0.1 reflectance too dark when the BOA offset was applied twice. The fetch fix is in `84d7f57`. The Ludhiana, Delhi and Mumbai backups were regenerated in `404039f`, and the separate cloudy-Mumbai demo fixture was corrected in `980cab7`. The original TIFFs are retained. The manifest records five active, checksum-verified backup chips. Offline unknown place names are refused. [Backup manifest](../data/fresh-backups/manifest.json) · [Integration log](plan/INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups)
- **Judge-style HTTP acceptance**: 5/5 checks passed on Mac CPU on 18 Nov (land cover, flood, change refusal, outside-India refusal, cloud refusal), each with a trace. Source: `docs/plan/E2E_ACCEPTANCE.md:3`, `:9-13`.
- **Latest test totals**. See the integration log for the full suite and production browser checks. [Integration log](plan/INTEGRATION_LOG.md)

## Experimental

- **Land cover**. Europe-trained BigEarthNet checkpoints serve uncalibrated scores with `confidence: null`. Built areas ranked first in live Ludhiana (**0.955**) and Delhi (**0.949**), and cropland ranked first on the Punjab fixture (**0.378**). These are uncalibrated presence scores. The transfer benchmark still records an India–Europe gap (S1 0.516 vs 0.688, S2 0.576 vs 0.915 mAP). [Recorded API results](plan/INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups) · [Transfer results](plan/RESULTS.md)
- **India land-cover promotion closed**. G1 passed on held-out regions (S1 0.717 vs 0.689 floor, S2 0.688 vs 0.659 floor). Both calibration attempts failed (Platt and temperature scaling), with opposite biases across the region folds. The original Ludhiana G3 failure came from the BOA offset bug. Corrected Ludhiana and Punjab both passed G3 in a record-only rerun. The India weights were not shipped. [India promotion decision](../results/landcover-india-promotion/DECISION.md)
- **Flood water (Sentinel-1 per-scene Otsu)**: India-test IoU 0.631 vs gate 0.65: FAIL; not tuned afterwards. India-test is 14 chips from one event. Source: `outputs/T07c-flood.md:3`, `:21`, `:28`; `core/tool_status.json:35-36`.
- **Mac VQA**. Ask serves the pinned 8-bit MLX B13 worker with `SATQUERY_VQA_MLX=1` (`1273a83`). Counting is refused. On the RSVQA-LR test sample (**400 questions**), B13 achieved **73.25%** macro (95% CI **69.99–76.50%**), base **54.50%**, and train-majority **57.25%**. The real-image advantage over grey was **11.75 points** (95% CI **7.50–16.00**). The run recorded **0 malformed answers** and **0.393 s** median tool latency. G6 run 2 kept at least **3.227 GiB** available against the **3 GiB** bar. [Fresh-test gate `78cf3fc`](gates/VQA-MAC-FRESH-TEST.md) · [Results `8231a76` and `cca98d0`](../results/vqa-mac-fresh-test/REPORT.md) · [Counting refusal](../core/gate.py#L78-L79)
- **Mac/CUDA parity**. The 8-bit attempt (`705e505` / `3522388`) and the bf16 attempt (`3b5c02e` / `f598393`) remain failed on base-model malformed answers. Investigated base outputs shifted on near-ties, and bf16 also missed the app headroom gate. The fresh test measures performance and readiness of the 8-bit deployment on a held-out sample. [Attempt 1](../results/vqa-mac-parity/REPORT.md) · [Attempt 2](../results/vqa-mac-parity/attempt2/REPORT.md) · [Near-tie diagnosis](gates/VQA-MAC-PARITY-ATTEMPT2.md)
- **CUDA providers**. The PyTorch Qwen and Grounding DINO providers remain unavailable on the demo Mac. The frozen Qwen RSVQA-LR run recorded open accuracy **0.165** and binary accuracy **0.667**. [Provider behavior](plan/E2E_ACCEPTANCE.md) · [Frozen Qwen run](../results/RESCORE_NOTE.md)
- **Optical-SAR**: deterministic index and backscatter summaries, not a learned fusion model; no accuracy evaluation. Source: `core/tool_status.json:18`.
- **Gate thresholds** (India rectangle, cloud 50%, nodata 50%, OOD 2σ): provisional guards, not calibrated cutoffs. Source: `core/gate.py:12-16`.

## Disabled or parked

- **Change QA**: disabled. CDVQA presence labels are predictable from question text alone (82.8%, 82.3%); pixel differencing (56.0%, 55.0%) did not beat train majority (56.2%, 55.5%). [Change QA decision](plan/DECISIONS.md) · [T18 baseline report](../results/t18/REPORT.md)
- **Flood U-Net**: parked. S1-only 0.897 vs gate > 0.9; S1+S2 worst chip 0.744 vs required > 0.95. Source: `docs/plan/DECISIONS.md:54`.

## Resolution evidence

- **Resolution and counting**. LoveDA v3 used **200 tiles** and both real and grey controls for base and B13. At **10 m**, pooled presence accuracy was **0.5029** for base and **0.5043** for B13. Both were degenerate and neither had a pooled real-minus-grey interval that excluded zero. V2 had an always-yes building result at 10 m under its own question protocol. The DIOR counting resolution gate remains untested because GSD annotations were unavailable. [Ladder v3 report](../results/ladder-v3/REPORT.md) · [Ladder v2 analysis](../results/ladder-v2/ANALYSIS.md) · [Counting limitations](https://github.com/sohamsssssssssssssssss/sih2026/blob/1837259d18fe9e96aca2339160789dd3c9c73a50/results/count-dior/ANALYSIS.md#L85-L90)

## Not evaluated

- No evaluation on ISRO/SAC products (Cartosat, RISAT). Source: `docs/research/ps-requirement-gap-matrix.md`.
