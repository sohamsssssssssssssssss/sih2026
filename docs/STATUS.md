# SatQuery AI status (20 Nov 2026)

Every line names its source. A number with no recorded source is not listed.

## Validated (engineering checks, not model accuracy)

- **Live scene fetch**. Earth Search Sentinel-2 L2A and Planetary Computer Sentinel-1 RTC use 120 × 120 windows. S2 has a 60-second per-read timeout (`ecb651e`). The 20-second overall deadline includes place lookup, STAC search and band reads (`5a778ba`). On expiry Ask displays “Live fetch timed out after Ns — showing cached scene from <date>” when a matching backup exists, or returns a clear refusal. [Integration log](plan/INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture)
- **Optical inputs and backups**. Earth Search inputs could be 0.1 reflectance too dark when the BOA offset was applied twice. The fetch fix is in `84d7f57`. The Ludhiana, Delhi and Mumbai backups were regenerated in `404039f`, and the separate cloudy-Mumbai demo fixture was corrected in `980cab7`. The original TIFFs are retained. The manifest records five active, checksum-verified backup chips. Offline unknown place names are refused. [Backup manifest](../data/fresh-backups/manifest.json) · [Integration log](plan/INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups)
- **Judge-style HTTP acceptance**: 5/5 checks passed on Mac CPU on 18 Nov (land cover, flood, change refusal, outside-India refusal, cloud refusal), each with a trace. Source: `docs/plan/E2E_ACCEPTANCE.md:3`, `:9-13`.
- **Latest recorded tests (20 Nov 2026)**. pytest **1,401 passed, 47 skipped, 0 failed**. Vitest **50 passed**. TypeScript **0 diagnostics**. Playwright **9 passed** across the live and offline runs. [Integration log](plan/INTEGRATION_LOG.md#2026-11-20--live-fetch-deadline-and-mumbai-fixture)

## Experimental

- **Land cover**. Europe-trained BigEarthNet checkpoints serve uncalibrated scores with `confidence: null`. Built areas ranked first in live Ludhiana (**0.955**) and Delhi (**0.949**), and cropland ranked first on the Punjab fixture (**0.378**). These are uncalibrated presence scores. The transfer benchmark still records an India–Europe gap (S1 0.516 vs 0.688, S2 0.576 vs 0.915 mAP). [Recorded API results](plan/INTEGRATION_LOG.md#2026-11-20--sentinel-2-boa-offset-and-regenerated-backups) · [Transfer results](plan/RESULTS.md)
- **India land-cover promotion closed**. G1 passed on held-out regions (S1 0.717 vs 0.689 floor, S2 0.688 vs 0.659 floor). Both calibration attempts failed (Platt and temperature scaling), with opposite biases across the region folds. The original Ludhiana G3 failure came from the BOA offset bug. Corrected Ludhiana and Punjab both passed G3 in a record-only rerun. The India weights were not shipped. [India promotion decision](../results/landcover-india-promotion/DECISION.md)
- **Flood water (Sentinel-1 per-scene Otsu)**: India-test IoU 0.631 vs gate 0.65: FAIL; not tuned afterwards. India-test is 14 chips from one event. Source: `outputs/T07c-flood.md:3`, `:21`, `:28`; `core/tool_status.json:35-36`.
- **Qwen2.5-VL VQA and Grounding DINO**: need CUDA; unavailable on the demo Mac. Frozen Qwen RSVQA-LR open accuracy 0.165, binary 0.667. Source: `docs/plan/E2E_ACCEPTANCE.md:24`; `README.md` Baseline section; `results/RESCORE_NOTE.md:30-32`.
- **Optical-SAR**: deterministic index and backscatter summaries, not a learned fusion model; no accuracy evaluation. Source: `core/tool_status.json:18`.
- **Gate thresholds** (India rectangle, cloud 50%, nodata 50%, OOD 2σ): provisional guards, not calibrated cutoffs. Source: `core/gate.py:12-16`.

## Disabled or parked

- **Change QA**: disabled. CDVQA presence labels are predictable from question text alone (82.8%, 82.3%); pixel differencing (56.0%, 55.0%) did not beat train majority (56.2%, 55.5%). [Change QA decision](plan/DECISIONS.md) · [T18 baseline report](../results/t18/REPORT.md)
- **Flood U-Net**: parked. S1-only 0.897 vs gate > 0.9; S1+S2 worst chip 0.744 vs required > 0.95. Source: `docs/plan/DECISIONS.md:54`.

## Not evaluated

- No evaluation on ISRO/SAC products (Cartosat, RISAT). Source: `docs/research/ps-requirement-gap-matrix.md`.
