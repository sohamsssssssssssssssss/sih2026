# SatQuery AI status (20 Nov 2026)

Every line names its source. A number with no recorded source is not listed.

## Validated (engineering checks, not model accuracy)

- **Live scene fetch**: Sentinel-2 L2A from Earth Search, Sentinel-1 RTC from Microsoft Planetary Computer, 120 × 120 window per band. Source: `backend/fresh_scenes.py:25-26`, `:32`.
- **Offline backups**: five cached chips, each with a source item ID and SHA-256. Offline, an unknown place name is refused ("Place lookup is unavailable"). Source: `data/fresh-backups/manifest.json`; `backend/fresh_scenes.py:161`.
- **Judge-style HTTP acceptance**: 5/5 checks passed on Mac CPU on 18 Nov (land cover, flood, change refusal, outside-India refusal, cloud refusal), each with a trace. Source: `docs/plan/E2E_ACCEPTANCE.md:3`, `:9-13`.
- **Python test suite**: last recorded full run 1,366 passed, 47 skipped (reasons listed), 0 failed. For anything newer, see the latest test run. Source: `docs/plan/INTEGRATION_LOG.md:592`.

## Experimental

- **Land cover**: Europe-trained model; India adaptation evaluated, failed promotion gate (G2, G3) — not shipped.
  - Europe checkpoints in service; India mAP s1 0.516, s2 0.576 vs Europe s1 0.688, s2 0.915. Source: `docs/plan/RESULTS.md:33-34`, `:59-60`.
  - India adaptation recipe (T13b, region-held-out, seed means): s1 0.516 → 0.711, s2 0.576 → 0.678; P1/P2/P3 PASS. Source: `results/t13b/REPORT.md:7`, `:13`, `:16`, `:20`, `:23`.
  - Promotion gate (frozen before the run): G1 PASS (s1 0.717, s2 0.688). G2 FAIL: Platt calibration raised fold-B macro ECE (s1 0.124 → 0.179, s2 0.178 → 0.187). G3 FAIL: on the Ludhiana S2 scene the top class was water 0.858, not built. Joint decision FAIL; Europe checkpoints retained. Source: `feature/ben-landcover` at `5e3fcee`, `results/landcover-india-promotion/REPORT.md:9-16`.
  - Input preprocessing for the India model is under diagnosis; no result yet.
  - Scores are uncalibrated; `confidence` is `null`. Source: `docs/plan/E2E_ACCEPTANCE.md:17`.
- **Flood water (Sentinel-1 per-scene Otsu)**: India-test IoU 0.631 vs gate 0.65: FAIL; not tuned afterwards. India-test is 14 chips from one event. Source: `outputs/T07c-flood.md:3`, `:21`, `:28`; `core/tool_status.json:35-36`.
- **Qwen2.5-VL VQA and Grounding DINO**: need CUDA; unavailable on the demo Mac. Frozen Qwen RSVQA-LR open accuracy 0.165, binary 0.667. Source: `docs/plan/E2E_ACCEPTANCE.md:24`; `README.md` Baseline section; `results/RESCORE_NOTE.md:30-32`.
- **Optical-SAR**: deterministic index and backscatter summaries, not a learned fusion model; no accuracy evaluation. Source: `core/tool_status.json:18`.
- **Gate thresholds** (India rectangle, cloud 50%, nodata 50%, OOD 2σ): provisional guards, not calibrated cutoffs. Source: `core/gate.py:12-16`.

## Disabled or parked

- **Change QA**: disabled. CDVQA presence labels are predictable from question text alone (82.8%, 82.3%); pixel differencing (56.0%, 55.0%) did not beat train majority (56.2%, 55.5%). Source: `docs/plan/DECISIONS.md:57-66`; full table on branch `task/T18-change-baseline` at `1d0cbaf`, `results/t18/REPORT.md`.
- **Flood U-Net**: parked. S1-only 0.897 vs gate > 0.9; S1+S2 worst chip 0.744 vs required > 0.95. Source: `docs/plan/DECISIONS.md:54`.

## Not evaluated

- No evaluation on ISRO/SAC products (Cartosat, RISAT). Source: `docs/research/ps-requirement-gap-matrix.md`.
