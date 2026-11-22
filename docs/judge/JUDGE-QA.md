# Judge Q&A: 34 hard questions

Prefixes: `integ` = ~/sih2026-integ, `ben` = ~/sih2026-ben, `count` = ~/sih2026-count, `gate` = ~/sih2026-gate. The first 5 are the most dangerous.

## The 5 most dangerous

**1. Does your VQA actually look at the image?**
On the preregistered RSVQA-LR test sample of 400 questions, 8-bit MLX B13 scored 73.25% macro on real images and 61.50% with a constant grey image. The paired gain was 11.75 percentage points (95% CI 7.50–16.00). Base scored 54.50% and train-majority 57.25%. The grey-image condition still gets many answers right, so question cues also contribute. Ask serves B13 on the Mac with `SATQUERY_VQA_MLX=1` and refuses counting.
[Fresh-test gate](../gates/VQA-MAC-FRESH-TEST.md) · [Fresh-test results](../../results/vqa-mac-fresh-test/REPORT.md) · [Mac integration](https://github.com/sohamsssssssssssssssss/sih2026/commit/1273a83ef58c8a4b6857ba1e24c560d3ede4f638)

**2. Why didn't the Mac parity gate pass?**
The CUDA comparison required the Mac base model to produce no extra malformed outputs. In the 8-bit attempt, quantisation flipped a near-tied base answer. The bf16 attempt still produced extra malformed base answers and left only 1.81 GiB available with the app open against a 3 GiB requirement. Both attempts remain failed. The fresh test asks whether the shipped 8-bit B13 model beats the base, train-majority and grey-image controls on held-out RSVQA-LR, and whether it fits the demo. G6 run 2 reached 3.227 GiB minimum available memory, and the test recorded 0 malformed answers and 0.393 s median tool latency.
[8-bit gate](../gates/VQA-MAC-PARITY.md) · [8-bit parity report](../../results/vqa-mac-parity/REPORT.md) · [bf16 gate and near-tie diagnosis](../gates/VQA-MAC-PARITY-ATTEMPT2.md) · [bf16 parity report](../../results/vqa-mac-parity/attempt2/REPORT.md) · [Fresh-test results](../../results/vqa-mac-fresh-test/REPORT.md)

**3. What happens at 10 m?**
On LoveDA ladder v2, the model said yes to every building-presence question at 10 m and scored 0.700, exactly the always-yes baseline. At 5 m its yes-rate was 0.970. The 0.3 m rung scored 0.790 for building presence. Dominant-class accuracy stayed below its majority baseline at every rung. The preregistered v3 ladder has yet to run.
[Ladder v2 analysis](../../results/ladder-v2/ANALYSIS.md) · [Ladder v3 gate](../gates/LADDER-V3.md)

**4. Why isn’t the India-adapted model shipped if its recipe improved held-out mAP?**
G1 passed, but neither Platt calibration nor temperature scaling met the region-transfer gate. One fold’s scores ran high and the other’s ran low. Corrected scene checks passed G3 in a record-only rerun. The promotion decision is closed and Ask uses the Europe-trained model.
Source: [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)

**5. Change detection is in the problem statement. Why is it switched off?**
We tested it and it did not work. On CDVQA, the question text alone predicts the "did it change?" answer 82.8% of the time, and our pixel-differencing baseline (56.0%) did not beat the train-majority answer (56.2%). Our adapter also kept much of its accuracy when both images were identical, so it was not using the images. We refuse change questions instead of guessing.
Source: integ:docs/plan/DECISIONS.md:57-66; integ:docs/plan/RESULTS.md:126-130

## Data and resolution

**6. Where does the imagery come from, and is it live?**
Sentinel-2 L2A comes from Earth Search and Sentinel-1 RTC from Microsoft Planetary Computer. Each band read uses a 120 × 120 window. Built areas ranked first in live Ludhiana (0.955) and Delhi (0.949). Cropland ranked first on the Punjab fixture (0.378). Ask identifies cached imagery separately.
Source: integ:backend/fresh_scenes.py; integ:docs/plan/INTEGRATION_LOG.md, 2026-11-20, BOA offset entry.

**7. What happens if the provider or internet is down?**
Ask labels fallback imagery as cached and shows where it came from. Backups have item IDs and SHA-256 checksums. Offline, an unknown place name receives “Place lookup is unavailable”. A deadline fallback requires a matching backup.
Source: integ:data/fresh-backups/manifest.json; integ:backend/fresh_scenes.py; integ:`5a778ba`.

**8. ISRO sensors are sub-metre; you work on 10 m Sentinel. How do you handle the resolution gap?**
Ladder v2 degraded 200 LoveDA tiles from 0.3 m to 10 m and exposed a strong yes bias on the coarse building-presence rungs. The ladder is a proxy for resolution loss on those tiles. The DIOR counting evaluation lacked ground sample distance (GSD) annotations, so the counting resolution gate was untested. The Mac demo refuses counting.
[Ladder v2 analysis](../../results/ladder-v2/ANALYSIS.md) · [Counting limitations](https://github.com/sohamsssssssssssssssss/sih2026/blob/1837259d18fe9e96aca2339160789dd3c9c73a50/results/count-dior/ANALYSIS.md#L85-L90) · [Counting refusal](../../core/gate.py#L78-L79)

**9. Have you tested on Cartosat, RISAT or any ISRO/SAC product?**
No. No ISRO/SAC dataset evaluation is recorded. We would need authorised representative data and locked splits first.
Source: integ:docs/research/ps-requirement-gap-matrix.md:22

**10. What are your labels? Is WorldCover ground truth?**
No. Land-cover labels are ESA WorldCover map fractions, a map product with its own errors. Our mAP numbers measure agreement with that map, not with field truth.
Source: integ:docs/plan/RESULTS.md:104; ben:docs/gates/LANDCOVER-INDIA-PROMOTION.md:3

## India vs Europe

**11. Why did SatQuery misclassify urban Ludhiana, and what proves the fix?**
Our fetch applied the Sentinel-2 BOA offset twice, darkening reflectance by 0.1. Removing the duplicate correction put built areas first at 0.955 on a live fetch of the same item. Delhi and Punjab provide two additional scene checks.
Source: integ:`84d7f57`, `404039f`; integ:docs/plan/INTEGRATION_LOG.md, 2026-11-20, BOA offset entry.

**12. How big is the India vs Europe gap, exactly?**
For S2, India mAP is 0.576 and Europe is 0.915 on the same five classes, a geography drop of −0.339 (CI95 −0.356 to −0.319). For S1 it is 0.516 vs 0.688.
Source: integ:docs/plan/RESULTS.md:33-34, :59-60, :75

**13. Did adaptation help every Indian region?**
No. Full fine-tuning got worse in some regions: s2 Kerala (−0.012), and s1s2 Rajasthan (−0.035) and Punjab (−0.045). These are reported but are not part of the gate.
Source: integ:results/t13b/REPORT.md:97-104

**14. Is the improvement real, or just batch-norm statistics?**
Partly batch-norm. The BN-only arm alone reaches s1 0.677 and s2 0.597, and the analysis attributes part of the gain to BN re-estimation by rule (e). Full fine-tuning still beats the label-shuffle control in all three modes.
Source: integ:results/t13b/REPORT.md:3, :19, :26, :60

## Leakage and evaluation

**15. How do you know there is no train/test leakage in the India adaptation?**
Test regions are held out by region: fold A adapts on Punjab, Kerala and Assam and tests on Delhi, Sundarbans and Rajasthan, and fold B swaps them. Index sets are pairwise disjoint, and a SHA-256 hash of every patch's pixels found no duplicates between test and train/val. 18/18 audits passed.
Source: integ:results/t13b/REPORT.md:153, :177-179

**16. Six regions and three seeds: are your confidence intervals honest?**
Only partly. The intervals are patch-level bootstrap and leave out seed and region variance; folds are fixed, so split variability is not captured. We state this in the report.
Source: integ:results/t13b/REPORT.md:169, :283

**17. Were thresholds or gates changed after you saw results?**
Thresholds were never changed. After the first India promotion attempt failed calibration (Platt), we preregistered a second attempt (temperature scaling) before computing new results. The first failure stays on record. Attempt 2 also failed, and we stopped there; the model was not shipped. Flood Otsu was not tuned and its threshold was not relaxed after it failed. The Otsu clamp was set before the run, not learned from test.
Source: ben:docs/gates/LANDCOVER-INDIA-PROMOTION.md:3, :23; ben:docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md (gate `db6e127`, before results `ce8449e`); integ:docs/plan/DECISIONS.md:55; integ:outputs/T07c-flood.md:30

**18. Can you count objects?**
Not reliably. On a frozen 1,000-pair DIOR sample, Grounding DINO is right on 32.4% of counts, while always answering 0 scores 50.0%. Its MAE (1.53) is no better than always-0 (1.48). A presence gate on top also failed to beat always-0.
Source: count:results/count-dior/ANALYSIS.md:7, :85; gate:results/count-dior/gate/GATE.md:3

## Failure modes and trust

**19. Your flood tool fails its own gate. Why show it at all?**
It is labelled experimental: India-test IoU 0.631 against a 0.65 gate, and we did not tune it after seeing the result. The answer text says "water alone does not establish flooding". The test set is 14 chips from a single event, so we cannot estimate error on a new flood. The learned U-Net is parked after missing its gates.
Source: integ:outputs/T07c-flood.md:3, :21, :28; integ:docs/plan/DECISIONS.md:54-55; integ:docs/plan/E2E_ACCEPTANCE.md:95

**20. What confidence does the product report, and how should a judge read the scores?**
The API returns `confidence: null`. The land-cover model shows relative evidence of class presence and labels each output “uncalibrated score”. The value has no probability or area-fraction interpretation.
Source: integ:core/tools/landcover.py; integ:docs/plan/E2E_ACCEPTANCE.md; [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)

**21. What does the system refuse, and why?**
It refuses scenes outside the India box [68, 6, 98, 38], optical scenes over 50% cloud (and suggests S1), scenes over 50% nodata, and all change questions. These thresholds are marked PROVISIONAL: they are guards, not calibrated cutoffs.
Source: integ:core/gate.py:12-16; integ:docs/plan/E2E_ACCEPTANCE.md:11-13

**22. Your cloud check: does it measure cloud in the chip?**
No. It uses the scene-level `eo:cloud_cover` from the STAC catalogue for the whole Sentinel-2 tile, not the 120×120 chip. When a patch has no cloud metadata, the answer says "Cloud cover not checked".
Source: integ:docs/plan/E2E_ACCEPTANCE.md:17, :20; integ:README.md:95

**23. Why should a scientist trust any single answer?**
Every answer records which tool, version and checkpoint ran, and which gates fired, in a SHA-256 hash-chained trace that can be verified. Live and cached results are labelled separately. Trust comes from that audit trail, not from the model's accuracy, which is still experimental for every tool.
Source: integ:docs/plan/E2E_ACCEPTANCE.md:24, :163-172; integ:core/tool_status.json:5-36

**24. What happens without a GPU?**
Without a supported accelerator, the PyTorch Qwen VQA and Grounding DINO providers report `CUDA_UNAVAILABLE`. On Apple Silicon, the separate MLX worker can serve the pinned B13 adapter in Ask. Land cover, flood and optical-SAR run on CPU.
[PyTorch provider behavior](../plan/E2E_ACCEPTANCE.md) · [Mac runbook](../RUNBOOK.md)

**25. How well does the base VLM do on remote-sensing questions?**
The frozen Qwen2.5-VL-3B baseline on the full RSVQA-LR test split (10,004 questions) recorded open accuracy 0.165 and binary accuracy 0.667. The Mac adaptation has a separate held-out test sample and uses macro accuracy across its question types.
[Frozen Qwen test](../../results/RESCORE_NOTE.md) · [Mac fresh-test results](../../results/vqa-mac-fresh-test/REPORT.md)

**26. How many tests pass?**
The integration log records the latest full pytest, Vitest, TypeScript and production Playwright totals, including the separate offline browser case.
[Integration log](../plan/INTEGRATION_LOG.md)

## Novelty

**27. What is actually new here?**
Not a new model. What we contribute is an agent that refuses rather than guesses: preregistered gates, controls such as shuffled labels, identical image pairs and question-only baselines, and a verifiable trace for every answer. The clearest result is negative: CDVQA change answers can be predicted from the question text, which we found and documented.
Source: integ:docs/plan/DECISIONS.md:57-66; integ:docs/hypotheses/T13b.md; integ:results/t13b/REPORT.md:5

**28. Is optical-SAR "fusion" a learned model?**
No. It reports deterministic Sentinel-2 indices and Sentinel-1 backscatter summaries on co-registered inputs. It has no accuracy evaluation yet.
Source: integ:README.md:19; integ:docs/research/ps-requirement-gap-matrix.md:13; integ:core/tool_status.json:18

## The failed promotion gate

**29. Did one Ludhiana scene decide the India promotion?**
G3 checked class rankings on Ludhiana and Punjab. The original Ludhiana input carried a duplicate BOA offset. Once corrected, built areas and cropland each ranked first on their respective scene. The independent calibration criterion still failed, so the candidate stays out of deployment.
Source: [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)

**30. What did the second calibration attempt show?**
Temperature scaling improved S1 from fold B to A and S2 from A to B. The reverse directions got worse in each mode, so both modes failed the requirement to improve in both directions. Per-class Platt calibration had already worsened held-out macro calibration error for both modes.
Source: [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)

**31. What changed in the optical data after the bug fix?**
Earth Search S2 acquisition now checks whether the BOA offset has already been applied. We regenerated the Ludhiana, Delhi and Mumbai backups from the same item IDs and grids and corrected the separate cloudy-Mumbai fixture. The older TIFFs remain available for comparison.
Source: integ:`84d7f57`, `404039f`, `980cab7`; integ:docs/plan/INTEGRATION_LOG.md, 2026-11-20.

## Follow-up questions

**32. Why are your scores uncalibrated?**
The India region folds have opposite score biases and different class frequencies. Both tested calibration methods failed the transfer requirement. The Europe-trained checkpoint therefore serves uncalibrated presence scores with a null confidence field.
Source: [India promotion decision on `feature/ben-landcover` at `90ccde3`](https://github.com/sohamsssssssssssssssss/sih2026/blob/90ccde3aa4c4d6eb0ee6e7d2d8591e4b792d905a/results/landcover-india-promotion/DECISION.md)

**33. How did you find the offset bug?**
We compared the exact Earth Search window with the same acquisition from Planetary Computer and checked the ESA L2A metadata. Earth Search had already removed 1,000 DN, and the old fetch removed it again. The corrected input restored built areas to first place on live Ludhiana.
Source: integ:`84d7f57`; integ:docs/plan/INTEGRATION_LOG.md, 2026-11-20, BOA offset entry.

**34. What happens on bad Wi-Fi?**
The overall live fetch budget is 20 seconds. When a matching cache exists, Ask displays “Live fetch timed out after Ns — showing cached scene from <date>”. Otherwise it returns a clear refusal. The S2 per-read timeout is 60 seconds. The controlled one-second Ludhiana trial returned the dated cached message and the usual land-cover answer.
Source: integ:`ecb651e`, `5a778ba`; integ:docs/plan/INTEGRATION_LOG.md, 2026-11-20, live-fetch deadline entry.
