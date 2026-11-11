# B13 evaluation evidence

QLoRA adapter (B13) on Qwen/Qwen2.5-VL-3B-Instruct (revision 66285546d2b821cf421d4f5eb2576359d3770cd3),
6k RSVQA-LR questions, 750 steps (run A budget-stopped at step 464, run B resumed to 750, `training_complete`).

- Where: Kaggle, T4 GPU
- When: training 10–11 Nov 2026 (UTC report timestamps 2026-11-10T19:55Z run A, 2026-11-10T21:07Z run B); evaluation 11 Nov 2026
- run_identity: 0da54fddc9da7abe56c2c52746f354d38fd34402bcc233e76bdd37e50d137582
- Extracted: 2026-11-11
- Not re-run: contents copied verbatim from the Kaggle output zips.

## Source zips

| zip | sha256 |
|---|---|
| b13-eval-results.zip | b6eac417b9c01835b38b2f4515232129285331a1ea681c59718ff3cd683da488 |
| b13-final-750.zip | a719340e3e504d4c047383d6ffb3d2f516ad5053c5d40810eb26254ff424b513 |

## Files in this directory

| file | from | sha256 |
|---|---|---|
| rgb-b13-evaluation-01/result.json | b13-eval-results.zip: b13-eval/result.json | 93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd |
| rgb-b13-evaluation-01/exit.json | b13-eval-results.zip: b13-eval/exit.json | 16427eab67f16dab2fe3399647dbbfb62f83f69bd6d53a70b5c333bfc6bab47f |
| rgb-b13-evaluation-01/session.json | b13-eval-results.zip: b13-eval/session.json | 6c12fe88658db63c2a621f6aa32ad3399a34fff972d3c2083d28aa14975aad8c |
| rgb-b13-evaluation-01/worker.log | b13-eval-results.zip: b13-eval/worker.log | e474a3ebd6be6596fb0ce02e9ea277c7077c8ab0e454d7c20860c328f91d7e16 |
| training/run-a-training-report.json | b13-final-750.zip: b13-resume-112-a/training-report.json | 068106d233c9d5c2c80735f89c44eccf7adcb255652fddbd4edd5a1b4157bbb6 |
| training/run-b-training-report.json | b13-final-750.zip: b13-resume-112-b/training-report.json | 02b30ff8b747898456417d9c43fdf579b6862d4da2981cde795c4c3f55932f3e |

## Adapter (not in git)

- Mac: ~/Downloads/b13-final-750.zip (path inside: b13-resume-112-b/adapter/)
- Kaggle dataset: sohamssssss/b13-final-750-zip
- adapter_model.safetensors sha256: 7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a
  (computed from the zip; matches `adapter_identity.files_sha256` in result.json)

## Headline results (from rgb-b13-evaluation-01/result.json)

Validation subset: 389 questions (presence 100, comp 100, count 100, rural_urban 89), 89 images.
Exploratory paired pilot; no official benchmark claim. Macro = equal-weight mean over the four types.

| condition | macro | presence | comp | count | rural_urban | malformed |
|---|---|---|---|---|---|---|
| base_real | 0.4683 | 0.5500 | 0.6900 | 0.0600 | 0.5730 | 4/389 |
| base_constant | 0.3275 | 0.5600 | 0.7100 | 0.0400 | 0.0000 | 109/389 |
| adapter_real | 0.6957 | 0.8800 | 0.9100 | 0.2400 | 0.7528 | 0/389 |
| adapter_constant | 0.5814 | 0.7700 | 0.8100 | 0.2400 | 0.5056 | 0/389 |

Note: base_constant rural_urban is malformed on 89/89 answers, so its 0.0000 there is a format failure,
not a content signal. (*_constant = grey RGB(127,127,127) image of the same size.)

Paired differences, macro (paired percentile bootstrap over image clusters, seed 17, 2000 reps):

| comparison | macro difference | CI95 |
|---|---|---|
| adapter_real_minus_base_real | +0.2274 | [+0.1841, +0.2715] |
| base_real_minus_base_constant | +0.1408 | [+0.0961, +0.1860] |
| adapter_real_minus_adapter_constant | +0.1143 | [+0.0764, +0.1532] |
| adapter_constant_minus_base_constant | +0.2539 | [+0.2074, +0.3012] |
