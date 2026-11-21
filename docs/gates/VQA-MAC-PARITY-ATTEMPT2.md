# Mac VQA parity gate, attempt 2 (MLX bf16)

Written on 21 Nov 2026 before any attempt-2 inference. [Attempt 1](VQA-MAC-PARITY.md) (gate `705e505`, results `3522388`, [report](../../results/vqa-mac-parity/REPORT.md)) **FAILED** on the malformed check and stays recorded as FAIL. Neither the attempt-1 gate nor its report is edited.

## Diagnosis of attempt 1

Four of the five malformed MLX 8-bit base answers were the same malformed answers CUDA gave (`validation-48846`, `-49157`, `-53879`, `-56187`). One, `validation-56181`, was malformed on MLX only. With the same prompt, chat template, repetition context, 32 new tokens, greedy decoding and 392×392 input, only the weight precision changes the first token:

| Weights | P("1") | P("No") | Answer |
|---|---:|---:|---|
| MLX 8-bit (attempt 1) | 0.2339 | 0.2302 | `1` (malformed) |
| MLX bf16 (pinned HF weights) | 0.2227 | 0.2520 | `No` (matches CUDA) |

The MLX prompt is byte-identical to the chat template recorded in the CUDA reference, and the first-token probabilities are identical with the 1.05 repetition penalty on and off. The cause is 8-bit quantisation drift on a near-tie.

## Method change

Both conditions, base and B13, use **MLX bf16 weights with no quantisation**. They are converted from the pinned local HF weights of revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, the weights whose SHA-256 the CUDA run verified, into `~/models/qwen2.5-vl-3b-mlx-bf16/` with a MANIFEST of file SHA-256s, the conversion command and versions. The evaluation code `eval/vqa_mac_parity.py` is unchanged. Everything else in the protocol is unchanged.

## Protocol and pass criteria (copied from attempt 1, word for word)

Freeze this gate before inference. Compare Qwen2.5-VL-3B on MLX with and without the live B13 LoRA on the same 389 frozen RSVQA-LR validation rows. The manifest SHA-256 is `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`. Use the [CUDA per-question reference](https://github.com/sohamsssssssssssssssss/sih2026/blob/20596be/results/rgb-pilot-b13/rgb-b13-evaluation-01/result.json) from `20596be`, SHA-256 `93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd`, and verify every image used against that result's image hash.

Use the ~~8-bit local MLX conversion~~ **bf16 local MLX conversion (attempt-2 method change)** of base revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, SHA-256 verified via the conversion MANIFEST. B13 weights are pinned at `7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a`. Use the P28 settings for both conditions: RGB resized to 392×392, identical prompt, greedy decoding, 32 new tokens, repetition penalty 1.05 and context 4096. Use the CUDA reference's exact stripped case-insensitive match and its allowed answer formats. Report each type's accuracy and unweighted macro over presence, comparison, count and rural/urban, along with exact normalized answer agreement against CUDA. For the paired accuracy difference, resample image clusters with replacement, retaining all their rows and sharing each draw between conditions, using 2,000 percentile bootstrap replicates with seed 17. Exclude draws missing a question type from macro differences.

All checks must pass:

- MLX B13 macro is within 0.030 of the specified 0.696 CUDA reference.
- The paired 95% CI for MLX B13 minus MLX base macro excludes zero.
- Each MLX condition has median question latency at most 10 seconds, zero crashes, and zero malformed answers beyond its corresponding CUDA condition. Report load, first answer, median, p95 and peak memory for both.
- After the full B13 pass, run 30 mixed questions in the same process. Shuffle each type's ordered rows with seed 17, then take seven per type plus one extra each for presence and comparison. Shuffle those 30 together using the same generator. Compare peak observed memory during the soak against its starting level. Monitor MLX active plus cached bytes and process RSS separately. Neither may grow by more than 1 GiB. Require zero soak crashes.

The results record the gate commit, validation manifest hash, CUDA result hash and conversion MANIFEST hash. Only a passing gate permits app integration.

## Added check (set before running): memory headroom with the demo app running

- **Pass:** at least 3 GiB of system memory stays available on the 16 GB Mac throughout the measurement.
- **Setup:** start the demo app with `bash scripts/run-demo.sh` (backend on port 8000, frontend on port 3000) and confirm both answer HTTP before measuring. In a separate process, load the bf16 base with B13 and answer the same 30 soak questions, in the same order.
- **Measure:** `eval/vqa_mac_headroom.py` samples `vm_stat` every 0.25 s from before the model load until after the last answer. Available memory = (free + inactive + speculative pages) × page size. The check uses the **minimum** sample.
- **Also report:** that process's peak RSS and MLX peak, and the same minimum measured with the app stopped.

A bf16 conversion or run failure is recorded as not run, never retried with changed settings.
