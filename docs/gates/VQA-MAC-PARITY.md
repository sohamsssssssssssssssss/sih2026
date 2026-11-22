# Mac VQA parity gate

Freeze this gate before inference. Compare Qwen2.5-VL-3B on MLX with and without the live B13 LoRA on the same 389 frozen RSVQA-LR validation rows. The manifest SHA-256 is `b55842a801b76e680e5aee3cded8bc47dad72498a10629b5648db2c07831973c`. Use the [CUDA per-question reference](https://github.com/sohamsssssssssssssssss/sih2026/blob/20596be/results/rgb-pilot-b13/rgb-b13-evaluation-01/result.json) from `20596be`, SHA-256 `93608083c5df7b1045e02d3787e040b46ccb334b83bc845f22e0044a7022fefd`, and verify every image used against that result's image hash.

Use the 8-bit local MLX conversion of base revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, SHA-256 verified via the conversion MANIFEST. B13 weights are pinned at `7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a`. Use the P28 settings for both conditions: RGB resized to 392×392, identical prompt, greedy decoding, 32 new tokens, repetition penalty 1.05 and context 4096. Use the CUDA reference's exact stripped case-insensitive match and its allowed answer formats. Report each type's accuracy and unweighted macro over presence, comparison, count and rural/urban, along with exact normalized answer agreement against CUDA. For the paired accuracy difference, resample image clusters with replacement, retaining all their rows and sharing each draw between conditions, using 2,000 percentile bootstrap replicates with seed 17. Exclude draws missing a question type from macro differences.

All checks must pass:

- MLX B13 macro is within 0.030 of the specified 0.696 CUDA reference.
- The paired 95% CI for MLX B13 minus MLX base macro excludes zero.
- Each MLX condition has median question latency at most 10 seconds, zero crashes, and zero malformed answers beyond its corresponding CUDA condition. Report load, first answer, median, p95 and peak memory for both.
- After the full B13 pass, run 30 mixed questions in the same process. Shuffle each type's ordered rows with seed 17, then take seven per type plus one extra each for presence and comparison. Shuffle those 30 together using the same generator. Compare peak observed memory during the soak against its starting level. Monitor MLX active plus cached bytes and process RSS separately. Neither may grow by more than 1 GiB. Require zero soak crashes.

The results record the gate commit, validation manifest hash, CUDA result hash and conversion MANIFEST hash. Only a passing gate permits app integration.
