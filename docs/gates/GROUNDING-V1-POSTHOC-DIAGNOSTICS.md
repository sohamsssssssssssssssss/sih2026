# Grounding v1 post-hoc diagnostics

These diagnostics are post-hoc, informational, and outside the registered gate. They cannot change any official score, threshold, adapter bar, or verdict.

## Qwen fence handling

Use the frozen base Qwen matched, shuffled, and grey predictions and the same frozen eval rows. For each reply, ignore outer whitespace when looking for a wrapping fence. Remove one pair only if the first line is exactly three backticks, optionally followed by lowercase `json`, and the last line is exactly three backticks. Pass the enclosed text to the registered box parser unchanged. If the reply has no such pair, pass the original reply unchanged. Do not remove nested fences, recover individual coordinates, or apply any other normalization.

For each condition, report parse failures, Acc@0.5 and mean IoU by dataset and as an equally weighted pool. Keep all frozen rows in the denominator and score parser failures as zero IoU. Use the registered seed-17 stratified draws, 10,000 replicates, and linear 95% percentile intervals for both metrics.

## Captioning sanity check

Independently recompute base Qwen CIDEr-D with the pinned COCO caption implementation and its standard PTB tokenizer on the same frozen VRSBench references and corresponding predictions. Compare at full precision with the committed score. Stop and report any difference beyond rounding without changing the scorer.
