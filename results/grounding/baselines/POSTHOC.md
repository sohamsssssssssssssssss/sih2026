# Grounding v1 post-hoc diagnostics

**Post-hoc, informational, not a gate result.** Official scores and the adapter bar are unchanged. [Protocol](../../../docs/gates/GROUNDING-V1-POSTHOC-DIAGNOSTICS.md), committed as `62a91de6581a809e86a840185ef229f3cc613aa4` before scoring.

## Single fence pair removal

Each dataset contributes 1,000 frozen eval rows. The pooled score weights both datasets equally. The intervals use 10,000 seed-17 stratified bootstrap draws and linear percentiles.

| Qwen condition | Dataset | Acc@0.5 [95% CI] | Mean IoU [95% CI] | Remaining parse failures |
|---|---|---:|---:|---:|
| Matched | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.000000 [0.000000, 0.000000] | 1,000 |
| Matched | VRSBench | 0.00% [0.00%, 0.00%] | 0.000312 [0.000000, 0.000853] | 995 |
| Matched | Pooled | 0.00% [0.00%, 0.00%] | 0.000156 [0.000000, 0.000426] | 1,995 |
| Shuffled | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.000000 [0.000000, 0.000000] | 999 |
| Shuffled | VRSBench | 0.00% [0.00%, 0.00%] | 0.000065 [0.000000, 0.000195] | 999 |
| Shuffled | Pooled | 0.00% [0.00%, 0.00%] | 0.000033 [0.000000, 0.000098] | 1,998 |
| Grey | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.000000 [0.000000, 0.000000] | 1,000 |
| Grey | VRSBench | 0.00% [0.00%, 0.00%] | 0.000028 [0.000000, 0.000085] | 998 |
| Grey | Pooled | 0.00% [0.00%, 0.00%] | 0.000014 [0.000000, 0.000043] | 1,998 |

Fence removal recovered zero valid boxes in all three conditions. Of the matched replies, 1,975 had an eligible outer fence pair. Every one of those replies contained a JSON list with a length other than four.

## Captioning sanity check

Direct COCO `CiderScorer` with `pycocoevalcap==1.2`, standard PTB tokenization, sigma 6, and the same 1,000 frozen VRSBench references reproduced the committed scorer exactly.

| Committed CIDEr-D | Direct CIDEr-D | Difference |
|---:|---:|---:|
| 0.00134279990555146 | 0.00134279990555146 | 0 |
