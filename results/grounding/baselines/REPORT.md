# Grounding v1 baseline results

DINO selection used 500 held-out grounding expressions per dataset. The selected threshold was **box 0.25, text 0.25**, applied unchanged to eval and both DINO controls.

Each grounding eval set contains 1,000 expressions. Pooled metrics weight the datasets equally. Absolute Acc@0.5 and paired differences use the same 10,000 seed-17 stratified bootstrap draws and linear 95% percentiles.

| Method | Dataset | Acc@0.5 [95% CI] | Acc@0.25 | Mean IoU | Malformed | Missing | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| Base Qwen | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 0 | 1000 |
| Base Qwen | VRSBench | 0.00% [0.00%, 0.00%] | 0.00% | 0.0003 | 99.50% | 0 | 995 |
| Base Qwen | Pooled | 0.00% [0.00%, 0.00%] | 0.00% | 0.0002 | 99.75% | 0 | 1995 |
| Grounding DINO | DIOR-RSVG | 18.40% [16.00%, 20.90%] | 27.10% | 0.2002 | 4.90% | 15 | 34 |
| Grounding DINO | VRSBench | 22.00% [19.40%, 24.60%] | 28.50% | 0.2150 | 44.40% | 10 | 434 |
| Grounding DINO | Pooled | 20.20% [18.45%, 21.95%] | 27.80% | 0.2076 | 24.65% | 25 | 468 |
| Centre-box prior | DIOR-RSVG | 5.90% [4.50%, 7.40%] | 19.90% | 0.1218 | 0.00% | 0 | 0 |
| Centre-box prior | VRSBench | 0.70% [0.20%, 1.30%] | 15.50% | 0.1021 | 0.00% | 0 | 0 |
| Centre-box prior | Pooled | 3.30% [2.55%, 4.10%] | 17.70% | 0.1119 | 0.00% | 0 | 0 |
| Qwen shuffled | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 99.90% | 0 | 999 |
| Qwen shuffled | VRSBench | 0.00% [0.00%, 0.00%] | 0.00% | 0.0001 | 99.90% | 0 | 999 |
| Qwen shuffled | Pooled | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 99.90% | 0 | 1998 |
| Qwen grey | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 0 | 1000 |
| Qwen grey | VRSBench | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 0 | 1000 |
| Qwen grey | Pooled | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 0 | 2000 |
| DINO shuffled | DIOR-RSVG | 8.20% [6.50%, 10.00%] | 14.80% | 0.1068 | 7.30% | 39 | 34 |
| DINO shuffled | VRSBench | 13.00% [10.90%, 15.10%] | 17.60% | 0.1351 | 54.00% | 30 | 510 |
| DINO shuffled | Pooled | 10.60% [9.25%, 12.00%] | 16.20% | 0.1210 | 30.65% | 69 | 544 |
| DINO grey | DIOR-RSVG | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 88 | 912 |
| DINO grey | VRSBench | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 30 | 970 |
| DINO grey | Pooled | 0.00% [0.00%, 0.00%] | 0.00% | 0.0000 | 100.00% | 118 | 1882 |

Baseline controls are descriptive. Changes are matched-input Acc@0.5 minus control Acc@0.5.

| Comparison | Pooled difference (percentage points) | Paired 95% CI (percentage points) |
|---|---:|---:|
| Base Qwen − DINO | -20.20 | [-21.95, -18.45] |
| Base Qwen − shuffled | +0.00 | [+0.00, +0.00] |
| Base Qwen − grey | +0.00 | [+0.00, +0.00] |
| DINO − shuffled | +9.60 | [+7.80, +11.40] |
| DINO − grey | +20.20 | [+18.45, +21.95] |

The shuffled control had 2 same-wording donors for Qwen and 2 for DINO.

Base Qwen scores 0.00% pooled. Its shuffled and grey controls score 0.00% and 0.00%. The centre-box prior scores 3.30% pooled.
The centre-box prior beats base Qwen on both datasets. The registered box parser rejected 1,995 of 2,000 matched Qwen replies. Of those, 1,993 started with a code fence. None of the five valid boxes reached 0.5 IoU.

| VRSBench captioning (1,000 images) | CIDEr | BLEU-4 | METEOR |
|---|---:|---:|---:|
| Base Qwen | 0.001343 | 0.024821 | 0.170333 |

## Adapter decision criteria

Grounding requires pooled Acc@0.5 strictly above **0.00%** (base Qwen) and **20.20%** (DINO), so the point-score minimum is **405/2,000 (20.25%)**. Both paired 95% lower bounds must be above zero, and the adapter's matched accuracy must exceed each of its own controls.

Captioning requires CIDEr strictly above **0.00134279990555146** and a paired gain whose 95% lower bound is above zero. An adapter verdict is pending.

## Provenance

The Kaggle receipt, all nine checksums, all four reuse pins, and exact frozen ID coverage passed verification. The raw outputs and original checksum file are retained locally because the GitHub repository is public. Use the published manifest to verify the original checksum file before checking the raw files. [Run manifest](MANIFEST.json) · [Scores](scores.json)
