# Ladder v3 preregistration

Freeze this protocol and the question manifest before running inference. LoveDA is a spatial-resolution proxy from China. It does not validate Indian geography, ISRO sensor performance or the fidelity of synthetic sensor degradation. The masks provide the original tile labels at every rung.

## Cohort and questions

Use the 200 v2 tiles in their stored order, identified by the native rows in [v2](../../outputs/results_ladder_qwen_full_v2.json). The original Kaggle notebook `sohamssssss/sih26167-ladder-qwen-full` cloned `450cf04` and called the ladder generator with its default seed **26167**. Use that seed and the same RNG sequence across tiles and rungs. Use `eval/ladder.py::degrade` for native 0.3 m and simulated 1, 2, 5 and 10 m. Degradation uses PSF blur, area-average decimation and Poisson shot noise plus Gaussian read noise. The model processor subsequently prepares the resulting chip for inference.

[Question manifest](../../data/manifests/ladder-v3/questions.jsonl) SHA-256: **593497be0d09e701eb18e8d5f167d5613c71ab03c555dc97561f9fcbe06afd38**. It records original RGB and mask file hashes, source tile identities and gold labels. Pair the original dataset's flattened paths with the [official Train/Rural archive](https://zenodo.org/records/5706578) by tile filename. The builder verifies both original v2 gold answers for every tile.

- **Per-class presence.** Ask whether the tile contains each LoveDA class (background, building, road, water, barren, forest, agriculture). Any pixel with the mask class counts as present. Code 0 is ignored. For each class, keep equal counts from each side by taking the smaller side's count from both sides, ordered by SHA-256 of `26167:<class>:<yes/no>:<tile_id>`. If only one side exists, keep all its rows and disclose the missing side. Freeze selections across rungs and conditions.
- **Dominant class.** Ask which class occupies the largest area and show the seven labels as the allowed answers. Use the v2 mask-mode label, with v2's first-observed tie rule.
- **Rural or urban.** Ask the source domain for every tile, with both labels in the prompt. The fixed v2 cohort is entirely Train/Rural. Report this as a prediction-prior check with no urban recall or domain balanced accuracy.

The prompt requests exactly one listed label. Strip whitespace, case-fold and strip trailing periods, exclamation marks and question marks. Accept only whole labels. The full synonym map is `buildings→building`, `roads→road`, `bare land→barren`, `woodland→forest`, `cropland→agriculture`, `farmland→agriculture`, `1→yes`, `0→no`. An alias must resolve to a label allowed for the question. Any other output is counted as malformed and wrong.

## Conditions and scoring

Use base real, base constant, B13 real and B13 constant, all on the same questions and rungs. Pin Qwen2.5-VL-3B-Instruct at `66285546d2b821cf421d4f5eb2576359d3770cd3` with verified local weights and processor. Pin B13 to the final adapter from the `b13-final-750.zip` archive recorded in `20596be`. The Kaggle input exposes the extracted files. Verify both adapter files by SHA-256. Use FP16, the same 392 × 392 processor inputs, deterministic greedy decoding and an eight-token output cap for each condition. Each constant input is RGB (127, 127, 127) at its rung's chip dimensions before the identical processor path. Reset the degradation RNG before repeating the cohort with B13.

- Report accuracy, balanced accuracy and yes-rate per presence class and over pooled presence, plus accuracy and balanced accuracy for dominant class and domain. Yes-rate uses all answers as the denominator and applies to presence. Balanced accuracy averages recall of both labels for binary tasks and of observed gold classes for dominant class. A missing required class makes balanced accuracy undefined.
- Use a percentile paired bootstrap over the 200 source tiles, **2,000 draws with seed 17**, preserving the same sampled tiles for every condition, question type and rung. Include all of a tile's questions in each draw. A replicate with an absent required class contributes no balanced-accuracy estimate. A replicate without questions for a task contributes no accuracy estimate for that task.
- A **degenerate rung** for a condition is one whose pooled presence yes-rate is **strictly above 0.85** or whose pooled presence balanced-accuracy CI **includes 0.5**. Apply the same flag separately to each presence class. This chance rule applies to binary presence.
- Label a matched model/rung/type contrast **image-dependent** when the 95% CI of real minus constant **accuracy** excludes zero. Report the sign so a negative effect remains visible. Intervals are descriptive and have no multiple-comparison adjustment. Record every cell, including degenerate ones.

## Execution and outputs

The frozen set has 1,094 questions per rung and 21,880 answers across the five rungs and four conditions. Presence has 694 questions and each other type has 200.

Attach `growinfame/loveda`, `sohamssssss/satquery-qwen-pinned` and `sohamssssss/b13-final-750-zip`. Use [the Kaggle cell](../../kaggle/ladder_v3_cell.sh) with the full registration commit in `LADDER_V3_COMMIT`. The runner checks the exact image and mask bytes, the model inputs and the frozen manifest before inference. Output beneath `/kaggle/working/ladder-v3/results/ladder-v3/` includes `results.json` with raw answers, metrics, registration commit, gate SHA-256 (config hash) and manifest SHA-256, alongside `curve.png` and `REPORT.md`.
