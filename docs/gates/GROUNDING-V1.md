# Grounding v1 preregistration

## Frozen data

Use [the seed-17 manifest](../../results/grounding/MANIFEST.json) and verify all input and split hashes before running. [Data provenance and licence terms](../../results/grounding/DATA-CHECK.md)

- DIOR-RSVG eval uses 1,000 expressions from its released test list. DINO selection uses 500 expressions from its validation list. Training uses 5,000 expressions from its training list.
- VRSBench eval uses 1,000 expressions from its released test set and the corresponding 1,000 reference captions. DINO selection holds out 500 grounding expressions from the released train set. Training uses 2,500 grounding expressions and 2,500 captions for those same train images.
- Sampling uses one grounding expression per source image parent in each manifest. VRSBench crops carry their DOTA or DIOR parent ID, with DIOR parents shared across datasets. Equal ZIP member sizes and CRC-32 values group possible image aliases under the lexically first parent ID before any split is drawn. The two eval sets use distinct parents. All publisher test parents are reserved from selection and training, and selection parents are reserved from training. VRSBench rows without a valid, nonempty task annotation are excluded before sampling. Evaluation rows are fixed thereafter.
- This registration applies to the newly frozen splits. The prior T4 zero-shot and Mac study were run under earlier protocols and are recorded in [LOCATE.md](../../results/grounding/LOCATE.md).

## Grounding outputs and metrics

The output is one axis-aligned box. Convert DIOR-RSVG pixel xyxy by the 800 × 800 source dimensions. Use the VRSBench release's horizontal xyxy answer, divided by 100. Evaluate predictions in normalized image coordinates with the same intersection-over-union calculation for every condition.

- Primary metric is Acc@0.5. Report Acc@0.25, mean IoU and malformed-box rate as well. An IoU exactly equal to a threshold is a hit. All frozen rows remain in the denominator.
- A missing, unparseable, nonfinite, inverted, zero-area, extra or out-of-range box is a miss with IoU zero and counts toward the malformed-box rate. Also report the missing and invalid counts separately.
- Report all metrics by dataset and for the equally weighted mean of the two datasets. The equal-weight mean of Acc@0.5 is the primary score.

## Baselines and controls

- **Frozen base Qwen2.5-VL-3B-Instruct.** Use the [pinned model revision and weight hashes](../../configs/model_artifacts.json). The base and adapter use the same image preprocessing, prompt and greedy decoding. For grounding, use the user text `Locate the object described by: {expression}. Return exactly one JSON array [x_min, y_min, x_max, y_max] with integer coordinates normalized to 0–1000. Return no other text.` with the image. Decode at most 64 new tokens. Divide the four output coordinates by 1,000. A single array of four integers is required.
- **Grounding DINO Swin-T zero-shot.** Use `groundingdino_swint_ogc.pth` SHA-256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`, with raw expressions through the standard provider and its matching tokenizer. Evaluate the Cartesian grid of box thresholds {0.20, 0.25, 0.30, 0.35} and text thresholds {0.15, 0.20, 0.25} using the select manifests only. Maximize pooled Acc@0.5 across the two equal-sized sets. Break ties by higher box threshold, then higher text threshold. Take the highest-confidence returned box and break confidence ties by first detector output order. Apply the winning threshold pair to eval once.
- **Centre-box prior.** Use the fixed box `[0.25, 0.25, 0.75, 0.75]` for every expression.
- **Shuffled expression.** For the adapter, order eval rows within each dataset by the SHA-256 of the UTF-8 string `17:shuffle:{parent}`, then give each row the expression from the next row, wrapping at the end. Keep the original image and gold box. The donor always belongs to a different parent. Report the number of donors whose wording happens to match.
- **Grey image.** For the adapter, replace the original image with constant RGB (128, 128, 128) at the same dimensions, keeping the expression and gold box.

## Decision and uncertainty

Compute paired adapter-minus-baseline differences for the two registered main comparisons. For each, bootstrap 10,000 replicates with NumPy `default_rng(17)`, sampling eval images with replacement separately within each dataset and using the same draws for each pair of conditions. Average the two dataset deltas per replicate. Use the 2.5th and 97.5th percentiles with linear interpolation for the two-sided 95% interval. Reuse these draws to report the control differences and intervals. A point tie or a confidence bound at zero cannot meet a strictly positive requirement.

**Grounding PASS** requires both adapter-minus-base and adapter-minus-DINO primary differences to be positive, each with a strictly positive lower 95% bound. The adapter's primary accuracy must also be strictly higher with real, matched inputs than with each control. The control intervals and per-dataset breakdowns are reported as context.

## Captioning

Score the base and adapter on the same VRSBench caption eval images with the prompt `Describe this aerial image in one detailed paragraph.` and greedy decoding up to 256 new tokens. Use the COCO caption implementation in `pycocoevalcap==1.2` with its PTB tokenizer and `Cider` scorer. Fit the scorer's reference document frequencies on the full frozen reference set once, then bootstrap the paired per-image CIDEr differences with 1,000 images drawn with replacement in each of 10,000 replicates, seed 17. The comparison uses the same reference-derived weights for both models. Report the point gain and two-sided percentile interval as for grounding. Captioning passes when the gain and its lower 95% bound are both strictly positive.

## Release

Grounding promotion requires a grounding pass. If grounding fails, record the failure and report the measured base-versus-DINO comparison. A captioning-only pass supports a labelled captioning-only release. If both fail, ship neither. All runs, ties, missing outputs and failures stay on the record.
