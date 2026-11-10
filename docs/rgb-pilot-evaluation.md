# Frozen RGB pilot evaluation v1

Prepared evaluation code only; no model evaluation was run during B5.
The original three-channel Qwen projection is used. A2 twelve-band initialization
and Run 03 trained states are outside this path.

For a separately authorized run with verified, provisioned local artifacts:

```bash
python scripts/evaluate_remote_sensing_adapter.py \
  --pilot-provenance data/manifests/rsvqa/pilot.v1/provenance.json \
  --dataset-manifest data/manifests/rsvqa/pilot.v1/validation.jsonl \
  --image-root /Users/soham/Downloads/6344334/Images_LR \
  --model-path <pinned-local-base> --adapter-path <local-safe-adapter> \
  --image-size 392 --seed 17 --bootstrap-samples 2000 --out <fresh-report.json>
```

Pilot mode pins manifest identity
`1eb622b1ca4bec1d2d849f049623f015f3e015c423ce8fab146e91e5b86f43ea`.
It evaluates all 389 frozen validation rows, ignoring the legacy `--max-samples`
option. It validates every frozen manifest and selected TIFF hash before model
loading and again before publishing a report. The pinned base revision and both
weight checksums must pass the existing verifier. Adapters require
`adapter_config.json` and `adapter_model.safetensors`; both are hashed without
pickle loading and rechecked after evaluation.

Four conditions share row order, image dimensions, image entries, question,
answer instruction, processor, explicit resize, and greedy decoding:
base/real, base/constant, adapter/real, adapter/constant. Constant input is an
in-memory RGB image filled with (127,127,127), constructed at the source image's
size before the same resizing path. It retains image tokens; it is not a
text-only baseline. Both models are evaluated with gradients disabled. No blind
training is performed.

Scoring version `rsvqa-rgb-pilot-exact-raw-integer-v1` is **our pilot policy**,
not a claim to implement official RSVQA category scoring. Strip and casefold
answers, then require exact equality. Presence/comparison accept yes/no;
rural/urban accepts rural/urban. Counts require canonical nonnegative decimal
integers (`0` or `[1-9][0-9]*`): no bins, tolerance, number words, punctuation,
signs, leading zeros, or numeric yes aliases. Malformed predictions score
incorrect and are counted separately for each type. Malformed gold fails.

Accuracy includes malformed predictions in the denominator. Macro accuracy is
the equal-weight mean of the four type accuracies. Differences are paired:
adapter-real minus base-real, real minus constant within each model, and
adapter-constant minus base-constant. Bootstrap samples whole images with
replacement, retaining all their QA rows in every condition. Within-type
accuracy is QA-weighted; macro weights types equally. Intervals are percentile
95% intervals with a recorded seed and repetition count. Missing-type replicates
are excluded from that type's interval; macro replicates must contain all four
types. Valid-replicate counts are reported. Fewer than two image clusters yields
no interval. These exploratory intervals do not establish a benchmark result.

Reports include QA/image IDs and types, predictions, malformed flags, all
condition metrics, manifest identity, image hashes, verified base revision and
hashes, adapter identity, processor configuration, prompt/chat template,
decoding configuration, and installed package versions.
