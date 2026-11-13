# CDVQA G0b package (two-image change-VQA pilot, pre-GPU)

- Base: Qwen/Qwen2.5-VL-3B-Instruct, revision 66285546d2b821cf421d4f5eb2576359d3770cd3 (pinned, unchanged).
- Train: `data/manifests/cdvqa/train-8k.v2.jsonl`: 8,000 QA, 1,000 per question type, spread over all 1,600
  train pairs (5 per pair). v1 (`train-8k.v1.jsonl`, 500 pairs) is kept for reference only.
- Eval: `eval-test1-400.v1.jsonl`, `eval-test2-400.v1.jsonl` (50 per type, about 100 pairs each).
- Prompt: [before image, after image, question], with the wording "The first image is the earlier image and
  the second image is the later image" and the allowed answers for the question type.
- Conditions: base_real, adapter_real, adapter_identical (after := before), adapter_swapped (after, before).
- Training hyperparameters match B13: LoRA r16 a32 dropout 0.05 on q/k/v/o, lr 2e-4, batch 1, grad accum 8,
  image 392, seed 17, 4-bit NF4, fp16, 1000 steps, checkpoint every 16 steps.
- No images, no weights, no tokenizer files are in this package. See KAGGLE.md.
