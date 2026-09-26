# Qwen remote-sensing adaptation

SatQuery's first trainable remote-sensing VLM path adapts `Qwen/Qwen2.5-VL-3B-Instruct` with PEFT LoRA. The default GPU configuration uses 4-bit NF4 base-weight quantization with fp16 computation, commonly called QLoRA. GPU feasibility is established only when the dry run and bounded training command succeed on the target runtime.

## Dataset choice and contract

The initial dataset is the official RSVQA-LR release from Zenodo record 6344334. It already has genuine remote-sensing images, questions, answers, and official train/validation/test splits, and the repository already uses its test split for frozen-Qwen evaluation. This is a smaller provisioning path than BigEarthNet.txt. BigEarthNet.txt contains text annotations but no pixels; its separate native imagery is still required before it could supply complete training examples.

Training never downloads data. `prepare_rsvqa_training_manifest.py` reads an already-provisioned RSVQA-LR directory and emits JSONL records with `image`, `instruction`, `response`, `dataset`, `source`, `split`, and `sample_id`. Loading fails on malformed JSON, absent or corrupt images, empty text or identity fields, duplicate sample identities, unsupported splits, or the same dataset image appearing across train and evaluation splits. Selection within a split is deterministic from the seed and sample identity.

Training uses only `split=train`. Adapter selection should use `validation`; the official `test` split remains held out for a final locked comparison. Dataset files and generated training artifacts remain outside Git.

## Adaptation configuration

Every setting below is configurable on the command line. Defaults are: 4-bit NF4 with double quantization, fp16 compute, LoRA rank 16, alpha 32, dropout 0.05, target modules `q_proj,k_proj,v_proj,o_proj`, image size 392, batch size 1, gradient accumulation 8, one epoch, learning rate `2e-4`, maximum 512 examples, and seed 17. Image size must be a multiple of Qwen's 28-pixel vision factor. `--quantization none` runs ordinary LoRA, and `--precision bf16` is available only for hardware that supports it. No default is a claim that a particular GPU can complete training.

Each prompt contains one resized image, the original RSVQA question, and the same concise-answer instruction used by the SatQuery Qwen provider. Only assistant target tokens contribute to loss. The base weights remain frozen and the selected attention projections receive LoRA adapters.

`training-report.json` records the base model ID and local path, the revision passed with `--model-revision` (otherwise `null`), checksum `null`, dataset manifest SHA-256, dataset/source identities, split, selected sample IDs and count, seed, all training and adapter settings, dependency versions, adapter location, runtime time, status, peak CUDA memory, wall-clock time, and, when training completes, trainer metrics, the full trainer log history and the adapter size. It contains no improvement claim. `kaggle/run_vlm_program.py` passes the pinned revision only after its env stage has checksum-verified the local files.

## Kaggle GPU procedure

This sequence assumes the reviewed migration commit has been pushed by a human. Replace every angle-bracket placeholder from the mounted Kaggle inputs; the scripts never download the model or dataset.

### 1. Check out and identify the exact code

```bash
git clone https://github.com/sohamsssssssssssssssss/sih2026.git /kaggle/working/sih2026
cd /kaggle/working/sih2026
git checkout <REVIEWED_MIGRATION_COMMIT_SHA>
test "$(git rev-parse HEAD)" = "<REVIEWED_MIGRATION_COMMIT_SHA>"
```

For a fully offline notebook, upload a repository archive instead and run the same `git rev-parse HEAD` check in the extracted checkout.

### 2. Install only provisioned dependencies

Use wheels compatible with the notebook's installed CUDA PyTorch. Network-free installation is:

```bash
python -m pip install --no-index --find-links /kaggle/input/satquery-wheels \
  -r requirements-training.txt
python - <<'PY'
import importlib.metadata as m
for name in ('torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes', 'qwen-vl-utils'):
    print(name, m.version(name))
PY
```

If compatible packages are already installed, record their versions and skip installation. Do not change versions during a result-producing run without recording the resulting environment.

### 3. Locate and verify the local base checkpoint

```bash
export SATQUERY_QWEN_MODEL_DIR=<LOCAL_QWEN_CHECKPOINT_DIRECTORY>
test -f "$SATQUERY_QWEN_MODEL_DIR/config.json"
test -f "$SATQUERY_QWEN_MODEL_DIR/preprocessor_config.json"
find "$SATQUERY_QWEN_MODEL_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
```

The directory must contain a complete local `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint. Revision and checksum remain `null` unless independently verified.

### 4. Provision and inspect RSVQA-LR

Set the root to the already extracted official RSVQA-LR release. The preparation script expects these metadata files at the root:

```text
LR_split_train_images.json       LR_split_train_questions.json       LR_split_train_answers.json
LR_split_val_images.json         LR_split_val_questions.json         LR_split_val_answers.json
LR_split_test_images.json        LR_split_test_questions.json        LR_split_test_answers.json
```

TIFFs may be below that root; they are discovered recursively by exact `<image_id>.tif` filename. Validate what was mounted before generating a manifest:

```bash
export RSVQA_ROOT=<EXTRACTED_OFFICIAL_RSVQA_LR_ROOT>
for split in train val test; do
  test -f "$RSVQA_ROOT/LR_split_${split}_images.json"
  test -f "$RSVQA_ROOT/LR_split_${split}_questions.json"
  test -f "$RSVQA_ROOT/LR_split_${split}_answers.json"
done
find "$RSVQA_ROOT" -type f -name '*.tif' | head
```

### 5. Generate and validate the manifest

```bash
export RSVQA_MANIFEST=/kaggle/working/rsvqa-lr.jsonl
python scripts/prepare_rsvqa_training_manifest.py \
  --dataset-root "$RSVQA_ROOT" \
  --out "$RSVQA_MANIFEST"
sha256sum "$RSVQA_MANIFEST"
wc -l "$RSVQA_MANIFEST"
```

The preparation command now runs the same full manifest/image validation used by training. It fails on missing or corrupt images, malformed records, duplicate identities, unsupported splits, or image leakage across splits.

### 6. Run the one-sample gradient dry run

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DRY_OUT=/kaggle/working/satquery-qwen-rsvqa-dry-run
python scripts/train_remote_sensing_adapter.py \
  --model-path "$SATQUERY_QWEN_MODEL_DIR" \
  --dataset-manifest "$RSVQA_MANIFEST" \
  --image-root "$RSVQA_ROOT" \
  --output-dir "$DRY_OUT" \
  --max-samples 1 --image-size 224 --batch-size 1 \
  --gradient-accumulation-steps 1 --precision fp16 \
  --quantization 4bit --lora-rank 16 --lora-alpha 32 --dry-run
python -m json.tool "$DRY_OUT/training-report.json"
test ! -e "$DRY_OUT/adapter"
```

The dry run constructs a real multimodal prompt, tokenizes and collates it, loads only local model files, attaches LoRA, and performs one finite forward/backward gradient check. It performs no optimizer step and writes no adapter.

### 7. Run the bounded first training job

Proceed only if the dry run passes and observed T4 memory is safe:

```bash
export TRAIN_OUT=/kaggle/working/satquery-qwen-rsvqa-first-run
python scripts/train_remote_sensing_adapter.py \
  --model-path "$SATQUERY_QWEN_MODEL_DIR" \
  --dataset-manifest "$RSVQA_MANIFEST" \
  --image-root "$RSVQA_ROOT" \
  --output-dir "$TRAIN_OUT" \
  --seed 17 --max-samples 128 --image-size 224 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --max-steps 10 --precision fp16 --quantization 4bit \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
python -m json.tool "$TRAIN_OUT/training-report.json"
test -f "$TRAIN_OUT/adapter/adapter_config.json"
```

This configuration discovers T4 feasibility; it is not a tuned recipe or performance claim.

### 8. Compare base and adapter on validation

```bash
export EVAL_REPORT=/kaggle/working/satquery-qwen-rsvqa-validation.json
python scripts/evaluate_remote_sensing_adapter.py \
  --model-path "$SATQUERY_QWEN_MODEL_DIR" \
  --adapter-path "$TRAIN_OUT/adapter" \
  --dataset-manifest "$RSVQA_MANIFEST" \
  --image-root "$RSVQA_ROOT" \
  --subset eval/subsets/rsvqa-lr-validation-dev1000.v1.json \
  --expected-subset-sha256 e3b43319645767c2c559fba82413f66818193689d4316e067f7339eb9ec4e20c \
  --image-size 252 --out "$EVAL_REPORT"
python -m json.tool "$EVAL_REPORT"
```

Omit `--adapter-path` to evaluate the base model alone, or add `--skip-base` to evaluate only the adapter. Use validation for configuration decisions and keep the official test split untouched until the configuration is locked. Metric definitions (strict, lenient, count-bin and invalid rate) are in `eval.rsvqa_research.METRICS`; case-insensitive exact match is a narrow metric and does not fully characterize remote-sensing VQA quality. The staged research program, including the locked final test, is `kaggle/run_vlm_program.py`; see `docs/research/vlm-kaggle-runbook.md` and `docs/research/rsvqa-lr-audit.md`.

### 9. Preserve evidence without committing large artifacts

Download or attach as external artifacts: the manifest (or its SHA-256), dry-run and training `training-report.json`, dependency/version output, relevant trainer logs, the complete adapter directory, and the validation report. Keep checkpoints, adapter weights, dataset pixels, and Kaggle output outside Git. Training completion alone does not demonstrate improvement, and these runs do not establish VRSBench, CDVQA, ISRO/SAC, or general remote-sensing performance.
