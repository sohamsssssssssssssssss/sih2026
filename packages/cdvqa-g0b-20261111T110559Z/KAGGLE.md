# Kaggle run book: CDVQA two-image pilot (G0b package)

Dataset slugs marked **GUESS** are placeholders for the upload you make. Fix them before running.
The base-weights dataset path is the one B13 already used (from the B13 package README).

## 0. Upload

| what | Kaggle path | status |
|---|---|---|
| this package (folder or zip) | `/kaggle/input/datasets/sohamssssss/cdvqa-g0b-package/` | **GUESS** slug `cdvqa-g0b-package` |
| CDVQA images (`images/im1`, `images/im2`, only the pairs in the manifests) | `/kaggle/input/datasets/sohamssssss/cdvqa-images/` | **GUESS** slug `cdvqa-images`; built from `cdvqa-images.zip` |
| pinned base weights | `/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned` | known (B13) |

Set `IMAGES` to the folder that contains `im1/` and `im2/` (the runner resolves `im1/<file>` and `im2/<file>` under it).

## 1. Environment (T4 runtime, Python 3.11)

```sh
pip install -q uv
uv venv -p 3.11 /kaggle/working/venv
. /kaggle/working/venv/bin/activate
cd /kaggle/input/datasets/sohamssssss/cdvqa-g0b-package/
uv pip install -r requirements-target.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"
```

`requirements-target.txt` has the same pins as the B13 package (torch 2.6.0, transformers 4.49.0, peft 0.15.2,
qwen-vl-utils 0.0.8, bitsandbytes 0.45.3). `bitsandbytes` 0.45.3 is listed as unverified in B13's config.

Use a writable working copy for runs, because the stages write `pilot-checkpoint.json` and budget files
under `/kaggle/working`. Export the variables below once:

```sh
export PKG=/kaggle/input/datasets/sohamssssss/cdvqa-g0b-package
export W=/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned
export IMAGES=/kaggle/input/datasets/sohamssssss/cdvqa-images
export LEDGER=/kaggle/working/ledger/cdvqa-budget.json
mkdir -p /kaggle/working/ledger
cd $PKG
```

## 2. Preflight (CPU, no weights)

```sh
CUDA_VISIBLE_DEVICES="" python pilot.py preflight --image-root $IMAGES
```

Expect `payload`, `manifests`, `leakage` = PASS and `train_pairs_resolved` = 8000.

## 3. Token preflight (CPU, real processor, needs weights)

```sh
CUDA_VISIBLE_DEVICES="" python pilot.py token-preflight --weights $W --image-root $IMAGES
```

Expect `all_fit: true` and an empty `failures` list. Do not start training until it passes.

## 4. G1: zero-shot base-only evaluation (no adapter)

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --output /kaggle/working/g1-base-test1 --budget-ledger $LEDGER --max-seconds 2400
```

Writes `result.json` with `base_real` only, the majority baseline, and per-type and macro accuracy.

## 5. Train (fresh output directory, one visible GPU)

```sh
python pilot.py train --weights $W --image-root $IMAGES \
  --output /kaggle/working/cdvqa-run-a --budget-ledger $LEDGER --max-seconds 3600
```

Check `/kaggle/working/cdvqa-run-a/training-report.json`. `status` is `training_complete` (1000 steps) or
`budget_stopped_incomplete`.

## 6. Resume an incomplete run

Use the newest checkpoint with `status: budget_stopped_incomplete`. Resume it in a **new** output directory.

```sh
CKPT=$(ls -d /kaggle/working/cdvqa-run-a/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)
python pilot.py train --weights $W --image-root $IMAGES --resume-from $CKPT \
  --output /kaggle/working/cdvqa-run-b --budget-ledger $LEDGER --max-seconds 3600
```

The resume is refused unless the checkpoint's `pilot-checkpoint.json` matches this package's identity,
every hash matches, the schedule is 1000 steps, and the checkpoint is incomplete.

## 7. Evaluate the adapter under all four conditions

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --adapter /kaggle/working/cdvqa-run-b/adapter \
  --output /kaggle/working/cdvqa-eval-test1 --budget-ledger $LEDGER --max-seconds 2400
```

Result fields: `conditions` (base_real, adapter_real, adapter_identical, adapter_swapped), `paired_differences`
(3 comparisons, 2000 pair-cluster bootstrap reps, seed 17), `majority_baseline`, `malformed_n` per condition.

## Budget (read before running)

- Shared ledger cap is 7200 s (inherited from B13). Each invocation is charged its `--max-seconds` up front.
  1000 training steps plus evaluations may not fit in one 7200 s ledger. Check the ledger file before each run.
  A new ledger needs a new approval.
- Per-invocation caps: train 3600 s, evaluate 2400 s. The B13 step rate is not measured for this run,
  so 1000 steps in one 3600 s window is unverified.
