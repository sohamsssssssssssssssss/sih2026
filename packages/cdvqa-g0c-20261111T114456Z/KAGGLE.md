# Kaggle run book: CDVQA two-image pilot (G0c package)

Slugs marked **GUESS** are placeholders for the uploads you make. Fix them before running.
The base-weights path is the one B13 already used.

## 0. Uploads

| what | Kaggle path | status |
|---|---|---|
| this package (folder or zip) | `/kaggle/input/datasets/sohamssssss/cdvqa-g0c-package/` | **GUESS** slug `cdvqa-g0c-package` |
| `cdvqa-images.zip`, unzipped | `/kaggle/input/datasets/sohamssssss/cdvqa-images/` | **GUESS** slug `cdvqa-images` |
| pinned base weights | `/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned` | known (B13) |

`cdvqa-images.zip` contains only `im1/` and `im2/` (3,572 images: every pair referenced by `train-8k.v2` and both
eval manifests, no labels). Its root is the image root, so `$IMAGES` must be the folder that directly contains
`im1/` and `im2/`. Check that with `ls $IMAGES`.

## 1. Environment (T4, Python 3.11)

```sh
pip install -q uv
uv venv -p 3.11 /kaggle/working/venv
. /kaggle/working/venv/bin/activate
cd /kaggle/input/datasets/sohamssssss/cdvqa-g0c-package/
uv pip install -r requirements-target.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"
```

`requirements-target.txt` has the same pins as the B13 package (torch 2.6.0, transformers 4.49.0, peft 0.15.2,
qwen-vl-utils 0.0.8, bitsandbytes 0.45.3). bitsandbytes 0.45.3 is listed as unverified in the B13 config.

Export the variables once:

```sh
export PKG=/kaggle/input/datasets/sohamssssss/cdvqa-g0c-package
export W=/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned
export IMAGES=/kaggle/input/datasets/sohamssssss/cdvqa-images
export TRAIN_LEDGER=/kaggle/working/ledger/cdvqa-train-budget.json   # cap 25200 s (from pilot-config)
export EVAL_LEDGER=/kaggle/working/ledger/cdvqa-eval-budget.json     # cap 7200 s (from pilot-config)
mkdir -p /kaggle/working/ledger
cd $PKG
```

Stages write only under `/kaggle/working`. Each stage's ledger is chosen by you: give training runs `$TRAIN_LEDGER`
and evaluation runs `$EVAL_LEDGER`. The cap for each ledger comes from `pilot-config.json` (`budget.ledger_total_seconds`),
so `train` uses 25200 s and `evaluate` uses 7200 s. `--ledger-total N` overrides the cap for one run.

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

## 4. G1: zero-shot base-only evaluation

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --output /kaggle/working/g1-base-test1 --budget-ledger $EVAL_LEDGER --max-seconds 2400
```

Writes `result.json` with `base_real` only, the majority baseline, and per-type and macro accuracy.

## 5. Train (fresh output directory, one visible GPU)

```sh
python pilot.py train --weights $W --image-root $IMAGES \
  --output /kaggle/working/cdvqa-run-a --budget-ledger $TRAIN_LEDGER --max-seconds 3600
```

Check `/kaggle/working/cdvqa-run-a/training-report.json`: `status` is `training_complete` (1000 steps) or
`budget_stopped_incomplete`.

## 6. Resume an incomplete run (repeat until complete)

Use the newest checkpoint from the latest run, and resume into a **new** output directory each time.

```sh
CKPT=$(ls -d /kaggle/working/cdvqa-run-a/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)
python pilot.py train --weights $W --image-root $IMAGES --resume-from $CKPT \
  --output /kaggle/working/cdvqa-run-b --budget-ledger $TRAIN_LEDGER --max-seconds 3600
```

A resume is refused unless the checkpoint's `pilot-checkpoint.json` matches this package's identity, every
hash matches, the schedule is 1000 steps, and the checkpoint is incomplete.

## 7. Evaluate the adapter under all four conditions

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --adapter /kaggle/working/cdvqa-run-b/adapter \
  --output /kaggle/working/cdvqa-eval-test1 --budget-ledger $EVAL_LEDGER --max-seconds 2400
```

Result fields: `conditions` (base_real, adapter_real, adapter_identical, adapter_swapped), `paired_differences`
(3 comparisons, 2000 pair-cluster bootstrap reps, seed 17), `majority_baseline`, and `malformed_n` per condition.

## Budget

- Training ledger cap: 25200 s = 7 invocations of 3600 s. Each invocation is charged its `--max-seconds` up front.
- Evaluation ledger cap: 7200 s, separate from training.
- Per-invocation cap for train: 3600 s (resume across Kaggle sessions as needed).
- Per-invocation cap for evaluate: 2400 s (inherited from B13). A 4-condition adapter evaluation of 400 QA
  is 1600 generations; whether it fits in 2400 s is unmeasured on this hardware.
- B13 step rate is not measured for this run, so 1000 steps inside 3600 s per invocation is unverified.
