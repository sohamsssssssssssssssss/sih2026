# Kaggle run book: CDVQA two-image pilot (CDVQA pilot)

Placeholders: `TIMESTAMP` in the package folder name is filled in when the package is built. Slugs marked
**GUESS** are placeholders for the uploads you make. The weights dataset slug is the one B13 already used.

## 0. Uploads and paths

Kaggle unpacks each dataset to `/kaggle/input/datasets/<owner>/<slug>/`. The package folder sits **inside** its
dataset, so the package path includes the inner folder:

| what | Kaggle path | status |
|---|---|---|
| this package | `/kaggle/input/datasets/sohamssssss/cdvqa-pilot-package/cdvqa-g0d-TIMESTAMP/` | slug **GUESS** `cdvqa-pilot-package` |
| `cdvqa-images.zip`, unzipped | `/kaggle/input/datasets/sohamssssss/cdvqa-images/` (contains `im1/`, `im2/`) | slug **GUESS** `cdvqa-images` |
| pinned base weights (flat folder of files) | `/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned/` | known (B13) |

Check the layout once: `ls $IMAGES` must show `im1` and `im2`, and `ls $PKG` must show `pilot.py`.

## 1. Environment (T4, Python 3.11)

```sh
pip install -q uv
uv venv -p 3.11 /kaggle/working/venv
. /kaggle/working/venv/bin/activate
cd $PKG
uv pip install -r requirements-target.txt \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"
```

`requirements-target.txt` has the same pins as B13 (torch 2.6.0, transformers 4.49.0, peft 0.15.2,
qwen-vl-utils 0.0.8, bitsandbytes 0.45.3). bitsandbytes 0.45.3 is listed as unverified in the B13 config.

Export the variables once:

```sh
export PKG=/kaggle/input/datasets/sohamssssss/cdvqa-pilot-package/cdvqa-g0d-TIMESTAMP
export SRC_W=/kaggle/input/datasets/sohamssssss/satquery-qwen-pinned
export W=/kaggle/working/qwen-pinned
export IMAGES=/kaggle/input/datasets/sohamssssss/cdvqa-images
export TRAIN_LEDGER=/kaggle/working/ledger/cdvqa-train-budget.json   # cap 25200 s (pilot-config)
export EVAL_LEDGER=/kaggle/working/ledger/cdvqa-eval-budget.json     # cap 21600 s (pilot-config)
mkdir -p /kaggle/working/ledger
```

## 2. Weights setup (required before any stage that loads weights)

Kaggle datasets drop hidden directories. `verify_checkpoint` accepts the pinned revision only from `config.json`
`_commit_hash` or from `.cache/huggingface/download/config.json.metadata`, so the pinned revision must be restored
by hand. The runner is **not** weakened: the weight sha256 check still runs on the real files, and the revision
check still must pass.

```sh
mkdir -p $W/.cache/huggingface/download
for f in $SRC_W/*; do ln -sf "$f" $W/; done                      # symlink the weight files, no copies
printf '%s\n' 66285546d2b821cf421d4f5eb2576359d3770cd3 > $W/.cache/huggingface/download/config.json.metadata
```

What this proves on Kaggle: the revision proof is the restored metadata (pinned commit
66285546d2b821cf421d4f5eb2576359d3770cd3), **plus** the sha256 match of each weight file, which `verify_base`
checks on every stage. Each result records this in `base_verification.revision_source` and `revision_proof`.

## 3. Preflight (CPU, no weights)

```sh
cd $PKG
CUDA_VISIBLE_DEVICES="" python pilot.py preflight --image-root $IMAGES
```

Expect `payload`, `manifests`, `leakage` = PASS and `train_pairs_resolved` = 8000.

## 4. Token preflight (CPU, real processor; needs step 2)

```sh
CUDA_VISIBLE_DEVICES="" python pilot.py token-preflight --weights $W --image-root $IMAGES
```

Expect `all_fit: true`, `max_sequence_length: 32768`, `revision_source: hf_download_metadata`, and no failures.
Do not start training until this passes.

## 5. G1: zero-shot base-only evaluation

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --output /kaggle/working/g1-base-test1 --budget-ledger $EVAL_LEDGER --max-seconds 2400
```

Writes `result.json` with `base_real` only, the majority baseline, and per-type and macro accuracy.

## 6. Train (fresh output directory, one visible GPU)

```sh
python pilot.py train --weights $W --image-root $IMAGES \
  --output /kaggle/working/cdvqa-run-a --budget-ledger $TRAIN_LEDGER --max-seconds 3600
```

Check `/kaggle/working/cdvqa-run-a/training-report.json`: `status` is `training_complete` (1000 steps) or
`budget_stopped_incomplete`.

## 7. Resume an incomplete run (repeat until complete)

```sh
CKPT=$(ls -d /kaggle/working/cdvqa-run-a/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)
python pilot.py train --weights $W --image-root $IMAGES --resume-from $CKPT \
  --output /kaggle/working/cdvqa-run-b --budget-ledger $TRAIN_LEDGER --max-seconds 3600
```

Use the newest checkpoint from the latest run and resume into a **new** output directory each time. A resume is
refused unless the checkpoint's identity, file hashes, scaler state and schedule (1000 steps) match this package.

## 8. Evaluate the adapter under all four conditions

```sh
python pilot.py evaluate --weights $W --image-root $IMAGES --eval test1 \
  --adapter /kaggle/working/cdvqa-run-b/adapter \
  --output /kaggle/working/cdvqa-eval-test1 --budget-ledger $EVAL_LEDGER --max-seconds 7200
```

Result fields: `conditions` (base_real, adapter_real, adapter_identical, adapter_swapped), `paired_differences`
(3 comparisons, 2000 pair-cluster bootstrap reps, seed 17), `majority_baseline`, and `malformed_n` per condition.

## Budget

- Training ledger cap: 25200 s = 7 invocations of 3600 s. Each invocation is charged its `--max-seconds` up front.
- Evaluation ledger cap: 21600 s, separate from training (test1 + test2 G3 plus headroom).
- Per-invocation cap for train: 3600 s. Per-invocation cap for evaluate: 7200 s.
- B13's step rate is not measured for this run. 1000 steps inside 3600 s per invocation is unverified.
- A 4-condition adapter evaluation of 400 QA is 1600 generations. Use the 7200 s evaluation allowance; runtime remains hardware-dependent.

## G0f evaluation-cap provenance

Eval-cap change only vs g0e; adapter trained with g0e run_identity 057f7d52d09e8497f11edc226916a828b84b6f712d778cf3e25e69e12591879a.

Only evaluation wall-clock allowances changed: evaluate invocation 2400 → 7200 s, evaluation ledger 7200 → 21600 s. All training allowances, prompts, decoding, scoring, image sizes, conditions, bootstrap settings and manifests are unchanged. Use the adapter trained with g0e for evaluation. The g0f binding identifies this evaluation package; it is not the adapter training identity. Do not use g0f to resume a g0e training checkpoint.
