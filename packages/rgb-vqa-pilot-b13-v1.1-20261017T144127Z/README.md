# SatQuery B13 RGB VQA training package

This derivative preserves B11.4's pinned Qwen model, RGB processing, NF4/LoRA
settings, optimizer, checkpoint integrity policy, and frozen 389-question
validation/scoring. It trains a fresh adapter on the frozen B13 Strategy A
manifest: 6,000 questions, 558 images, one epoch, 750 optimizer steps, seed 17.
The B13 training manifest identity is
`16f97f1f332725d7ddf6cfbe643d1e0a61f20ac32f0ad83399bf5e367de01282`.
No weights, B12 checkpoints, or outputs are included. The B11.4 recovery probe
is excluded because its 128-step boundary is specific to B12.

With the pinned target environment and weights provisioned separately, run
these **CPU-only** checks from the package root before any training:

```sh
CUDA_VISIBLE_DEVICES="" /tmp/satquery-b6-env/bin/python pilot.py preflight
CUDA_VISIBLE_DEVICES="" /tmp/satquery-b6-env/bin/python pilot.py token-preflight \
  --weights /kaggle/input/datasets/sohamssssss/satquery-qwen-pinned
```

The second command verifies the pinned weight files and revision without
loading the model. It processes every B13 example with the real processor,
checks image tokens, prompt/answer masks and the 32,768-token limit, and prints
all per-example failures before returning a nonzero status. Do not proceed to
GPU training unless both checks pass. The first command records a SHA256 of the
seed-17 Trainer dataloader order after verifying a second pass has the same
order and contains all 6,000 sample IDs exactly once. The training session
records that order hash.

The training entry point remains `pilot.py train`. It requires separately
provisioned pinned weights, a **new** output directory, an external budget
ledger, and an explicit bounded `--max-seconds`. `--resume-from` must name a
verified incomplete B13 checkpoint from the same package/config/manifest
identity and starts in a new output directory. Checkpoint files are hashed and
include LoRA adapter, optimizer, scheduler, RNG and Trainer state; scaler state
must agree with the runtime. Saves occur every 16 steps and at final step 750.
The inherited per-invocation limit is 3,600 seconds and shared ledger limit is
7,200 seconds. Full-run sufficiency of those limits is unverified; assess and
approve a suitable bounded GPU budget before invoking training.

The unchanged B12 evaluation path remains available through `pilot.py evaluate`
with a separately provided adapter. The package contains the frozen B12
validation files and scoring code. It never selects official test questions.
