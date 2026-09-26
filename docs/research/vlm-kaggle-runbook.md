# VLM Kaggle runbook (brief Days 1–8)

A single runner, `kaggle/run_vlm_program.py`, drives every GPU stage. Each stage runs the existing training or evaluation script in a fresh process and writes one immutable record to `<work>/registry/SQ-<date>-1XX.json`. FAILED and INCONCLUSIVE outcomes are recorded as well. Everything below is CPU-tested (`kaggle/test_run_vlm_program.py`). **Nothing here has run on a GPU yet.**

## What the user must provide

1. The completed canonical migration branch pushed to `https://github.com/sohamsssssssssssssssss/sih2026`. After all migration commits are final, record that final commit SHA and run from a clean clone pinned to it. Do not pin an intermediate migration commit.
2. A Kaggle account with GPU quota and Internet enabled for the notebook. This needs a phone-verified account; no credentials are stored in the repo.
3. Two private Kaggle Datasets, created after S1 and updated as you go (instructions below): `satquery-models` for the verified Qwen snapshot, and `satquery-vlm-state` for registry records, BASELINE V0 and the chosen adapters.

## Fixed protocol (do not change between sessions)

| setting | value | why |
|---|---|---|
| model | `Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546d2b821cf421d4f5eb2576359d3770cd3` | checksum-verified by the env stage in every session |
| GPU | one T4: `export CUDA_VISIBLE_DEVICES=0` | Kaggle's "T4 ×2" would let `device_map="auto"` shard the model, which changes timing and memory |
| image size | 252 | pixel-identical to the provider's native path (qwen-vl-utils 0.0.14); no upsampling |
| decoding | greedy, `max_new_tokens=32`, fp16 base at evaluation | |
| training | QLoRA NF4, fp16, batch 1 × accumulation 8, 1 epoch, dropout 0.05, `q,k,v,o_proj`, seed 17 | |
| selection data | `eval/subsets/rsvqa-lr-validation-dev1000.v1.json` (`e3b43319…`) | validation only; footprint-disjoint |
| final data | whole official test split (`b21e5046…`), once, after `lock` | |

## Session plan

The time estimates use **assumed, not measured** T4 rates (`ASSUMED_*` constants in the runner): 0.25–0.6 s per evaluated sample, 0.6–1.5 s per training sample, 30–120 s per model load. After S1, compare the measured `eval_seconds / n` from the base-eval record, and after S2 the `train_runtime_seconds / N=100`. Re-plan if either is outside its range.

| session | stages | est. T4 hours |
|---|---|---:|
| S1 (Days 1–3) | env, load-check, dry-run, base-eval (plus ~1–2 min of checksumming) | 0.1–0.3 |
| S2 (Days 4–5) | ladder N = 100 / 500 / 1,000 / 5,000 at r16, α32, lr 2e-4 | 1.4–3.7 |
| S3 (Day 6) | matrix at N = 1,000: r8 α16, r32 α64 (lr 2e-4), r16 at lr 1e-4 and 4e-4. The r16/2e-4 arm is the ladder run | 1.0–2.6 |
| S4 (Day 7) | train-eval of the chosen configuration, e.g. N = 20,000 | 3.4–8.6 (all 57,223 samples: 9.6–24, needs resume across sessions) |
| S5 (Day 8) | lock, then final-test with base and adapter on 10,004 test QA each | 1.4–3.4 |
| **total** | | **~7.4–18.5** (within one week's ~30 h Kaggle quota) |

## Commands

### Every session: set up (Internet on, accelerator "GPU T4")

```bash
cd /kaggle/working
git clone https://github.com/sohamsssssssssssssssss/sih2026.git satquery && cd satquery
export SATQUERY_CODE_SHA=<FINAL_CANONICAL_MIGRATION_COMMIT_SHA>
git checkout "$SATQUERY_CODE_SHA"
test "$(git rev-parse HEAD)" = "$SATQUERY_CODE_SHA"
pip install -q "qwen-vl-utils==0.0.14" peft bitsandbytes   # versions are recorded by the env stage; keep them fixed across S2-S5
export CUDA_VISIBLE_DEVICES=0
export WORK=/kaggle/working/satquery-vlm RSVQA_ROOT=/tmp/rsvqa_lr MANIFEST=/tmp/rsvqa-lr.jsonl
python -c "from pathlib import Path; from eval.suites.rsvqa import download_rsvqa_lr; download_rsvqa_lr(Path('$RSVQA_ROOT'))"
python scripts/prepare_rsvqa_training_manifest.py --dataset-root $RSVQA_ROOT --out $MANIFEST
mkdir -p $WORK && cp -r /kaggle/input/satquery-vlm-state/* $WORK/ 2>/dev/null || true   # from S2 on
```

Model: in **S1 only**, download the pinned snapshot. Use a path outside `/kaggle/working` (capped at about 20 GB and saved as notebook output); check `df -h /tmp` first. Then publish the folder as the private dataset `satquery-models`:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', revision='66285546d2b821cf421d4f5eb2576359d3770cd3', local_dir='/tmp/qwen2.5-vl-3b-instruct'))"
export MODEL_DIR=/tmp/qwen2.5-vl-3b-instruct              # S1
# export MODEL_DIR=/kaggle/input/satquery-models/qwen2.5-vl-3b-instruct   # S2 onward
```

Then define the runner once per session:

```bash
run() { python kaggle/run_vlm_program.py "$@" --work-dir $WORK --model-dir $MODEL_DIR --rsvqa-root $RSVQA_ROOT --manifest $MANIFEST; }
run env      # must be PASSED; every later stage refuses to run otherwise
```

### S1: Days 1–3

```bash
run load-check
run dry-run        # Day 2 gate: STOP and diagnose if this record is FAILED
run base-eval      # BASELINE V0; also copies the report to $WORK/rsvqa-qwen-base-v0.json
```

If base-eval is INCONCLUSIVE (a degenerate yes/no rate), stop. The number is not a measurement.

### S2: Days 4–5

```bash
run ladder         # 4 records; each has train loss history, peak VRAM, adapter size, dev metrics, McNemar vs V0
```

### S3: Day 6

```bash
run matrix --matrix-samples 1000
```

Decide the Day 7 configuration from the S2/S3 dev-subset records only: strict accuracy, its Wilson interval, the McNemar p-value against V0, and cost. Write the rationale down *before* S4.

### S4: Day 7

```bash
run train-eval --max-samples <N> --lora-rank <r> --lora-alpha <2r> --learning-rate <lr>
# If the session ends first: publish $WORK/<id>/checkpoints to satquery-vlm-state, then in the next session
# run train-eval ... --resume-from-checkpoint $WORK/<id>/checkpoints/checkpoint-<step>
```

### S5: Day 8 (the only time test is used)

```bash
run lock --adapter-path $WORK/<chosen-id>/adapter --selected-from SQ-... SQ-... --rationale "<one sentence>"
run final-test     # verifies the adapter SHA against the lock and the test subset SHA against b21e5046…
```

This reports official-split metrics (primary) and footprint-disjoint metrics (4 overlapping images removed) for both base and adapter, plus an exact McNemar test. Report the result whichever direction it goes.

## End of every session: what to bring back

Download from `$WORK` (small JSON only):

| from Kaggle | to the repo |
|---|---|
| `registry/SQ-*.json` (all, including FAILED and INCONCLUSIVE) | `eval/registry/` |
| `rsvqa-qwen-base-v0.json` | `eval/results/rsvqa-qwen-base-v0.json` (the Day 3 deliverable) |
| `<id>/eval-report.json`, `<id>/training-report.json` | `eval/results/vlm/<id>-eval.json`, `…-training.json` |
| final-test `eval-report.json` (about 6 MB; all per-sample predictions for Day 9) | `eval/results/vlm/<id>-final-test.json` |

Also update the private dataset `satquery-vlm-state` with `registry/`, `rsvqa-qwen-base-v0.json` and every candidate experiment directory `<id>/` (it must keep `adapter/` next to `training-report.json`, which `lock` reads). Restoring `registry/` at session start also keeps same-day experiment IDs unique.

**Never commit** adapters, `checkpoints/`, model weights, `*.safetensors`, dataset pixels, or `/kaggle` caches.
