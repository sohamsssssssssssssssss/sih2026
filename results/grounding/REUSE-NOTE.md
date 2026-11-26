# Grounding baseline Qwen reuse

The only accepted producer is `0641ab77c94a6e39c50bf4d30c25e01eb3bc2d15`. That Kaggle job finished all four Qwen conditions before failing in the first DINO condition. Recoverability depends on whether the failed Kaggle version retained its output files.

| Eligible filename | Expected lines |
| --- | ---: |
| `qwen-eval.jsonl` | 2,000 |
| `qwen-shuffled.jsonl` | 2,000 |
| `qwen-grey.jsonl` | 2,000 |
| `qwen-caption.jsonl` | 1,000 |

The runner derives the counts from the verified pack. Use `--reuse-dir`, `--reuse-manifest`, and `--reuse-manifest-sha256` together. The manifest is JSON with a `commit` string and a `files` object mapping each recovered basename to its lowercase SHA-256. A valid manifest can pin a subset and the runner generates the other conditions. The DINO-only Kaggle cell requires the exact four-file dataset and its four independently supplied pins. Reuse validation checks the manifest hash, producing commit, eligible filenames, file hashes and counts. The runner writes the verified bytes and records the producing commit, line count and digest per file in `run-info.json`, alongside the reuse manifest digest. The separate run commit in that receipt describes the current run. The producing commit pin is an operator attestation about the recovered files.

## Qwen source comparison

Compared the runner to the producer with `git diff 0641ab77c94a6e39c50bf4d30c25e01eb3bc2d15 -- kaggle/run_grounding_baselines.py`. Four fragment comparisons using `git diff --no-index --exit-code` returned 0 with empty diff output:

- The base Qwen backend, after removing the optional adapter parameter and the adapter-only block that the base run skips. The processor, model load, chat template, generation and decode statements are byte-identical.
- The Qwen row logic, after dedenting the moved block and normalizing its two control bindings (`model == "qwen"` versus `model != "dino"`, and `args.stub` versus `stub`). The image preparation, donor fields, prompt selection, token limits and result construction are byte-identical.
- The grounding and caption prompts.
- The box parser and shuffle implementation.

`git diff --exit-code 0641ab77c94a6e39c50bf4d30c25e01eb3bc2d15 -- configs/model_artifacts.json kaggle/grounding_qwen_hashes.json kaggle/grounding_qwen_inputs.py results/grounding/MANIFEST.json 'results/grounding/*-eval.jsonl'` also returned 0 with empty output. The baseline still selects the same four Qwen conditions in the same order. The runner resets the original seeds after the DINO preflight and isolates the extra Qwen inference in a Torch RNG fork. Output JSON uses the original serialization and UTF-8 bytes. This source comparison establishes equivalence of the per-row inference path. It does not establish bitwise reproducibility across GPU environments.
