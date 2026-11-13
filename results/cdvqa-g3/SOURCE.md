# CDVQA G3 evaluation: source and provenance

## Run
- Kaggle notebook **cdvqa-g3-eval, Version 1** (committed run), runtime 1 h 1 m 47 s, GPU T4 x2.
- Package **g0f**: `packages/cdvqa-g0f-20261113T115304Z.zip`, sha256 `4d9df4f534a1093ec47ea6070846795056c73c5503c3b6b1ecf1b372336a7f1c` (re-hashed locally, matches), built at commit `0f676f2`. g0f = g0e with the eval cap raised to 7,200 s; run identity in result.json and session.json: `47deaabb735210e86fc708d97dfb94ced4bfa81ed841f4dab2d29d320f9304cc`.
- Adapter: G2 `cdvqa-train-6/checkpoints/checkpoint-1000`, trained under g0e run identity `057f7d52d09e8497f11edc226916a828b84b6f712d778cf3e25e69e12591879a`. The adapter file hashes recorded by the runner in both result.json files (`adapter_identity`):
  - `adapter_config.json` `74ff7d6693c6e05d121a4d0915d5bb3981ae1fd3eb31ec64bc000469f73c2b2c`
  - `adapter_model.safetensors` `834cf85782e9e2eb549c2f76a63ed26a63477211c0dce2b85e25db3888d73fea`
  - The adapter files were not found in ~/Downloads, so these hashes were not re-computed locally. No weights are committed.
- Base: Qwen/Qwen2.5-VL-3B-Instruct rev `66285546d2b821cf421d4f5eb2576359d3770cd3`, fp16 eval. Revision proof: restored HF download metadata (pinned commit) plus weight sha256 match (`base_verification.passed: true`).
- Decoding greedy, `max_new_tokens` 32. Images 392x392, prompt order [before image, after image, question].
- Eval manifests (frozen before training): `eval-test1-400.v1` sha256 `4689eb8a45fecb6e0ca42f3c209d64425f3b9068091337469ece8da588e16993` and `eval-test2-400.v1` sha256 `f79da8027377422e15af62edf4797c412c1ed61241d10b2ee55a685362314425`. Each has 400 questions, 50 per type.
- Seed 17. Paired percentile bootstrap over image pairs, 2,000 reps (99 pairs in test1, 100 in test2).
- Budget ledger: test1 charged 1,818 s and test2 1,807 s of the 7,200 s cap. Both runs ended `{"exit_code": 0, "status": "completed"}`.
- The `.log` files hold only the exit status and a harmless `sitecustomize` / `wrapt` import warning; the scores come from result.json. The macro scores in result.json were checked against the values read from the notebook output, and all of them match. Malformed answers: 0 in every condition.

## Files (sha256)
- `ANALYSIS.md` `37fe9bc9c60123e96fce2e48a53bfa6deeae863f63aac4deaab390ff07953a1b`
- `analyze_g3.py` `df6c3ecf61703d56f6d2a813dc15fb4b5e99ac8749b29f7fd9a64c408af0be4f`
- `cdvqa-eval-budget.json` `54f68e4ed60b0bb7f899a32c7016bfa22ea4d6c66b0c45a49791fa2966032d6d`
- `g3-test1.log` `41211ecb4f4729fdbbd8c9ef06365b265e6d16308104c716b4784b3c6398101f`
- `g3-test2.log` `41211ecb4f4729fdbbd8c9ef06365b265e6d16308104c716b4784b3c6398101f`
- `test1-exit.json` `16427eab67f16dab2fe3399647dbbfb62f83f69bd6d53a70b5c333bfc6bab47f`
- `test1-result.json` `e6fe36f6f8eb9caf4f520a3c049c27b9fdc66e6731a81b58922797fd2877559b`
- `test1-session.json` `7c63ba0e9e7bce8e813510cffcfde816a8215103f385a40aa409641e80b5c421`
- `test1-worker.log` `d495f7953cd623d030a5a11c2fad922d68fd0653107ae41095bf5a08ddaf2e6b`
- `test2-exit.json` `16427eab67f16dab2fe3399647dbbfb62f83f69bd6d53a70b5c333bfc6bab47f`
- `test2-result.json` `7749e2168b54c46979cbff22c0c00edce31df040fb13ae2867f31445f4319602`
- `test2-session.json` `7c63ba0e9e7bce8e813510cffcfde816a8215103f385a40aa409641e80b5c421`
- `test2-worker.log` `b8e6b319f6331ab7497165eab3171b312522262bcfccb90bbd73001dab3ddafb`
- `SOURCE.md` is not listed here, because a file cannot contain its own hash.

Inputs were copied from the downloaded, already-unzipped Kaggle output (`results/` folder in ~/Downloads, files dated 14 Nov 2026 01:40). File renames: `g3-adapter-{split}/result.json` became `{split}-result.json`; `session.json`, `exit.json` and `worker.log` gained a `{split}-` prefix; `ledger/cdvqa-eval-budget.json` became `cdvqa-eval-budget.json`. `ANALYSIS.md` is produced by `analyze_g3.py`.

## Claim status
ALLOWED: macro accuracy of adapter vs base and vs majority on two frozen 400-question samples (single seed).
NOT ALLOWED: "uses temporal order"; "detects change without text priors"; any comparison to the paper's 73-75%; SOTA; anything about Cartosat/RISAT. cdvqa_validated stays false; the change route stays experimental_unvalidated.
