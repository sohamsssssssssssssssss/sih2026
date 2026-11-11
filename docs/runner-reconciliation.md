# RGB pilot runner reconciliation

Audit base: `36e9cb5` (`training/cdvqa-g0` at task start), 2026-11-12. Reports live on `cleanup/runner-reconcile` in `/Users/soham/sih2026-runner-reconcile`. Neither active checkout was switched or edited. The two runner files were read, not changed, deleted, or moved.

## Recommendation

**Treat `packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/scripts/rgb_pilot_package.py` as the canonical frozen B13 runner.** Its 6,000-example/750-step contract, manifest guards, real-example dataloader checks, token preflight, and final checkpoint behavior match B13. Its SHA256 is recorded in the package's manifest and source-state inventory. Preserve that package as evidence; changing it in place changes the run/payload identity.

The root `scripts/rgb_pilot_package.py` is a **legacy B6-contract runner**, still consumed by 1,024-example/128-step smoke and checkpoint-recovery helpers. It is not a safe B13 entry point. No immediate deletion or overwrite is recommended. A future reconciliation should explicitly version the B6 consumers and port the B13 runner **with its companion verifier/config/manifests**, then generate packages from one maintained source. That migration is outside this report-only task.

## File identity and diff

| Copy | Bytes | SHA256 |
|---|---:|---|
| Root | 19,534 | `1cf5e7508e9619188be209339baa6c4d6162f6b5b59902666d2948070091700a` |
| B13 package | 31,621 | `6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac` |

Both files were preserved in commit `da6f50bc504116da3a91aca0d4c965a1176352ba`. They are divergent implementations, not byte-identical duplicates.

The AST comparison finds **no root-only top-level functions**, seven package-only functions, ten changed shared functions, and five unchanged shared functions. The package adds the B13 identity, manifest SHA256, sample-count and step constants. These are the substantive differences:

| Area (root line → package line) | Root behavior that the package no longer has | Package additions/replacement |
|---|---|---|
| `check_payload` (31 → 35) | Pilot payload and pilot validation only; accepts packages without B13-specific metadata. | Also verifies frozen B13 training manifest identity/digest/schedule and every training image digest. |
| `config` (40 → 75) | Selects `pilot.v1/pilot.jsonl`, the older B6 training selection. | Selects `b13-6k.v1/train.jsonl`. |
| `preflight` (91 → 126) | Requires 1,024 examples; walks one synthetic `range(1024)` Trainer dataset, checks 128 steps/batch 1/accumulation 8. | Requires 6,000 unique real example IDs; walks the real example list twice under seed 17, verifies identical order and exactly-once consumption, records Trainer order digest and 6,000 planned presentations. |
| Base verification (124 → 207) | Three-argument `verify_checkpoint`; metadata proof and weight hashes. | Passes `repository_id` to its package-local verifier, enabling the pinned official-HF LFS allowlist proof when local revision metadata is absent. Still hashes both weight files. |
| `binding` (154 → 238) | One `manifest: PILOT_IDENTITY` in identity binding. | Separately binds B13 training identity and frozen pilot validation identity. Existing checkpoint/run identities are therefore not interchangeable. |
| `verify_resume` (163 → 248) | Requires incomplete 128-step schedule. | Requires incomplete 750-step schedule. Both retain checkpoint inventory/hash/scaler/step checks. |
| `train` (197 → 282) | Hard-coded 1,024-example guard, completion at step 128; no B13 final-step save trigger. | Validates Trainer's 750-step/batch-1/accumulation-8 schedule, saves explicitly at final step 750, reports B13 training identity. Save interval remains 16. |
| `baseline` (249 → 339) | Non-resumable in-memory baseline; evaluates all real images, then all constant images; records include the full original row via `**row`. Does not require `args.resume`. | Durable, resumable per-question JSONL; interleaves real/constant for each row; validates saved prefix, scores and complete condition counts. Stores an explicit row-field subset plus condition, run config/identity and completed/expected counts. Requires `args.resume`. |
| `supervise` (270 → 452) | Always opens a fresh exclusive worker log. | Appends the worker log on baseline resume. |
| `main` (280 → 462) | Stages: preflight/inventory/baseline/train/evaluate; new output/session each time; ordinary result write. | Adds `token-preflight`, baseline `--resume`, session/ledger/config guards, baseline-completed refusal, Trainer-order session evidence, and durable baseline result publication. |

Package-only functions:

- `check_b13_spec:51`, `check_b13_examples:65`, `force_final_checkpoint:71` — frozen manifest/schedule/example/final-save guards.
- `tokenization_preflight:162` — CPU processor/collator walk over all training examples, image-token presence, exact answer masking, sequence limits and per-example failures, without loading model weights into a model.
- `_write_durable_json:406`, `_append_baseline_record:417`, `_read_baseline_progress:423` — fsync/atomic JSON, fsynced records, truncated-tail recovery and validated resume prefix.

Unchanged function ASTs: `paths`, `trainable_guard`, `synthetic_checks`, `inventory`, `installed`. Shared safety behavior includes local-only/offline loading, no automatic weight provisioning, frozen attention-only rank-16 LoRA policy, checkpoint hash/inventory checks, GPU worker supervision, and a locked external budget ledger. The 1,200/3,600/2,400-second baseline/train/evaluate limits and 7,200-second shared budget remain unchanged; their sufficiency for a full B13 run is not established by the diff.

**Porting hazards:** copying only the package runner into root would call `verify_checkpoint(..., repository_id=...)` against a root helper that lacks that keyword (`scripts/stage1_runtime.py:11` versus package helper `:19`). It would also break root B6 tests/helpers that require 1,024 examples, 128-step resume, and the old `baseline` argument shape. Both runners derive `ROOT` from their own file; the root CLI currently has no root-level `pilot-config.json`, `package-manifest.json` or `images/` package payload, so the existence of a root executable does not make it a complete standalone B13 package.

## Every observed caller/importer

Search scope: tracked files plus unpacked untracked package directories, including Python, shell, Markdown and JSON. Additional searches of notebook/YAML/text/TOML/JS/TS/RST/CFG/INI/bash/zsh files found no further module references. No archive was unpacked, notebook executed, external service queried, or code in the B13 app worktree read. Static searches cannot enumerate external notebook cells or code hidden only inside ZIP bytes.

### Root runner

| Location | Relationship |
|---|---|
| `scripts/probe_rgb_pilot_recovery.py:17` | Imports `REQUIRED_CHECKPOINT_FILES`, `binding`, `check_payload`, `config`, `trainable_guard`, `verify_base`, `verify_resume`. Its continuity and checkpoint checks still require 1,024 examples and a 128-step schedule. |
| `scripts/smoke_rgb_pilot_train.py:17` | Imports `binding`, `check_payload`, `config`, `trainable_guard`; `validate_frozen_config:68` expects 1,024 samples/128 steps. |
| `scripts/test_rgb_pilot_package.py:10` | Direct module import; synthetic policy/checkpoint/supervisor/Trainer tests exercise B6 semantics, including step 16 of 128. |
| `scripts/test_probe_rgb_pilot_recovery.py:11` | Indirect consumer through the recovery helper. |
| `scripts/test_smoke_rgb_pilot_train.py:13` | Indirect consumer through the smoke helper; fixture at line 19 refers to an **untracked** B10.4d package. |
| `results/cdvqa-g0/SOURCE.md:179,271` | Documentation/test-result references to root package-runner test names; not executable imports and not proof that the packaged B13 runner was tested. |

No other tracked root module, backend path, CLI wrapper or documentation command was found to call the root runner directly. It also has its own `if __name__ == '__main__': main()` entry point.

### B13 v1.1 package runner

All paths below are relative to `packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/`.

| Location | Relationship |
|---|---|
| `pilot.py:1` | Imports `main` and invokes it: the deployed package entry point. |
| `inventory-cell.py:4` | Spawns `pilot.py inventory --weights ...` using the current interpreter; requires package working directory. |
| `scripts/test_b13_runner.py:10` | Direct import; frozen 6k/750 guards, final save, resume integrity, real token-mask preflight and repeatable Trainer order. |
| `scripts/test_rgb_pilot_baseline_resume.py:8` | Direct import; durable baseline/resume corruption/config/CLI tests, invokes `baseline` and `main`. |
| `scripts/test_rgb_pilot_kaggle_metadata.py:8` | Direct import; upload metadata may be absent but runtime payload hashes remain required. |
| `README.md:16,17,30,41` | Commands/references to `pilot.py preflight`, `token-preflight`, `train`, `evaluate`. |
| `package-manifest.json:682,724`, `source-state.json:24` | SHA256/source-inventory references; not runtime callers. The recorded runner hash matches measured bytes. |

These imports all spell `scripts.rgb_pilot_package`, so the chosen file depends on `sys.path` and the already imported `sys.modules` entry. Running from root does **not** prove a package's tests exercised the package copy. Preserve a package-local process/working directory and verify `package.__file__` when running those tests. `pilot.py` and runner `ROOT` anchoring make the intended deployed package path clear.

### Other copies found during the requested grep

The premise names two copies of interest, but there are **additional unpacked historical copies** in the pre-existing untracked directories. Their `pilot.py` and local tests/helpers import their **own** same-named module, not necessarily either audited file. The raw search appendix below lists every observed literal match, including manifest-only references, so these are not silently misclassified as callers of today's root runner.

Searches used:

```sh
git grep -n -I -e rgb_pilot_package
rg -n --hidden --no-ignore \
  -g '!**/.git/**' -g '!**/.venv/**' -g '!**/node_modules/**' \
  -g '*.py' -g '*.md' -g '*.sh' -g '*.json' rgb_pilot_package
rg -n 'pilot.py|rgb_pilot_package' packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z -g '*.py' -g '*.md'
```

## Audit limits and prompt discrepancies

- Original status had **35 untracked entries** under `git status --porcelain --untracked-files=normal`, not 32. Directory entries are not counts of individual contained files. All original entries were left alone.
- More than two runner copies exist locally; the two specified byte sizes are correct.
- This task changed neither runner nor any CDVQA committed file. No model load, training, GPU work, push or merge was performed.

## Literal search appendix

The following 88 matches were captured before writing this report, relative to `/Users/soham/sih2026`. Package snapshots in this list are historical independent copies; JSON inventory references are evidence records, not imports.

```text
scripts/probe_rgb_pilot_recovery.py:17:from scripts.rgb_pilot_package import (
scripts/test_rgb_pilot_package.py:10:from scripts import rgb_pilot_package as package
results/cdvqa-g0/SOURCE.md:179:- Existing B13 package runner tests (`scripts/test_rgb_pilot_package.py`, `scripts/test_smoke_rgb_pilot_train.py`):
results/cdvqa-g0/SOURCE.md:271:- Existing B13 package-runner tests (`scripts/test_rgb_pilot_package.py`, `scripts/test_smoke_rgb_pilot_train.py`): 35 passed.
scripts/smoke_rgb_pilot_train.py:17:from scripts.rgb_pilot_package import binding, check_payload, config, trainable_guard
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/source-state.json:24:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/package-manifest.json:682:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/package-manifest.json:724:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/package-manifest.json:699:    "scripts/rgb_pilot_package.py": "fb53fac663bc2f5ecb242cd7fcbd417befbd59c925e708de61fae7f4b7664a7d",
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/package-manifest.json:706:    "scripts/test_rgb_pilot_package.py": "c3d225d1d11590a5edbe6d0d4f03ebb04d298f7ff910eedd68786dba196d4635",
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/package-manifest.json:733:      "scripts/test_rgb_pilot_package.py"
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/package-manifest.json:756:    "scripts/rgb_pilot_package.py": "fb53fac663bc2f5ecb242cd7fcbd417befbd59c925e708de61fae7f4b7664a7d",
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/package-manifest.json:760:    "scripts/test_rgb_pilot_package.py": "c3d225d1d11590a5edbe6d0d4f03ebb04d298f7ff910eedd68786dba196d4635",
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z/scripts/test_b13_runner.py:10:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/probe_rgb_pilot_recovery.py:17:from scripts.rgb_pilot_package import (
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/test_rgb_pilot_package.py:10:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/test_rgb_pilot_b10_hardening.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/scripts/test_rgb_pilot_b10_hardening.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/scripts/smoke_rgb_pilot_train.py:17:from scripts.rgb_pilot_package import binding, check_payload, config, trainable_guard
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/scripts/smoke_rgb_pilot_train.py:17:from scripts.rgb_pilot_package import binding, check_payload, config, trainable_guard
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/source-state.json:21:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/source-state.json:22:    "scripts/rgb_pilot_package.py": "fb53fac663bc2f5ecb242cd7fcbd417befbd59c925e708de61fae7f4b7664a7d",
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/source-state.json:26:    "scripts/test_rgb_pilot_package.py": "c3d225d1d11590a5edbe6d0d4f03ebb04d298f7ff910eedd68786dba196d4635",
packages/rgb-vqa-pilot-b11.4-20261016T184603Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/package-manifest.json:698:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4h-20261016T180940Z/package-manifest.json:738:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/source-state.json:687:    "?? packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/source-state.json:785:    "?? scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/source-state.json:788:    "?? scripts/test_rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/source-state.json:801:    "scripts/rgb_pilot_package.py": "43a2fd4877c49d617c1edcf1faba226b18c3a424beb1bb0e1c17656b8c694125",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/package-manifest.json:15:    "scripts/rgb_pilot_package.py": "43a2fd4877c49d617c1edcf1faba226b18c3a424beb1bb0e1c17656b8c694125",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z-b9-baseline-resume/package-manifest.json:701:    "scripts/rgb_pilot_package.py": "43a2fd4877c49d617c1edcf1faba226b18c3a424beb1bb0e1c17656b8c694125",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/scripts/test_b13_runner.py:10:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/source-state.json:687:    "?? packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/source-state.json:785:    "?? scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/source-state.json:788:    "?? scripts/test_rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/source-state.json:801:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/package-manifest.json:15:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b9.2-20261016T161544Z/package-manifest.json:719:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/source-state.json:687:    "?? packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/source-state.json:785:    "?? scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/source-state.json:788:    "?? scripts/test_rgb_pilot_package.py",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/source-state.json:801:    "scripts/rgb_pilot_package.py": "bafa61ca98958606bfa05e4b014fdc92d4fd7fe0829d5f4677a1fe39890f88e5",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/package-manifest.json:15:    "scripts/rgb_pilot_package.py": "bafa61ca98958606bfa05e4b014fdc92d4fd7fe0829d5f4677a1fe39890f88e5",
packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/package-manifest.json:701:    "scripts/rgb_pilot_package.py": "bafa61ca98958606bfa05e4b014fdc92d4fd7fe0829d5f4677a1fe39890f88e5",
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/source-state.json:24:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/package-manifest.json:687:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b13-v1-20261017T142249Z/package-manifest.json:729:    "scripts/rgb_pilot_package.py": "6b1e5dd8743be6ac18a6a7edd5a4e10344095c1c57cc3e1d3eb523ee86cb35ac",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/scripts/test_rgb_pilot_b10_hardening.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/source-state.json:21:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/package-manifest.json:686:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4a-20261016T173106Z/package-manifest.json:724:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/package-manifest.json:15:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/package-manifest.json:702:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/source-state.json:687:    "?? packages/rgb-vqa-pilot-b6-v1-20261015T173547Z/scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/source-state.json:785:    "?? scripts/rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/source-state.json:788:    "?? scripts/test_rgb_pilot_package.py",
packages/rgb-vqa-pilot-b9.1-20261016T160446Z/source-state.json:801:    "scripts/rgb_pilot_package.py": "adcb4db582255795b8b3963e0f04904c753a9b4b06ee35cead81c1f1adf86e68",
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/scripts/test_rgb_pilot_baseline_resume.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/scripts/test_rgb_pilot_kaggle_metadata.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/scripts/test_rgb_pilot_b10_hardening.py:8:from scripts import rgb_pilot_package as package
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/source-state.json:21:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/pilot.py:1:from scripts.rgb_pilot_package import main
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/scripts/smoke_rgb_pilot_train.py:17:from scripts.rgb_pilot_package import binding, check_payload, config, trainable_guard
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/package-manifest.json:697:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
packages/rgb-vqa-pilot-b10.4d-20261016T175351Z/package-manifest.json:737:    "scripts/rgb_pilot_package.py": "db0834089b6e06b0c1154af4b879f7c2465b56398fdf6504d51a571e726f545a",
```
