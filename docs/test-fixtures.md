# Stage-1 fixture provenance and restoration

Audit base: `36e9cb5`, 2026-11-12. Documentation-only branch `cleanup/runner-reconcile`, isolated worktree `/Users/soham/sih2026-runner-reconcile`. The original `/Users/soham/sih2026/packages/stage1-t4-smoke` directory and all other pre-existing untracked items were read-only. The B13 app worktree was not accessed.

## Findings

1. **The exact path has no reachable Git history.** `git log --all -- packages/stage1-t4-smoke` returns no commits, and `git ls-files packages/stage1-t4-smoke` returns no files. There is therefore no “last commit containing this directory” to restore with `git checkout` or `git archive`. This statement covers locally available refs; it does not claim knowledge of deleted/unfetched refs.
2. **It exists in the original checkout as untracked local material.** A new worktree gets tracked files only, explaining its absence in a clean checkout. The local package manifest describes CPU package preparation (`device: package preparation on CPU (py3.11.15 .venv-stage1)`), not a reproducible checked-in generator.
3. **No full-package generator or exact original generation command was found** in tracked source, unpacked local Python/shell/Markdown files, or the relevant reachable script history. Do not invent a `prepare_stage1_package.py` command: there is no such maintained entry point in this tree. `scripts/run_stage1_t4.py` consumes package configs and runs evidence stages; it does not build this fixture. `scripts/build_stage1_shards.py` and `scripts/extract_selected_s2.py` prepare different data products and do not reconstruct its configs, README, revision inventory or package manifest.
4. **A tracked derivative is available**, `packages/stage1-rgb-preserving-zero9-v1-20261015T134401792962Z`, preserved by `da6f50bc504116da3a91aca0d4c965a1176352ba`. It records `historical_package: packages/stage1-t4-smoke` and frozen historical digests. This is a *different path/package*, not a history commit for the missing original directory.
5. **Git alone cannot restore the whole payload.** The derivative includes configs, captions, normalization, revision, source and an inventory, but lacks `patches.tar.gz`. `.gitignore:41` (`*.tar.gz*`) excludes that archive. The existing local original archive is 1,592,788 bytes, SHA256 `b77d21791a91ff81328408e9701fb2b7ce48874fbf962d153c8ad3ecf520fa65`. It contains 192 TIFF files: 16 patches × 12 bands, no B10. Its bytes match both the historical and derivative manifests.

The original manifest SHA256 is `c84b7f83e87a06a087d04c672d6b0fbd42e3deb6387dc8b638ba74e76cbb1b57`; all **20 listed files** currently match their sizes and SHA256. Its `git_sha: 1ceb9d161e1f654da96b1f57b31724e48ac29250` describes preparation provenance, **not a commit tracking the fixture path**. Historical evidence is also recorded in `results/stage1-loop6-summary.json`, `results/stage1-loop7-summary.json`, and `results/stage1-revision-gate-package.json`.

## Exact offline restoration command (tested)

**This restores the existing historical local fixture; it does not regenerate pixels or reconstruct the original packaging process from Git alone.** It requires the original source directory to remain available. It writes only into the fresh cleanup worktree, refuses an existing target, and leaves the original untracked directory alone. No weights, network or accelerator are used; only Python 3.11's standard library is needed.

Run in a worktree where `packages/stage1-t4-smoke` is absent:

```bash
cd /Users/soham/sih2026-runner-reconcile
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python3.11 - <<'PY_RESTORE'
from pathlib import Path
import hashlib
import json
import shutil

source = Path('/Users/soham/sih2026/packages/stage1-t4-smoke')
target = Path.cwd() / 'packages/stage1-t4-smoke'
assert source.resolve() != target.resolve()
assert source.is_dir(), 'Original local fixture must be supplied separately'
assert not target.exists(), 'Refusing to overwrite an existing fixture'
manifest_bytes = (source / 'package-manifest.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == (
    'c84b7f83e87a06a087d04c672d6b0fbd42e3deb6387dc8b638ba74e76cbb1b57'
)
manifest = json.loads(manifest_bytes)
for name, entry in manifest['files'].items():
    path = source / name
    assert path.stat().st_size == entry['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
for name, entry in manifest['files'].items():
    path = target / name
    assert path.stat().st_size == entry['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
print('Restored and verified', len(manifest['files']), 'listed package files')
PY_RESTORE
```

Lightweight validation, also tested on CPU/offline:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python3.11 -m pytest --confcutdir=. -p no:cacheprovider \
  scripts/test_stage1_smoke.py::test_t4_smoke_package \
  scripts/test_stage1_smoke.py::test_t4_package_entry_and_stages -q
```

Observed result: **2 passed**. The audit's temporary restored copy was removed afterward; no generated package is part of this commit. The original local source directory is unchanged.

For a different computer with only a Git clone, **no complete regeneration command can currently be supplied** without separately provisioning the pinned archive or the original fixture. This is a reproducibility gap, not a CUDA requirement. Preserve the archive externally or add a small maintained generator/source-asset policy in a separately authorized implementation task.

## Tracked-derivative recovery alternative and its limit

From a checkout containing `da6f50bc504116da3a91aca0d4c965a1176352ba`'s tracked derivative, an offline recovery can copy that folder into an **absent** `packages/stage1-t4-smoke`, then copy in the separately supplied `patches.tar.gz` after verifying the digest above. It is about 1.9 MB, not a model bundle. No raw BigEarthNet download or Qwen model is needed.

This alternative was tested in the isolated cleanup worktree: **every derivative manifest size/hash passed**, the archive had 192 TIFFs, and both `src/scripts/run_stage1_smoke.py` and `run_stage1_t4.py` were byte-identical to root. However, the two lightweight tests yielded **1 passed, 1 failed**: `test_t4_package_entry_and_stages` expects the historical README phrases `hf download` and `private Kaggle dataset`, whereas the derivative ships an A2 initialization README. Do not call this an exact historical restoration or a passing full-suite solution.

Conversely, the exact historical local package passes those two tests, but its two packaged runner source files are **not** equal to current root sources; `test_updated_package_runner_matches_source` still has a real mismatch to resolve. Blindly copying either package does not establish that the whole suite passes. Editing an old manifest merely to conceal those differences would destroy the provenance distinction.

## Tests requiring these fixtures

The failure counts reported by the B13 agent concern its earlier checkout. Current root is `36e9cb5`, the original local fixture exists here, and test parametrizations/source have evolved. This audit did not rerun the full suite or assert that the exact earlier **5 failures / 40 setup errors** reproduces unchanged here.

| Test module / fixture | Required files and use |
|---|---|
| `scripts/test_stage1_smoke.py:10,29` | `test_t4_smoke_package`, `test_t4_package_entry_and_stages`: configs, captions, normalization, patch archive, revision, wrapper, packaged T4 runner, README. |
| `scripts/test_stage1_full.py:423` | `test_config_full_yaml_has_no_placeholder_steps`: `config-full.yaml`, expecting null step placeholders and a positive budget. |
| `scripts/test_stage1_revision_gate.py:15` | Fixture `proof` reads `config.yaml:22`; builds synthetic local weights/configs for the dependent tests listed below. |
| `scripts/test_stage1_resume_variant.py:15` | Fixture `tiny_run` reads `config-last4-vision-fp32.yaml:16`; all dependent tests use synthetic CPU models and rewritten configs. |
| `scripts/test_stage1_t4_package.py:27,151` | `PKG` and fixture `package_dir`: four smoke/last4 configs, captions, revision, normalization; separate integrity and root-source equality tests require the package manifest/source inventory. |

Only `scripts/test_stage1_t4_package.py` supports `SATQUERY_STAGE1_PACKAGE_ROOT` (introduced in the preserved training code). The other modules hard-code `packages/stage1-t4-smoke`, so setting that variable alone **does not fix the full group**.

### Fixture-dependent test functions at this audit base

`scripts/test_stage1_revision_gate.py` — `proof` dependency (parametrizations expand these names):

- `test_local_dir_metadata_proves_revision` (line 41)
- `test_missing_source_fails` (line 49)
- `test_metadata_mismatch_fails` (line 56)
- `test_weight_mismatch_fails` (line 64)
- `test_smoke_missing_source_never_reaches_model` (line 72)
- `test_config_source_has_priority` (line 84)
- `test_missing_weight_file_is_recorded_failure` (line 92)
- `test_empty_expected_weights_never_passes` (line 100)
- `test_failed_proof_is_not_overwritten` (line 155)

`scripts/test_stage1_resume_variant.py` — `tiny_run` dependency (parametrizations expand these names):

- `test_cpu_resume_restores_all_state_without_gpu_checkpoint_copy` (line 67)
- `test_resume_rejects_changed_identity` (line 96)
- `test_relocated_identical_config_preserves_identity` (line 114)
- `test_eval_cpu_load_and_shuffled_arm_keep_training_identity` (line 123)
- `test_missing_identity_and_bad_data_position_are_rejected` (line 143)
- `test_real_vision_finiteness_hooks_keep_gate` (line 159)
- `test_synthetic_new_initialization_metadata_and_resume` (line 199)

`scripts/test_stage1_t4_package.py` — `package_dir` dependency (parametrizations expand these names):

- `test_stage_verify_passes_when_hashes_match` (line 178)
- `test_stage_verify_aborts_on_weight_mismatch` (line 214)
- `test_stage_verify_aborts_on_revision_mismatch` (line 223)
- `test_stage_smoke_passes_first_attempt_and_records_rate` (line 234)
- `test_stage_smoke_falls_back_to_vision_fp32` (line 244)
- `test_stage_smoke_both_attempts_fail` (line 268)
- `test_stage_resume_stops_and_resumes_with_continuity` (line 279)
- `test_stage_resume_fails_beyond_threshold` (line 298)
- `test_stage_overfit_gates_pass_and_steps_sized_from_smoke` (line 327)
- `test_stage_overfit_shuffle_gate_fails_when_ratio_low` (line 345)
- `test_stage_extrapolate_uses_measured_rate` (line 364)
- `test_happy_path_runs_stages_in_registered_order` (line 374)
- `test_main_stops_at_first_failed_stage` (line 399)
- `test_memory_matrix_records_control_oom_and_variant` (line 431)
- `test_explicit_last4_fp32_reaches_resume_and_overfit` (line 452)
- `test_resume_only_runs_exact_variant_and_keeps_previous_evidence` (line 466)
- `test_resume_execution_failure_is_preserved` (line 487)

Additional direct package checks: `test_package_manifest_integrity` and `test_updated_package_runner_matches_source` in `scripts/test_stage1_t4_package.py`; the three direct tests in the table above. Helpers that only inspect imported root function signatures or constants do not by themselves require the package payload.

With test dependencies already installed, the exact command to exercise all five affected modules is:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python3.11 -m pytest \
  scripts/test_stage1_smoke.py scripts/test_stage1_full.py \
  scripts/test_stage1_revision_gate.py scripts/test_stage1_resume_variant.py \
  scripts/test_stage1_t4_package.py -q
```

Unlike restoration itself, these tests need the installed scientific/test stack (PyTorch, Transformers, PEFT, NumPy, rasterio, PyYAML, pytest and the dependencies their modules import). They mock actual pretrained inference/training. Do not use `python packages/stage1-t4-smoke/run.py all` to generate a fixture: it is an evidence/training consumer with model/GPU prerequisites.

## Evidence commands and boundaries

```bash
git log --all -- packages/stage1-t4-smoke
git ls-files packages/stage1-t4-smoke
git log --all -- packages/stage1-rgb-preserving-zero9-v1-20261015T134401792962Z
git log --all -G 'stage1-t4-smoke|patches.tar.gz' -- scripts '*.sh'
git grep -n -I -e stage1-t4-smoke -e patches.tar.gz -e SATQUERY_STAGE1_PACKAGE_ROOT
git check-ignore -v packages/stage1-t4-smoke/patches.tar.gz
```

No original package generator was found. No exact-path historical commit exists among reachable refs. Offline **restoration** was verified, not raw-data **regeneration**. No weights, GPU work, branch switches in active checkouts, push or merge occurred. Only this report and `docs/runner-reconciliation.md` are committed.
