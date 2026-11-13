# CDVQA G0 evidence

Gate G0 for the change-detection VQA adapter: data, manifests, majority baseline, runner support.
Status: **images not yet available** (manual Google Drive step, see below). Everything else is CPU-only and done.

## Data sources

QA JSON (Apache-2.0, https://github.com/YZHJessica/CDVQA). Downloaded to `~/data/cdvqa/qa/` (not in git):

| file | sha256 |
|---|---|
| Test2_answers.json | 33ea8ccd9ee219e558665c1a570390ccd52ffe54526ee410b46cffc6420cdba4 |
| Test2_images.json | 4ed7979ceca614c1c91c133e227a8a06d769040ab4f130d88350714f80b2ef07 |
| Test2_questions.json | 89245ef8f77e14b5a3a6b86dd4b3c6992fbd8ac5b816d604ea6b36104ab9408b |
| Test_answers.json | 6f1b7bb5db44f8fb003e12d44c685b1a84478986e1874d14c1f314543dc40ee1 |
| Test_images.json | 99b8151791ab4c4e6e57cb8d548dbb7c0e93c0890692379d61f63e4a940f1ba5 |
| Test_questions.json | 58eb0e091d143c719a979bb7b8fa34ff9bcabae126ee40b29309c699c806d9e2 |
| Train_answers.json | 8858fd1c281459f9a4d8422da58a7d958513e0037bf5f6b3dc2838e347f1e610 |
| Train_images.json | b75ac75022abb1ff9ddbbf239d629351f18d94ce11ce3119c6a17089c82f9a9c |
| Train_questions.json | 21a1ee7cb0de8e6873736461e83447c9afc7075c8f0788724d799df906679958 |
| Val_answers.json | 38b613a3190049971627893f896310b6ddf711fc2fd44c01d635c19a241d0b5d |
| Val_images.json | 72e3adda2a41e5ca56685652012bea42619baa9dce4469d0b98f3340446a37f0 |
| Val_questions.json | 45343b49d8b0a0bcdbd246367c3d8243e517732f08cea74407130e657912ac80 |

Source URL pattern: `https://raw.githubusercontent.com/YZHJessica/CDVQA/main/<file>`.

### Images: BLOCKED on a manual step

The CDVQA README only has a citation. The SECOND dataset page is https://captain-whu.github.io/SCD/ and
links the dataset to Google Drive, with two files:

- `second_dataset.zip`: https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view?usp=sharing
- `SECOND_train_set.rar`: https://drive.google.com/file/d/1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc/view?usp=sharing

A scripted download returns Google Drive's "Virus scan warning" page, which needs a manual "Download anyway"
click. That click was not automated or bypassed. Manual steps:
1. Download `second_dataset.zip` in a browser, then record `shasum -a 256`.
2. Extract under `~/data/cdvqa/images/`. Check that the layout is `im1/<name>.png` and `im2/<name>.png`
   (the convention the manifests assume; it is UNVERIFIED). Check that file names match `pair_id` in the manifests.
3. Then the runner can read the images. No mirrors were used.

## Schema (from the QA JSON)

- Each split is `{images, questions, answers}`. `images` entries are question groups: several entries share one
  `file_name`. One image pair = one distinct `file_name` (pair_id = stem).
- Question to pair: `question.img_id` indexes `images` (asserted `images[i].id == i`), and that `images` entry's
  `file_name` gives the pair. Each question has exactly one answer.
- Question types (8): change_or_not, increase_or_not, decrease_or_not, smallest_change, largest_change,
  change_to_what, change_ratio, change_ratio_types.
- Pairs and QA per split:

| split | image pairs | QA |
|---|---|---|
| Train | 1600 | 65967 |
| Val | 400 | 16441 |
| Test (test1) | 968 | 39686 |
| Test2 | 968 | 31036 |

Train 1600 and Val 400 pairs match the prompt. Test and Test2 have the same 968 file names (Test2 is a
different question set over the same images). No file name overlaps between Train, Val and Test.

## Manifests (`data/manifests/cdvqa/`)

Builder: `data/cdvqa/build_manifests.py`. Seed 17. Each row: id, split, question_type, question, answer,
before_image, after_image, pair_id, question_id. Full details in `data/manifests/cdvqa/provenance.json`.

| manifest | QA | pairs | per type |
|---|---|---|---|
| train-8k.v1.jsonl | 8000 | 500 | 1000 each (8 types) |
| eval-test1-400.v1.jsonl | 400 | 99 | 50 each (8 types) |
| eval-test2-400.v1.jsonl | 400 | 100 | 50 each (8 types) |

- Train: pairs in seed-17 sha256 order, QA rows taken while their type quota is open (quota 1000).
- Eval: 100 pairs in seed-17 order per manifest, then 50 QA per type inside those pairs. test1 uses 99 pairs
  because one chosen pair contributed no selected question.
- Leakage check (asserted in the builder and in tests): 0 pairs shared between train-8k and either eval manifest.
- Answer vocabulary per type = Train answers, casefolded. All Val/Test answers are inside it.

## Majority baseline

`data/cdvqa/majority_baseline.py` → `results/cdvqa-g0/majority-baseline.json`. Majority answer per type learned
from the full Train split (65,967 QA). Scoring: strip + casefold exact match.
Macro = equal mean over 8 types. No malformed answers are possible for a constant answer.

| type | majority answer (Train) | test1 acc (n=50) | test2 acc (n=50) |
|---|---|---|---|
| change_or_not | yes | 0.4600 | 0.5400 |
| increase_or_not | no | 0.5800 | 0.6600 |
| decrease_or_not | no | 0.6800 | 0.6800 |
| smallest_change | buildings | 0.3200 | 0.2800 |
| largest_change | nvg_surface | 0.3800 | 0.5200 |
| change_to_what | nvg_surface | 0.4000 | 0.3400 |
| change_ratio | 10_to_20 | 0.1600 | 0.1800 |
| change_ratio_types | 0 | 0.5400 | 0.4200 |
| **macro** | | **0.4400** | **0.4525** |

## Runner support (repo root)

- `training/remote_sensing.py`: `TrainingExample.after_image_path` (default None). With a value, the prompt is
  [before image, after image, question]. Both images are resized to the same `image_size`. The single-image
  path is unchanged.
- `eval/cdvqa_pairs.py`: pair manifest loader (validates fields, vocabulary, existing images, unique ids),
  conditions `real` / `identical_pair` (after := before) / `swapped_order` (after, before), strict scoring
  (malformed counted, not dropped), per-type and macro summary, leakage assert.
- The GPU eval loop for pairs is **not** wired into a stage yet. Training is not included.

## Tests

- `eval/test_cdvqa_pairs.py`: 9 tests. Two-image collator with CPU doubles (answer masking PASS, both images
  in order), single-image messages unchanged, condition construction, scoring, malformed counting, loader
  rejections, leakage assert, committed manifest checks.
- `training/test_remote_sensing.py` (root, existing): 19 tests, still pass.
- Combined run at repo root: 28 passed.
- Package preflight (`pilot.py`, CPU, no images): payload, manifests, vocabulary, leakage: PASS.
- Existing B13 CPU preflight (`packages/rgb-vqa-pilot-b13-v1.1-20261017T144127Z`): PASS. That package is unchanged.

## Package

- `packages/cdvqa-g0-20261111T104959Z/` (folder committed; zip not committed)
- `packages/cdvqa-g0-20261111T104959Z.zip` sha256: `508789b93eed26b2383e411f2d8768dd31057ec4facdc295102cba176bd3c40f`
- No images, no weights, no tokenizer files.


---

# CDVQA G0b evidence

## Images: NOT available on this machine (blocking)

- Expected: `~/Downloads/second_dataset.zip` and extracted files under `~/data/cdvqa/images/`.
- Found: no `second_dataset.zip` in `~/Downloads`, no `*second*` archive or folder anywhere on the disk, and
  `~/data/cdvqa/images/` is empty. The only `.crdownload` in Downloads is an unrelated February file.
- Consequently NOT done: the archive sha256, the image layout check, the 100% pair resolve check (512x512 RGB),
  and `cdvqa-images.zip`. Once the archive exists, run the steps in the G0b report again.
- The before/after folder convention (`im1`/`im2`) is still UNVERIFIED. Evidence gathered so far:
  - SECOND project page (captain-whu.github.io/SCD): gives no folder layout and no time order.
  - SECOND paper (arXiv 2010.05687, "Semantic Change Detection with Asymmetric Siamese Networks"): defines
    two multi-temporal images `I1`, `I2`, with `I1` the "first image" and `I2` the "second image", and
    `t = 1, 2` for the two inputs. It never says which of the two is captured earlier.
  - CDVQA questions (train QA JSON): they refer to the earlier image as "pre-change image" / "pre-event image"
    and to the later one as "post-change image" (4,421 questions) and to "first image" / "second image"
    (8,573 / 6,833). Consistent with `I1` = earlier, but this does not say which folder holds it.
  - Conclusion: time order (first = earlier) is supported by the CDVQA wording; the folder-to-time mapping is
    not documented anywhere reachable. Verify it from the extracted archive (for example, the release README if
    it is in the archive) before the first GPU run.

## Train v2 (`train-8k.v2.jsonl`)

- 8,000 QA, 1,000 per type (8 types), every type filled scarcest-first.
- Pairs used: 1600 of 1600 train pairs.
- QA per pair: min 5, max 5 (cap 5).
- Leakage: 0 image pairs shared between train-8k.v2 and either eval manifest (asserted in the builder and in the
  package preflight).
- v1 (`train-8k.v1.jsonl`, 500 pairs) is kept; its hash is unchanged.
- Eval manifests (`eval-test1-400.v1`, `eval-test2-400.v1`) are unchanged (byte-identical to G0).

## Majority answers (for the result.json baseline)

`majority_answer` (Train, casefolded) is stored in `provenance.json`; the stage computes per-type accuracy on each
eval manifest from it.

## Package

- `packages/cdvqa-g0b-20261111T110559Z/` (folder committed, zip not committed)
- `packages/cdvqa-g0b-20261111T110559Z.zip` sha256: `878704207d2710bd6de036a30ab5b42e35d07b985724721fd552e61c1cf518b5`
- Contents: pilot.py, scripts/ (cdvqa_pilot.py, train_remote_sensing_adapter.py, stage1_runtime.py),
  training/remote_sensing.py, eval/ (cdvqa_pairs.py, two test files), configs/model_artifacts.json (copied from
  the B13 package), data/manifests/cdvqa/ (v1, v2, eval, provenance), pilot-config.json, requirements-target.txt
  and requirements-cpu-preflight.txt (same pins as B13), README.md, KAGGLE.md, package-manifest.json.
- No images, no weights, no tokenizer files.

## Tests (CPU doubles, python3.11)

- `eval/test_cdvqa_pilot.py` (7): two-step pair train on a fake Qwen-shaped model with the real Trainer; resume
  from a verified checkpoint; resume rejects a tampered file, a wrong identity and a wrong schedule; wrong
  selection size rejected; all pair prompt conditions; evaluate stage over four conditions on a tiny manifest with
  malformed counts and the bootstrap result shape.
- `eval/test_cdvqa_pairs.py` (9) and the root `training/test_remote_sensing.py` (19): all pass.
- Existing B13 package runner tests (`scripts/test_rgb_pilot_package.py`, `scripts/test_smoke_rgb_pilot_train.py`):
  35 pass. The B13 package preflight still passes.
- Not run: any stage that needs weights, CUDA, 4-bit loading or the real processor (`token-preflight`, `evaluate`,
  `train` on GPU).


---

# CDVQA G0c evidence (images present)

## Archives (downloaded from the SECOND Google Drive folder, linked from captain-whu.github.io/SCD)

| file | sha256 |
|---|---|
| `SECOND_train_set.rar` | `5ee2a82b5824b3f5e3c5bfaf018835862e7623993ff9ac5ce57fd11fcfad2b4e` |
| `SECOND_total_test.zip` | `bf347e52bb802a17878faccee455319a38360dab94c202d88ee6746e9d7253c9` |

The browser auto-unpacked the outer `second_dataset.zip`, so there is no hash for that outer archive. This is
recorded as a fact, not an error. Both inner archives were extracted under `~/data/cdvqa/images/`.

## Layout and mapping

- `~/data/cdvqa/images/SECOND_train_set/{im1,im2,label1,label2}`: 2968 files each.
- `~/data/cdvqa/images/test/{im1,im2,label1,label2}`: 1694 files each.
- Every CDVQA file name (Train 1600, Val 400, Test 968, Test2 968) is in `SECOND_train_set`. Zero are in `test/`.
  The `test/` folder is unused by this pilot.
- Image root for the runner: `SECOND_train_set`. Manifest paths `im1/<f>` and `im2/<f>` resolve under it.
- Resolution: train-8k.v2 1600/1600 pairs (100%), eval-test1 99/99 (100%), eval-test2 100/100 (100%).
  All 3,572 distinct referenced images are 512x512 RGB (0 failures). Checks in `data/cdvqa/verify_images.py`
  and `results/cdvqa-g0/image-check.json`.
- Checksum list over every used image: `~/data/cdvqa/SHA256SUMS-used-images.txt` (not in git),
  sha256 `c1232cd0b038b1e375d1dd300a8ef66af00dcdb6833954ec9f004b24ff739da1`, 3572 entries.

## Before/after: im1 = earlier, im2 = later

Evidence, in order of strength:

1. **Label-map test on the SECOND annotations (data, not documentation).** The paper defines the land-cover maps
   `LM1` and `LM2` for the two images `I1` and `I2`, so `label1` pairs with `im1` and `label2` with `im2`. The CDVQA
   train split has 15,367 increase/decrease questions over 1,600 pairs,
   with yes/no answers. Each question asks whether the pixel area of one land-cover class grew or shrank. For each
   orientation, we search for the colour-to-class mapping that best matches the answers:
   - `label2` is the later map: **100.00%** agreement, under one consistent mapping
     (buildings=dark_red, low vegetation=dark_green, non-vegetated ground surface=grey, playgrounds=red, trees=bright_green, water=blue).
   - `label1` is the later map: **73.35%** agreement at best.
   Reproduce with `python3.11 data/cdvqa/orientation_check.py` (result: `results/cdvqa-g0/orientation-check.json`).
   The colour mapping is recovered from the data. It is not the palette I had guessed earlier, so no palette
   assumption is used anywhere in this pilot.
2. **CDVQA wording.** The questions call the earlier image the "pre-change" or "pre-event" image, and the later one
   the "post-change" image (4,421 questions), and use "first image" and "second image" (8,573 and 6,833 questions).
3. **SECOND paper** (arXiv 2010.05687): `I1` and `I2` are "two multi-temporal images"; no explicit time order is
   stated in the text. **Project page**: no folder layout and no time order. **Archives**: no README inside either archive.

Conclusion: `im2` is the later image, `im1` the earlier. The `im1`-to-`label1` pairing is an assumption taken from the
paper's LM1/LM2 naming; the data check above supports it only indirectly (it fixes which label map is later).

## Visual sanity

`results/cdvqa-g0/sanity_pairs.png`: 8 random eval-test1 pairs (seed 17), before | after, with question and answer
underneath. Generated by `data/cdvqa/contact_sheet.py`.

## Images zip

- `~/data/cdvqa/cdvqa-images.zip` (not in git): 3572 files (im1 and im2 only, no labels),
  1,433,633,467 bytes, sha256 `15d3c896a4f3c171e352bb1e11f39c6b9878bd99b1acc88f6c3f5fbffbb32db9`.
- Root of the zip contains `im1/` and `im2/`, which is the layout the runner expects.

## Budget config (pilot-config.json)

- `ledger_total_seconds`: train 25200 (7 x 3600), evaluate 7200. Absent block = B13 default 7200.
- CLI `--ledger-total` overrides per run. Tests cover defaults, the CDVQA values, overrides and the refusal guard.

## Package

- `packages/cdvqa-g0c-20261111T114456Z/` (folder committed; zip not committed).
- Zip sha256: `4c04fbb6611e1cff16c64507c74892675409704aeadcd39dc33d192e0809156c`.
- G0b (`packages/cdvqa-g0b-20261111T110559Z/`) is kept as built. G0c supersedes it: same contents, updated budget,
  KAGGLE.md and provenance with image hashes.

## What G0b left undone (now done)

- Archive hashes and source/auto-unpack note: done (above).
- Layout and mapping check against real files: done (100% resolve, all 512x512 RGB).
- Before/after confirmation: done with data evidence (above).
- `cdvqa-images.zip`: built (above).
- Contact sheet: built.
- Package rebuilt with image layout, budget and KAGGLE.md: done as G0c.

## Tests (python3.11)

- `training`, `eval/test_rgb_pilot.py`, `eval/test_cdvqa_pairs.py`, `eval/test_cdvqa_pilot.py`, `eval/test_answer_matching.py`,
  `eval/test_baseline_rescore.py`: 72 passed.
- Existing B13 package-runner tests (`scripts/test_rgb_pilot_package.py`, `scripts/test_smoke_rgb_pilot_train.py`): 35 passed.
- B13 package preflight: hashes PASS, answer masking PASS.
- G0c package preflight (with `--image-root SECOND_train_set`): payload, manifests, leakage PASS; 8000 train pairs resolved.
- Not run: any stage needing weights, CUDA or the real processor (`token-preflight`, GPU `evaluate`, GPU `train`).


---

# CDVQA G0d: first Kaggle token-preflight failures (fixed)

## Failures

1. `token_preflight` read `plan["max_sequence_length"]` from the shipped training block, which lacked it: KeyError.
   Fixed: `training.max_sequence_length = 32768` (B13 value, same as `TrainingConfig` default) in the shipped config.
   The token check now lives in `token_check(plan, examples, processor)`, which takes the shipped plan directly.
2. Revision proof on Kaggle: Kaggle datasets drop hidden directories, so `.cache/huggingface/download/config.json.metadata`
   is absent in the dataset and the revision proof fails although the weight sha256 values match. Workaround (documented
   as a setup step in KAGGLE.md): symlink the weight files into `/kaggle/working/qwen-pinned` and restore the pinned commit into
   the metadata file. `verify_checkpoint` is **not** weakened. Each result now records `revision_proof`
   ("restored .cache/huggingface/download/config.json.metadata (pinned commit) + weight sha256 match") next to
   `base_verification.revision_source`.

## Key audit: every key read by the pair stages (preflight, token-preflight, evaluate, train)

| key | read in | present in shipped files |
|---|---|---|
| `training.max_sequence_length` | token_check | yes (added) |
| `training.image_size` | token_check | yes |
| `training.*` (15 TrainingConfig fields, `**plan`) | train_config | yes (15/15 match TrainingConfig fields) |
| `training.gradient_accumulation_steps` | train_pairs | yes |
| `base_revision` | train_pairs report | yes |
| `dependency_pins` | main (worker) | yes |
| `budget.ledger_total_seconds` (train, evaluate) | ledger_total | yes (absent block = 7200 default) |
| `providers["qwen2.5vl-3b"].revision` | verify_base, evaluate, build_pair_result | yes |
| `providers["qwen2.5vl-3b"].weight_sha256` | verify_base | yes |
| `providers["qwen2.5vl-3b"].model_id` | evaluate result | yes |
| provenance `manifest_sha256` | token/preflight hash checks | yes |
| provenance `answer_vocabulary`, `answer_choices`, `majority_answer` | score, prompt, majority baseline | yes |
| package `payload_sha256` | check_payload, binding | yes (package) |
| checkpoint `pilot-checkpoint.json` (run_identity, files_sha256, scaler_expected, step) | verify_resume | written by this code |
| `trainer_state.json` (max_steps, global_step) | verify_resume | written by Trainer |
| `config.json` `_commit_hash` or `.cache/.../config.json.metadata` | verify_checkpoint | runtime files (workaround above) |

Tests `eval/test_shipped_config.py` read the shipped files directly and run token_check, train_config and
verify_base against them. The old config fails the token test with KeyError 'max_sequence_length' (checked).

## Package

- `packages/cdvqa-g0d-20261111T193240Z/` (folder committed, zip not committed). Built by `data/cdvqa/build_package.py` from
  repo sources and `data/cdvqa/package/` templates (config, model artifacts, KAGGLE.md, README, pins, pilot.py).
- Zip sha256: `b101b0b957325221c7b83be469258e05fdd7ccde671d1370465b509480604cce`.
- run_identity (package preflight): `a755bffebc4f0fb481735e3b12913cff7efb75c088269d19cb6efac4c2dfbdef`.
- Preflight with real images: payload, manifests, leakage PASS; 8000 train pairs resolved.


---

# CDVQA G0e: first G1 crash (answer_pair undefined) fixed

- Crash: `NameError: name 'answer_pair' is not defined` in `evaluate_gpu`.
- Cause: `answer_pair` was defined just above `evaluate_gpu` in 010362e. The G0d change that split out
  `token_check` replaced everything between `token_preflight` and `evaluate_gpu`, which included `answer_pair`,
  so the function was deleted in c219587. No other copy existed (checked with `git log -S` and grep).
- Fix: `answer_pair` is restored in `scripts/cdvqa_pilot.py`. It builds the prompt with `training.remote_sensing.messages`,
  the same function the pair collator uses, so eval and training prompts match. It calls `process_vision_info`,
  decodes with the B13 settings (do_sample=False, max_new_tokens=32), decodes only the new tokens, and returns the
  stripped raw string. Normalisation and scoring stay in `eval/cdvqa_pairs.py`. The single-image B13 `_answer`
  was not reused: it cannot build the two-image prompt.
- Regression tests (`eval/test_shipped_config.py`):
  - `test_evaluate_gpu_base_only_runs_end_to_end_on_stubs` runs the real `evaluate_gpu` end to end with stub
    model/processor/vision info. Before the fix it failed with the NameError above; after the fix it passes.
  - `test_eval_prompt_equals_training_collator_prompt` checks eval and training prompts are identical for the same example.
- Pyflakes (4.1.0) on all 10 `.py` files in the G0e package: no findings. On the repo CDVQA files it reported an unused
  import (`load_choices`) and an unused `torch` in a test, both fixed; no undefined names anywhere.
- Package: `packages/cdvqa-g0e-20261111T200003Z/`, zip sha256 `90b586356b75a39bae856db19e78d3321a2f371e813c3a43d6edbe1a1a83109f`, run_identity `057f7d52d09e8497f11edc226916a828b84b6f712d778cf3e25e69e12591879a`.


## G0f evaluation-cap-only package

- Base source commit: `bc84b4c`; package built with `data/cdvqa/build_package.py::build` (prefix `cdvqa-g0f`),
  ZIP made with `shutil.make_archive` using the package folder as archive root, as in G0e.
- Source change: `scripts/cdvqa_pilot.py` LIMITS.evaluate 2400 → 7200 seconds.
- Shipped config: budget.ledger_total_seconds.evaluate 7200 → 21600 seconds. The inherited fallback for
  plans without a budget block remains 7200. Training invocation/ledger caps remain 3600/25200.
- Root shipped-config assertions updated; actual CLI reservation/supervisor tests accept 7200 and reject 7201
  before preflight. Generic ledger tests with an explicit synthetic 7200 budget remain valid and unchanged.
- Eval-cap change only vs g0e; adapter trained with g0e run_identity 057f7d52d09e8497f11edc226916a828b84b6f712d778cf3e25e69e12591879a.
- G0f evaluation-package run_identity: `47deaabb735210e86fc708d97dfb94ced4bfa81ed841f4dab2d29d320f9304cc`. This differs by design from training provenance; evaluation
  loads the supplied G0e adapter unchanged. Do not resume a G0e training checkpoint using G0f.
- Package: `packages/cdvqa-g0f-20261113T115304Z/` (folder committed; ZIP untracked, as with G0e).
- ZIP: `packages/cdvqa-g0f-20261113T115304Z.zip`, 356384 bytes, SHA256 `4d9df4f534a1093ec47ea6070846795056c73c5503c3b6b1ecf1b372336a7f1c`.
- Exact differing package files: `scripts/cdvqa_pilot.py`, `pilot-config.json`, `package-manifest.json`, `KAGGLE.md`.
  Same file inventory (22 files). The whole runner differs by only the LIMITS literal. The whole config differs
  by only the package name and eval ledger cap. All training code/config, prompts, scoring, manifests, image sizes,
  decoding, conditions, bootstrap and bundled tests are byte-identical to G0e.
- Both G0e/G0f package preflights: payload/manifests/leakage PASS; run identities printed and independently checked.
  ZIP CRC and extracted payload bytes verified against the new package tree.
- Focused `eval/test_shipped_config.py eval/test_cdvqa_pilot.py`: 20 passed. Includes the existing
  `test_eval_prompt_equals_training_collator_prompt`, unchanged, and the two new CLI boundary cases.
- Branch baseline before edits, Python 3.11, `pytest -p no:cacheprovider --ignore=packages -q`:
  1218 passed, 2 failed, 0 setup errors, 28 subtests passed.
  After: 1221 passed, 1 failed, 0 setup errors, 28 subtests passed.
  Persistent failure: `scripts/test_stage1_t4_package.py::test_updated_package_runner_matches_source` (local
  untracked historical fixture has stale source). Baseline also hit the unchanged wall-clock-sensitive
  `scripts/test_stage1_full.py::test_sharded_smoke_produces_full_accepted_throughput`; it passed afterward.
- Unqualified `pytest -p no:cacheprovider -q` aborts collection: 41 errors before / 43 after, because archived
  package tests duplicate importable module names. Exactly two new errors are the copied G0f
  `eval/test_cdvqa_pairs.py` and `eval/test_cdvqa_pilot.py`. Excluding packages runs the complete repository source
  suite rather than collecting duplicate frozen snapshot tests; no source tests were otherwise excluded.
- No real GPU evaluation, model loading, training, push or merge was performed for this cap change.
