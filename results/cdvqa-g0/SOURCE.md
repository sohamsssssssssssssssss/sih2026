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
