# CDVQA G0 package (change-detection VQA, two-image pilot)

Gate G0 only: manifests, majority baseline and runner support. No training, no GPU work,
no weights, no images in this package.

- Base model (pinned, unchanged): Qwen/Qwen2.5-VL-3B-Instruct, revision 66285546d2b821cf421d4f5eb2576359d3770cd3
- Data: CDVQA QA JSON (YZHJessica/CDVQA, Apache-2.0); images from the SECOND dataset, NOT included.
  Before/after image paths are `im1/<file>` and `im2/<file>`; this convention is UNVERIFIED until
  the SECOND archive is extracted. See `data/manifests/cdvqa/provenance.json`.
- Manifests (`data/manifests/cdvqa/`): `train-8k.v1.jsonl` (8,000 Train QA, 1,000 per type),
  `eval-test1-400.v1.jsonl` and `eval-test2-400.v1.jsonl` (50 per type, from 100 seed-17 pairs).
- Prompt order for pairs: [before image, after image, question], both resized to the same image size.
- Eval conditions: `real`, `identical_pair` (after := before), `swapped_order` (after, before).
- Scoring: exact match after strip+casefold against the type's Train answer vocabulary;
  anything else is malformed (counted, not dropped). Macro = equal mean over 8 types.

CPU checks (no weights, no images required):

    python3.11 pilot.py                 # manifest/payload preflight
    python3.11 -m pytest -q eval/test_cdvqa_pairs.py training/test_remote_sensing.py

Not in this package yet: a GPU evaluation stage for pairs and any training entry point.
