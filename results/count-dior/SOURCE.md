# Source of the DIOR val count evidence

| Item | Value |
|---|---|
| Run date | 15 Nov 2026 |
| Kaggle notebook | `notebooka889c4bac0`, committed version (Save & Run All), GPU T4 x2, cell `kaggle/run_count_dior_gpu.txt` ended with `CELL DONE` |
| Code | commit `ed13580` on `feature/count-grounding`, uploaded as the Kaggle dataset `satquery-count-grounding-zip`; zip sha256 `ac804c5f55e1f0f60945e2b0c0f3b9da761a04d32f0cf8dabfb0916da8ac3902` |
| DIOR mirror | https://zenodo.org/records/11213149/files/DIOR-VOC.zip (third-party re-upload), 7,423,852,627 bytes, sha256 `7a386a56ba815333627d89d3b8b372d1af2a07b78955c017530ec62b7f1e0540` (pinned, re-verified by the evaluator) |
| Detector checkpoint | `ShilongLiu/GroundingDINO/groundingdino_swint_ogc.pth`, sha256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| Text encoder | `bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594`; this was identical in two separate runs |
| Packages | `groundingdino-py==0.4.0`, `transformers==4.49.0` |

## Evidence files (sha256)

| File | sha256 |
|---|---|
| `dior-count-val-evaluation.json` | `f77636624bd4824404b327f1168e40a3a450f74e7ca59a107fb3fb6375387df2` |
| `dior-count-val-seed17-1000.json` (frozen sample manifest) | `5d3abb80bab482ff9da6438f54567dfd54503d7738663cbca6caec64fd742634` |
| `count-dior-provisioning.json` | `34aa118f25c4c444b909d83c5aa519b9d2e7a88a79e6585c2a7f90642e8038d8` |

These cross-checks hold:
- The evaluation's `sample_manifest_sha256` equals the manifest file's sha256.
- Its `source.zip_sha256` equals the pinned mirror sha256.
- Its `detector.checkpoint_sha256` equals the provisioning `grounding_checkpoint_sha256`.

The checkpoint sha256 and BERT revision are recorded here only. `configs/model_artifacts.json` still has `sha256: null` and has no field for the text encoder, so they are not yet enforced at runtime.

## Claim status

**Allowed:** "On a frozen 1,000-pair DIOR val sample (500 present / 500 absent, 50 per class, seed 17, third-party mirror pinned by sha256), Grounding DINO Swin-T at box threshold 0.35: exact 32.4%, MAE 1.53, within ±1 80.4%; always-0 baseline exact 50.0%, MAE 1.48. On present-class pairs: exact 49.2%. Absent-class pairs: false count on 84%."

**Not allowed:**
- Any claim that detector counting beats a trivial baseline.
- Any Cartosat or RISAT claim.
- Any claim about the resolution gate. It was not tested, because DIOR GSD is not in the VOC annotations.

This is a single seed, on val only.
