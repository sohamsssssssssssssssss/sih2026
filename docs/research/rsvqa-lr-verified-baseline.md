# RSVQA-LR Verified Baseline

This note records the verified RSVQA-LR dataset state used by SatQuery before VLM adaptation experiments.

## Dataset

The official RSVQA-LR data was validated through SatQuery's manifest preparation pipeline.

Verified active records:

- Train: 57,223 QA records across 572 images
- Validation: 10,005 QA records across 100 images
- Test: 10,004 QA records across 100 images
- Total: 77,232 QA records across 772 unique images

No image-ID overlap was detected between the train, validation, and test splits.

All 772 referenced TIFF images were successfully opened and verified by the repository validation pipeline.

## Official Placeholder Handling

The official dataset contains inactive question placeholders that may omit fields such as `img_id`.

SatQuery now skips inactive questions before accessing active-record fields while continuing to fail closed for malformed active records.

## Verified Manifest

Validated records:

77,232

Manifest SHA-256:

23a74c573026a26620262eb2440a7effa10168f77010f40dee81a77996fed3e1

## Scope

This verification establishes dataset integrity and readiness for the next VLM adaptation experiments.

It does not claim improved VQA accuracy or model performance. Those claims require held-out base-versus-adapter evaluation.
