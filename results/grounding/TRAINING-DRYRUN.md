# Grounding adapter CPU preflight

The private train ZIP has 7,500 images and 10,000 rows. The build checked the frozen inputs and found zero training overlaps with eval or select by parent ID, qualified image path, or image SHA-256. The image payload before compression was 2,696,238,619 bytes.

The local pinned Qwen processor formatted eight grounding and eight caption examples. All 16 passed the assistant-token boundary and loss-mask assertions. A synthetic checkpoint receipt loaded once and rejected a deliberately changed state file. This exercised receipt loading only. No optimizer or model weights were used on CPU.

The prediction stubs processed 20 selected inputs. The adapter produced all three grounding conditions on ten eval expressions and four caption predictions. The scorer verified the output checksums and produced both report files in smoke mode without issuing a gate verdict.

## T4 planning estimate

At 2–5 seconds per training microbatch, 10,000 microbatches take 5.6–13.9 hours before setup and checkpoint I/O. Budget about 6–15 T4 hours and chain sessions if needed. At 1.5–3 seconds per grounding generation and 4–8 seconds per caption, 6,000 grounding and 1,000 caption generations take 3.6–7.2 hours. Budget about 4–8 T4 hours for eval. These are planning assumptions. GPU timing and optimizer-state resume remain to be verified on Kaggle.

## Kaggle handoff

Use Save & Run All with one visible T4 and Internet enabled for Git and pip. Attach private datasets for the pinned Qwen snapshot and the train pack for training, or the baseline pack for adapter eval. The pack can be an intact ZIP or the Kaggle-extracted tree.

For a continuation, publish the sealed checkpoint directory as a private dataset and set SATQUERY_RESUME_RECEIPT_SHA256 to the printed ARTIFACT.json SHA-256. For eval, attach the sealed final adapter directory privately and set SATQUERY_ADAPTER_RECEIPT_SHA256 to its printed receipt SHA-256. Both cells verify the mounted directory against its receipt before use.
