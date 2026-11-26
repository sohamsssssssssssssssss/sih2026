# Grounding v1 dry run

The CPU stub used the checksum-verified baseline ZIP and processed 20 source items. It used five grounding evaluation items and three DINO selection items per dataset, plus four VRSBench caption items. The runner produced eight prediction JSONL files and a checksummed run receipt. The scorer verified the output checksums and wrote results.json and REPORT.md. The caption path used pycocoevalcap 1.2 and pycocotools 2.0.8 with Java. Install the metric pins with `python -m pip install pycocoevalcap==1.2 pycocotools==2.0.8`.

The stub exercised the data and reporting paths. Its synthetic predictions carry no evidence about model quality.

## T4 runtime estimate

The full run has 6,000 Qwen grounding decodes capped at 64 tokens and 1,000 caption decodes capped at 256 tokens. Planning assumptions are 1.5–3 seconds per grounding item and 4–8 seconds per caption. DINO does 7,000 forward passes, reusing each pass across the 12 threshold pairs, with a planning allowance of 0.2–0.6 seconds per item including postprocessing. Allow another 0.5 hour for provisioning, checksums and output. The estimate is about 4.5–9 hours on one T4. These are planning assumptions and there is no measured T4 throughput for this runner yet.

Use a committed Kaggle run with Save & Run All and one T4. Bash-cell output becomes visible when the cell ends.

## ZIP and extracted pack parity

Each layout ran the same 20-item CPU stub through the runner and scorer. All eight prediction JSONLs, run-info.json, SHA256SUMS, results.json, and REPORT.md matched byte for byte between layouts. Each raw output directory contained 42,810 bytes across its files. Its SHA256SUMS digest was `04b272c5a4faabef5e7e81fee246154c3f6a599322adc10e7755d5425f0c61d9` for this pre-commit dry run. These outputs exercise the pipeline and carry no model-quality claims.

The cell accepts a single intact ZIP or a single extracted pack with the pinned root MANIFEST.json. Both layouts enforce the exact file set, byte lengths, and SHA-256s. The extracted layout ignores no metadata files. The Qwen check accepts the flat and owner-qualified Kaggle dataset mounts and verifies the exact model file inventory and hashes. The pack inputs stay under /kaggle/input. The checkout, environment, and model caches use /tmp. GitHub, package installation, and Hugging Face provisioning require internet before offline inference. Results go to a new /kaggle/working/grounding-baselines directory, with output checksums and a 200 MB size check.

## DINO fail-fast and Qwen reuse

Installed `groundingdino-py==0.4.0` into a throwaway venv. The PyPI source archive SHA-256 was `23de56af498b7cdcc024eb1062fc3179e3cc4dacd92adbfb625097065377700f`. Inspected the installed `groundingdino.util.inference.predict` body and ran it on CPU with the real Swin-T Grounding DINO class and randomly initialized weights. The test used two 320×320 synthetic images, a local BERT vocabulary and a randomly initialized one-layer BERT to avoid fetching pretrained weights. The 24 calls to `predict` exercised every grid pair on each image and a forward hook counted exactly two model forwards. Box, score and phrase lengths matched and the cached box tensors were finite.

`predict` calls `model.to(device)`, invokes `model(image[None], captions=[caption])` and reads `model.tokenizer`. The wrapper's `to` keeps the cache in place after the provider places the model on the inference device. Its `__call__` forwards the image and keyword arguments once. Its `__getattr__` exposes the real tokenizer, including the tokenizer's `decode` used for phrase extraction. The real model's forward uses `max_text_len` and `specical_tokens`. Both are also accessible through delegation. Package loading calls `eval()` on the real model before wrapping. The pinned `predict` body makes no direct call to `eval`.

In real baseline mode the runner loads DINO and checks two eval rows through image transforms, the complete grid and JSON serialization before loading Qwen. Qwen runs one eval row before any condition file is written or reused. A regression test injects the original tokenizer error and observes a hard stop with zero Qwen calls and no output directory. Its successful path checks preflight ordering, progress at DINO row 25, reuse provenance and the reused bytes. DINO writes each condition incrementally and renames it from `.jsonl.partial` only on completion. Output is checked against the 200 MB budget before each write and again before the receipt and checksum list.

The verified real baseline ZIP completed the 20-source-item stub with both preflights and 42,893 bytes of raw output plus receipts in the output directory. All nine listed checksums passed. All eight prediction JSONLs matched the producer's stub outputs byte for byte. The local Python suite finished with 1,457 passed and 48 skipped, including all eight new focused tests.

Both Kaggle cells require a single T4. The DINO-only cell still needs Qwen's model input for the one-row preflight. Execution of the four bulk Qwen conditions is replaced by byte-verified copies from the third input after all four hashes are filled. At the existing 0.2–0.6 seconds per DINO item and a 0.5-hour provisioning allowance, estimate **0.9–1.7 hours** DINO-only and **4.5–9 hours** for the full run. These are planning estimates. The production GPU path and full-run output size have yet to be measured.
