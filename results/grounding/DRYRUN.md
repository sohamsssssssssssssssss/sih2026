# Grounding v1 dry run

The CPU stub used the checksum-verified baseline ZIP and processed 20 source items. It used five grounding evaluation items and three DINO selection items per dataset, plus four VRSBench caption items. The runner produced eight prediction JSONL files and a checksummed run receipt. The scorer verified the output checksums and wrote results.json and REPORT.md. The caption path used pycocoevalcap 1.2 and pycocotools 2.0.8 with Java. Install the metric pins with `python -m pip install pycocoevalcap==1.2 pycocotools==2.0.8`.

The stub exercised the data and reporting paths. Its synthetic predictions carry no evidence about model quality.

## T4 runtime estimate

The full run has 6,000 Qwen grounding decodes capped at 64 tokens and 1,000 caption decodes capped at 256 tokens. Planning assumptions are 1.5–3 seconds per grounding item and 4–8 seconds per caption. DINO does 7,000 forward passes, reusing each pass across the 12 threshold pairs, with a planning allowance of 0.2–0.6 seconds per item including postprocessing. Allow another 0.5 hour for provisioning, checksums and output. The estimate is about 4.5–9 hours on one T4. These are planning assumptions and there is no measured T4 throughput for this runner yet.

Use a committed Kaggle run with Save & Run All and one T4. Bash-cell output becomes visible when the cell ends.

## ZIP and extracted pack parity

Each layout ran the same 20-item CPU stub through the runner and scorer. All eight prediction JSONLs, run-info.json, SHA256SUMS, results.json, and REPORT.md matched byte for byte between layouts. Each raw output directory contained 42,810 bytes across its files. Its SHA256SUMS digest was `04b272c5a4faabef5e7e81fee246154c3f6a599322adc10e7755d5425f0c61d9` for this pre-commit dry run. These outputs exercise the pipeline and carry no model-quality claims.

The cell accepts a single intact ZIP or a single extracted pack with the pinned root MANIFEST.json. Both layouts enforce the exact file set, byte lengths, and SHA-256s. The extracted layout ignores no metadata files. The Qwen check accepts the flat and owner-qualified Kaggle dataset mounts and verifies the exact model file inventory and hashes. The pack inputs stay under /kaggle/input. The checkout, environment, and model caches use /tmp. GitHub, package installation, and Hugging Face provisioning require internet before offline inference. Results go to a new /kaggle/working/grounding-baselines directory, with output checksums and a 200 MB size check.
