# Grounding v1 dry run

The CPU stub used the checksum-verified baseline ZIP and processed 20 source items. It used five grounding evaluation items and three DINO selection items per dataset, plus four VRSBench caption items. The runner produced eight prediction JSONL files and a checksummed run receipt. The scorer verified the output checksums and wrote results.json and REPORT.md. The caption path used pycocoevalcap 1.2 and pycocotools 2.0.8 with Java. Install the metric pins with `python -m pip install pycocoevalcap==1.2 pycocotools==2.0.8`.

The stub exercised the data and reporting paths. Its synthetic predictions carry no evidence about model quality.

## T4 runtime estimate

The full run has 6,000 Qwen grounding decodes capped at 64 tokens and 1,000 caption decodes capped at 256 tokens. Planning assumptions are 1.5–3 seconds per grounding item and 4–8 seconds per caption. DINO does 7,000 forward passes, reusing each pass across the 12 threshold pairs, with a planning allowance of 0.2–0.6 seconds per item including postprocessing. Allow another 0.5 hour for provisioning, checksums and output. The estimate is about 4.5–9 hours on one T4. These are planning assumptions and there is no measured T4 throughput for this runner yet.

Use a committed Kaggle run with Save & Run All and one T4. Bash-cell output becomes visible when the cell ends.
