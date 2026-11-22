# Mac VQA fresh test gate

Start at `f598393866034ec097e18a322e21e599e2aa41d7`. Parity [attempt 1](VQA-MAC-PARITY.md) (`705e505`, [results](../../results/vqa-mac-parity/REPORT.md) `3522388`) and [attempt 2](VQA-MAC-PARITY-ATTEMPT2.md) (`3b5c02e`, [results](../../results/vqa-mac-parity/attempt2/REPORT.md) `f598393`) both remain **FAIL**. The unshipped base model's malformed count was a poor release criterion. Small precision changes flipped near-ties, taking CUDA's four malformed base answers to five on 8-bit and seven on bf16. B13 had zero in both runs. The bf16 Mac demo also had 1.81 GiB of system headroom against a 3 GiB requirement. This evaluation asks whether the proposed 8-bit B13 release works on official RSVQA-LR test questions that did not enter B13 training or validation.

## Frozen data and procedure

- [Official RSVQA-LR release, Zenodo 6344334](https://zenodo.org/records/6344334). Verify local train/validation/test annotation files and the RGB archive against the release checksums pinned in `eval/suites/rsvqa.py`. [Source SHA-256s and train majority labels](../../data/manifests/rsvqa/mac-fresh-test/sources.json) SHA-256 `803f3822ecd551fa72ee6c512256f9c6e01b47e3e53330ba2ba06bf9549ac204`.
- [Question manifest](../../data/manifests/rsvqa/mac-fresh-test/questions.jsonl) SHA-256 `0ffd6d711f3b3061bcd4485671532ba689d18655f02b88a8d9e5efee94f430f6`. The 100 official test image IDs are disjoint from active train and validation memberships. In ascending image-ID order, draw one active question per type from question-ID-sorted candidates with one Python RNG seeded at 17, in presence, comparison, count, rural/urban order. Each image belongs to exactly one cluster and contributes four questions. The manifest has 400 rows, 100 per type, and includes the per-image SHA-256s verified from the release archive. The majority answer for each type is the mode of normalized answers on all eligible active train rows, with lexical tie-breaking.
- Compare 8-bit MLX base on real imagery, 8-bit B13 on real imagery, and 8-bit B13 on a constant RGB (127,127,127) image at the original dimensions. Pin the source revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, SHA-256 `e4ed41344ad53c218047253edf61d0e60a2c3d151651977ef8aff18094c99e6f` for the 8-bit conversion MANIFEST, and adapter SHA-256 `7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a`. Use the separate MLX venv. Use the same prompt, 392×392 RGB preprocessing, greedy 32-token limit, repetition penalty 1.05 and explicit context 4096, and stripped case-insensitive exact matcher with format validation from the parity runs.
- Report each type and equal-weight macro. Use 2,000 paired bootstrap draws over the 100 image clusters with NumPy seed 17 and percentile 95% intervals. Reuse each draw across all four conditions including the train-majority baseline. Gated contrasts use macro accuracy. Report count accuracy, per-type results and comparison with validation as context.

## Release rule

Require all six checks:

| Gate | Pass criterion |
|---|---|
| G1 | Lower 95% CI bound for B13 minus base macro is strictly positive. |
| G2 | Lower 95% CI bound for B13 minus train majority macro is strictly positive. |
| G3 | Lower 95% CI bound for B13 real minus B13 grey macro is strictly positive. |
| G4 | B13 real has at most 4 malformed answers out of 400. |
| G5 | B13 real has median latency ≤10 s. No crashes in the three evaluation runs or in the soak. Run 30 questions on a single already-loaded worker. Its peak growth over the start of the soak is ≤1 GiB both for process RSS and MLX active plus cached bytes. |
| G6 | Under the demo setup below, the minimum available system memory during the soak is ≥3 GiB with both demo servers responding and the 8-bit B13 process serving all 30 questions. |

### Demo setup and soak

Quit IDEs, other agent sessions and extra applications. Keep a terminal, one browser with the demo open, the backend and frontend started with `bash scripts/run-demo.sh`, and the separate B13 inference process serving the soak. OS services and the controlling terminal session are expected. Check ports 3000 and 8000 before launching. Record the application inventory. An unverified setup cannot pass G6. Questions use the frozen manifest's mixed 30-question seed-17 selection (8 presence, 8 comparison, 7 count, 7 rural/urban) from the existing soak helper. Take a system snapshot before model load and sample `vm_stat` and swap every 0.25 seconds during the soak. As in the prior gate, available memory is (free + inactive + speculative pages) × page size. Check demo endpoints before and after. The gate worker uses the separate venv and the same model and inference function before any app wiring.

Record results, evaluation commit and all input hashes in `results/vqa-mac-fresh-test/`. Keep these criteria fixed after observing results.
