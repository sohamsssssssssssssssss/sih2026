# DIOR-RSVG Grounding DINO artifact search

The original Kaggle T4 zero-shot result was found in Git history. A later Mac threshold study was also found. The separate Kaggle T4 threshold-sweep output is **lost in the searched locations**.

| Artifact | Git path or expected output | Commit | SHA-256 |
|---|---|---|---|
| Original Kaggle T4 zero-shot result | `results/grounding-dino-swint__dior-rsvg__20260911T084941Z.json` | `4d43caaaba248ba78060d2eaa0d9d40f687fb1e5` | `2e5b7b43fb0fbc50233336ca2acd7b8e9c9bf041465af722e67431e9d216c136` |
| Later CPU/MPS reproduction and threshold results | `eval/results/grounding/grounding-study-v1.json` | `3be90daa235ab83b7768a16128d2b1619bff84f1` | `80b07e27f50c6bf1ea4b5c044931d367ceb28076c2c9222d5b08263344688239` |
| Later study write-up | `docs/research/grounding-reproduction-and-threshold-study.md` | `3be90daa235ab83b7768a16128d2b1619bff84f1` | `38f6b2959f374cbf67c3b32e94751fb722865bd4b95b55ec25640738fe7fedb0` |
| DIOR-RSVG dataset audit | `eval/results/grounding/dior-rsvg-dataset-audit.json` | `3be90daa235ab83b7768a16128d2b1619bff84f1` | `3eacb908c65e50a985b47749e192c22f9827780ac29e364bf0f6a81cc8fae6a3` |
| Original Kaggle sweep | `/kaggle/working/results/grounding-dino-swint__dior-rsvg__threshold-sweep__<timestamp>.json` | Runner at `6d09a60` | Unavailable |

The recovered Kaggle result records run source `14e8c14a211eb14807678961d1222f396c7674f1`, Acc@0.5 of 65/400 (0.1625), and mean IoU 0.186203 at box/text thresholds 0.35/0.25. Its internal run ID ends in `085257Z`, which differs from the committed filename. The Mac study records validation-only threshold selection and a subsequent test readout. Both are historical evidence for the earlier protocol.

The search covered the `~/sih2026-*` directories, all locally available Git refs, `~/Downloads` (including the numbered results directories and grounding archive), `~/ben-work`, and project notes in the repo, Downloads, and Documents. `SATQUERY_CONTEXT.md` in Downloads records the sweep setup. The tracked `sih26167.zip` at `6d09a60` contains runner and test code. No evaluations were rerun.

Git blobs were hashed directly with `git show <commit>:<path> | shasum -a 256`.
