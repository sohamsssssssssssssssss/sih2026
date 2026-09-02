# Configs

One YAML per experiment. Every run is a file in git — that is what lets six
people on six Kaggle accounts run each other's experiments.

Naming: `s<stage>_<track>_<variant>.yaml` — e.g. `s2c_change_lora_r32.yaml`.

Every config `extends: base`. Never edit `base.yaml` to make one experiment
work; override in the child.

## Stage map

| Stage | Config prefix | Owner | Depends on |
|---|---|---|---|
| 0 — baseline | `s0_` | A1 | — |
| 1 — RS adaptation (shared) | `s1_` | A1 | Stage 0 |
| 2a — VQA | `s2a_` | A1 | **Stage 1** |
| 2b — grounding | `s2b_` | A1 | **Stage 1** |
| 2c — change VQA | `s2c_` | A2 | **Stage 1** |
| 2d — SAR encoder | `s2d_` | A3 | — |
| 2e — fusion | `s2e_` | A3 | Stage 1 + 2d |
| 3 — high-res adaptation | `s3_` | all | Stage 2 |
| 4 — calibration | `s4_` | Lead | Stage 3 |

Stage 1 blocks four tracks. It is the critical path.
