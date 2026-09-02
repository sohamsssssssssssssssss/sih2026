# SIH26167 — Agentic Geospatial Intelligence for Multi-Sensor Satellite Data

ISRO problem statement. Smart India Hackathon 2026.

## Two rules

1. **No number enters a deck, report, or conversation unless it came out of `scripts/eval.py`.**
   A result from someone's personal notebook does not exist.
2. **No job runs on the RTX Pro 6000 without passing `scripts/smoke.py` on a Kaggle T4 first.**
   A shape mismatch crashing at step 3 costs 30 of our 150 allocated hours.

## Quickstart (Kaggle)

Settings → Accelerator → **GPU T4 x2** · Internet **On** · Persistence → **Variables and Files**

```python
!git clone https://github.com/<org>/sih26167.git /kaggle/working/sih26167
%cd /kaggle/working/sih26167
!pip install -q -r requirements.txt
!python scripts/smoke.py --config s0_baseline_qwen3b
```

Then iterate by pushing to GitHub and running `!git pull`. **Never author training
code in a notebook cell** — code that exists only in a notebook cannot be run by
the other five people on their own Kaggle accounts, which throws away 5/6 of our compute.

## Layout

```
configs/          one YAML per experiment; base.yaml holds shared defaults
scripts/          smoke.py · eval.py · train.py — all take --config
src/sih/
  config.py       config loading with `extends:` inheritance
  contract.py     the model I/O contract every track implements
  models/         backbone construction (registry.py)
  data/           loader.py (B1) · suites.py (benchmark adapters)
  eval/           metrics.py · ladder.py
results/          JSON results — committed, this is the audit trail
```

## Roles

| Code | Owner | Owns |
|---|---|---|
| A1 | | RS-VQA + captioning/grounding, **and the shared Stage 1 backbone** |
| A2 | | Bi-temporal change VQA |
| A3 | | Optical–SAR fusion |
| B1 | | Data engineering, tile format, SAR pipeline |
| B2 | | GUI, demo, docs |
| Lead | Atharva | Orchestrator, integration, eval harness |

## Dates

| Date | Gate |
|---|---|
| **8 Sept** | A3 can read a SAR scene · internal-round deck + demo video |
| **25 Sept** | **Tile format frozen** — four tracks block on this |
| **30 Sept** | Smoke-test harness live |
| **15 Oct** | Baselines done · backbone locked · **Stage 1 shared backbone complete** |
| **5 Nov** | Fusion converging, or fall back to late fusion |
| **20 Nov** | Four models beating baseline, integrated |
| **~5 Dec** | Code freeze |

## Current status

- [x] Repo scaffolded
- [ ] Benchmark HF paths verified (`src/sih/data/suites.py` — all `<VERIFY>`)
- [ ] Stage 0 baseline number recorded
- [ ] Tile format frozen (B1)
- [ ] Stage 1 shared backbone (A1)
