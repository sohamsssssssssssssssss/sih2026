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

## Data

**Primary dataset: BigEarthNet.txt** (arXiv 2603.29630, CC-BY-4.0, distribution
at `txt.bigearth.net`). 464,044 co-registered Sentinel-1 SAR + Sentinel-2
multispectral pairs carrying 9.6M annotations in three types — captions
(LULC classes, spatial relations, environmental context), VQA pairs, and
referring-expression detection instructions with bounding boxes.

One ingestion, four views: `data.ann_type` filters the same shard set into a
captioning view (Stage 1), a VQA view (Stage 2a), and a grounding view
(Stage 2b). See the schema v2 docstring in `src/sih/data/loader.py`.

Two rules that are easy to get wrong:

- **Subsample to ~50k, stratified** on annotation type, geography, LULC class
  and season. 464k will not fit Kaggle's ~73 GB scratch, and random sampling
  drops the rare complex LULC classes — the ones the paper reports VLMs
  failing on. A random 50k flatters our own numbers by deleting the hard cases.
- **The manually-verified benchmark split is never subsampled and never
  trained on.** It is the only clean measurement surface we have.

**A2 has no PS-provided data source.** BEN.txt is single-epoch, so it carries
nothing bi-temporal; A2's change-VQA track needs a dataset that the problem
statement does not supply, and CDVQA has no mirror on the HF Hub. This blocks
A2's track, not just one suite — it needs resolving well before the 15 Oct
baseline gate.

## Current status

- [x] Repo scaffolded
- [x] Benchmark HF paths verified (`src/sih/data/suites.py`) — rsvqa, hrvqa live;
      vrsbench needs its 3.98 GB image zip; lrsvqa blocked on a 57 GB `.7z`
- [ ] BEN.txt path resolved (`bentxt` is still `<VERIFY-txt.bigearth.net>`)
- [ ] BEN.txt subsampled to ~50k, stratified
- [ ] Stage 0 baseline number recorded
- [ ] Tile format frozen (B1) — schema v2 drafted, `build_loader` still a stub
- [ ] Stage 1 shared backbone (A1)
- [ ] **A2 unblocked** — no bi-temporal source identified
