# sih26167

Contract-first scaffold for a multi-agent geospatial visual-question-answering system on Python 3.11 (recorded in `.python-version`). Install with `pip3 install -r requirements.txt`; if `rasterio` fails to install on macOS, omit it for the MVP demo because none of the scaffolded paths require it.

## Folders and invariants

`configs/` contains one YAML file per experiment. Every config must keep the common fields in `example.yaml` so training and smoke-test entry points remain interchangeable.

`data/` owns loading and the canonical tile schema. Every loaded sample must include non-null `gsd` and `sensor`, and arrays must preserve their specified float32 channel-first shapes.

`models/` owns the abstract inference contract and all implementations. Models may change internally, but `infer(image_paths: list[str], question: str)` must always return `answer`, `confidence`, and `evidence` with the documented types.

`orchestrator/` owns registration, routing, and hash-chained traces. It must communicate with models only through `Model.infer()` and must never inspect implementation internals; every routed call must append a trace record.

`eval/` owns authoritative metrics, smoke verification, and suite loader stubs. Reported evaluation numbers are valid only when produced by `eval/eval.py`, using exact case-insensitive stripped answer matching.

`scripts/` owns configuration-driven training entry points. It may select a registered model by name, but must not embed model-specific training logic or reach into model internals.

`demo_gui/` owns the Phase 0 Streamlit shell. It must call the router—not a model directly—so real models can replace the mock without changing the UI contract.

## Commands

```bash
python3 eval/smoke.py --config configs/example.yaml
python3 eval/eval.py --model mock --suite proxy --out results.json
python3 scripts/train.py --config configs/example.yaml
streamlit run demo_gui/app.py --server.headless true
```

## Resolution ladder

The ladder uses real LoveDA pixels and semantic masks; only spatial resolution and sensor noise are simulated. LoveDA is licensed for academic, non-commercial use. Download and extract Train+Val idempotently with `python3 data/download_ladder_data.py`, then generate up to 200 real source images at five rungs with `python3 eval/ladder.py --limit 200`. The degradation applies a Gaussian PSF, area-average decimation, and read plus shot noise while keeping `sensor: loveda`, never `synthetic`. Evaluate with `python3 eval/eval.py --model mock --suite ladder --out results.json` and plot with `python3 eval/plot_ladder.py results.json`. DOTA acquisition can be attempted with `python3 data/download_ladder_data.py --dataset dota`; Google Drive failures return promptly with manual-download instructions.

## Baseline

Stage 0, measured. Every later stage is judged against these numbers.

| Metric | Value |
|---|---|
| accuracy | 0.5131 |
| binary_accuracy | 0.6655 |
| **open_accuracy** | **0.1651** |
| pred_yes_rate_on_binary | 0.3635 |
| n | 10004 |
| binary_n / open_n | 6957 / 3047 |

- **Model:** frozen `Qwen/Qwen2.5-VL-3B-Instruct`, fp16, greedy decoding, no training or fine-tuning.
- **Split:** official RSVQA-LR test split, all 10,004 active test questions (`--full`).
- **Degeneracy check:** passed, no warning. A yes-rate of 0.3635 is well under the 0.85 threshold, so this is a real measurement rather than an always-yes artefact.
- **Result:** `results/qwen2.5vl-3b__rsvqa__20260903T175900Z.json`

**`open_accuracy` is the headline metric for this project.** Aggregate accuracy is dominated by the 6957 binary questions, where a coin-flip already scores near 0.5; it moves even when the model has learned nothing about the imagery. The open-ended stratum is what the problem statement actually asks for, and 0.1651 is the number Stages 1-3 have to beat. Report it alongside the aggregate, never instead of it.

### Sample-size warning

`binary_accuracy` is stable under small `--limit` runs. `open_accuracy` is **not**.

The split is 6957 binary to 3047 open-ended, so `--limit` draws roughly 70/30 in favour of binary and the open-ended stratum stays small: `--limit 200` lands on only ~61 open-ended questions. Same config, same matcher, two runs:

| Metric | `--limit 200` | full split (n=10004) |
|---|---|---|
| binary_accuracy | 0.699 | 0.6655 |
| open_accuracy | 0.4227 | 0.1651 |

`binary_accuracy` moved 3 points. `open_accuracy` moved by a factor of 2.5 — and the small-sample figure was optimistic, in the direction that flatters us.

**Do not use `--limit` below ~2000 to compare open-ended performance between models or checkpoints** (~2000 draws ~609 open-ended questions). Headline numbers use `--full`. `eval/eval.py` prints a warning before inference starts when a run would land on fewer than 500 open-ended questions, and records it as `sampling_warning` in the results JSON.

## Frozen Qwen RSVQA-LR baseline

`qwen2.5vl-3b` wraps the frozen `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint with lazy loading, greedy decoding, and no training or fine-tuning. `python3 scripts/dry_run_rsvqa.py` downloads the complete official RSVQA-LR release and validates two real samples plus a mocked forward call without loading model weights. The eval CLI uses the official test split, defaults to 200 deterministically spaced samples across the split, accepts `--limit N`, and uses all 10,004 active test questions with `--full`.

Push this repository to a public GitHub repository before using Kaggle, or upload it as a Kaggle dataset and copy it into `/kaggle/working/sih26167`. In a new Kaggle notebook with a T4 accelerator and Internet enabled, use these cells in order, replacing the repository URL:

```python
# Cell 1: fail before downloads if the T4 is not attached.
import torch
assert torch.cuda.is_available(), "Select a T4 accelerator in Kaggle settings"
print(torch.__version__, torch.cuda.get_device_name(0))
```

```python
# Cell 2: install Qwen2.5-VL runtime dependencies.
%pip install -q "transformers>=4.49" qwen-vl-utils accelerate
```

```python
# Cell 3: clone the pushed repository.
REPO_URL = "https://github.com/YOUR_USERNAME/sih26167.git"
!git clone "$REPO_URL" /kaggle/working/sih26167
%cd /kaggle/working/sih26167
```

If GitHub access is unavailable, replace Cell 3 with `!cp -R /kaggle/input/YOUR_DATASET_SLUG/sih26167 /kaggle/working/sih26167` followed by `%cd /kaggle/working/sih26167`.

```python
# Cell 4: download RSVQA-LR and run the 200-sample frozen baseline.
!python3 eval/eval.py --model qwen2.5vl-3b --suite rsvqa --out /kaggle/working/results.json
```

```python
# Cell 5: print the metric; results.json remains a downloadable Kaggle output artifact.
import json
with open("/kaggle/working/results.json") as handle:
    report = json.load(handle)
print("accuracy:", report["accuracy"], "n_samples:", report["n_samples"])
```

## SAR reading gate

The SAR gate submits five recent dual-polarization Sentinel-1 GRD scenes over diverse Indian landscapes for HyP3 gamma-0 RTC processing, then creates fixed-scale VV/VH/VV−VH false-color quicklooks for manual interpretation. Create a free [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new), link it in [ASF Vertex](https://search.asf.alaska.edu/), configure an Earthdata entry in `~/.netrc`, and run `pip3 install asf_search hyp3_sdk rasterio`. Run `python3 data/sar_gate/order_scenes.py`, wait for the jobs to succeed, then run `python3 data/sar_gate/process_scenes.py`. Rasterio is required for this gate even though the earlier MVP demo can run without it. Raw products, job IDs, and renders are intentionally ignored; only scripts and the blank manual annotation materials are versioned.
