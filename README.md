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
