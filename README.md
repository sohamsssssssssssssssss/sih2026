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
