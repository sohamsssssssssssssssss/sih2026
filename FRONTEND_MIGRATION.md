# SatQuery AI frontend migration

This branch adds a local FastAPI + Next.js interface without changing the
existing Streamlit fallback or any model, orchestration, evaluation, data, or
committed result source.

## Run locally

Terminal 1:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The original fallback remains available with
`streamlit run demo_gui/app.py`.

## Implemented checkpoints

- A: responsive command-center shell and scene workspace
- B: Resolution Lab driven by the committed ladder artifact
- C: SAR Validation driven by the committed analyst annotation
- D: golden Qwen analysis with explicit live/cached provenance and safe no-GPU fallback
- E (partial, intentionally lower priority): process-local execution history and trace verification are included; uploads remain visibly marked in development

The golden/cached path is runtime-offline. Live inference can use local cached
weights when CUDA is present. Neither path initiates an internet download.
