# SatQuery AI API

Run from the repository root so the API imports the existing `models/` and
`orchestrator/` packages without moving or duplicating them:

```bash
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.main:app --reload --port 8000
```

The service is offline-first. It sets Hugging Face and Transformers offline
flags before importing the model registry. Live Qwen inference is attempted
only when local pixels and model dependencies are available; an exact committed
result is otherwise returned with explicit cached provenance.
