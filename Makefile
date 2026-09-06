.PHONY: api frontend test

api:
	backend/.venv/bin/uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	python3 -m pytest backend/test_api.py -q
	cd frontend && npm run build
