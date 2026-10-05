.PHONY: api frontend test docker-cpu docker-gpu deploy-smoke docker-down

api:
	backend/.venv/bin/uvicorn backend.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	python3 -m pytest backend/test_api.py -q
	cd frontend && npm run build

# --- Docker deployment (docs/deploy.md) -------------------------------------
# One public URL through Caddy: /api/* -> backend, everything else -> frontend.
COMPOSE_CPU = docker compose -f docker-compose.yml
COMPOSE_GPU = docker compose -f docker-compose.yml -f docker-compose.gpu.yml
# deploy-smoke checks the stack started by docker-cpu (cpu) or docker-gpu (gpu).
SMOKE_MODE ?= gpu

docker-cpu:
	$(COMPOSE_CPU) up -d --build

docker-gpu:
	$(COMPOSE_GPU) up -d --build

deploy-smoke:
	deploy/smoke.sh $(SMOKE_MODE)

docker-down:
	$(COMPOSE_GPU) down
