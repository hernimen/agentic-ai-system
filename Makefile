PORT ?= 8000

.PHONY: help dev ingest install test lint typecheck lock compose-up compose-down compose-logs compose-build compose-restart benchmark

help:
	@echo "Targets:"
	@echo "  make install         - uv sync (creates .venv, installs from uv.lock)"
	@echo "  make lock            - Regenerate uv.lock"
	@echo "  make ingest          - Build/refresh the Chroma index from data/sops and data/equipment_manuals"
	@echo "  make dev             - Run API locally with uvicorn (reload)"
	@echo "  make test            - Run pytest"
	@echo "  make lint            - Run ruff"
	@echo "  make typecheck       - Run mypy"
	@echo "  make compose-build   - Build Docker Compose images"
	@echo "  make compose-up      - Start all services with Docker Compose"
	@echo "  make compose-down    - Stop all Docker Compose services"
	@echo "  make compose-logs    - Follow API logs in Docker Compose"
	@echo "  make compose-restart - Rebuild and restart all services"

install:
	uv sync --group groq

ingest:
	uv run python -m lims_agent.rag.ingest

dev:
	uv run uvicorn lims_agent.api.main:app --reload --host 0.0.0.0 --port $(PORT)

test:
	uv run pytest

benchmark:
	uv run python -c "from lims_agent.eval.dataset import build_default_benchmark_cases; from lims_agent.eval.benchmark import check_benchmark_thresholds; check_benchmark_thresholds(build_default_benchmark_cases(), min_pass_rate=0.8, min_grounded_rate=0.8, min_retrieval_hit_rate=0.8)"

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy

lock:
	uv lock

compose-build:
	docker compose build

compose-up:
	docker compose up -d
	@echo "Services starting..."
	@echo "API: http://127.0.0.1:8000/docs"
	@echo "Ollama: http://127.0.0.1:11434"

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f api

compose-restart: compose-down compose-build compose-up
