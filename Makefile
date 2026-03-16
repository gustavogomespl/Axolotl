.PHONY: dev infra test test-cov lint admin dev-all db-migrate db-seed build up down logs evals

# Development
infra:                    ## Start PostgreSQL + Redis + ChromaDB
	docker compose -f docker-compose.dev.yml up -d

dev: infra               ## Start backend with hot-reload
	cd backend && uvicorn app.main:app --reload --port 8000

admin: infra             ## Start admin UI (React dev server)
	cd admin_ui && npm run dev

dev-all: infra           ## Start everything (backend + admin)
	cd backend && uvicorn app.main:app --reload --port 8000 &
	cd admin_ui && npm run dev

# Quality
lint:                    ## Run linter + formatter + type checker
	ruff check . && ruff format --check . && mypy backend/

test:                    ## Run unit tests
	cd backend && pytest tests/ -v

test-cov:                ## Run unit tests with coverage (fails below 90%)
	cd backend && pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=90

evals:                   ## Run evaluation pipeline
	cd backend && pytest ../evals/ -v

# Database
db-migrate:              ## Run Alembic migrations
	cd backend && alembic upgrade head

db-seed:                 ## Seed database with example data
	cd backend && python -m app.scripts.seed

# Docker
build:                   ## Build all Docker images
	docker compose build

up:                      ## Start production stack
	docker compose up -d

down:                    ## Stop all containers
	docker compose down

logs:                    ## Follow Docker logs
	docker compose logs -f
