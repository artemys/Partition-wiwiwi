.PHONY: help venv install redis-start redis-stop redis-restart api worker ngrok restart-all kill-all dev web web-install web-build web-start compose-down

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TABSERVER_PUBLIC_URL ?= http://localhost:8000

help:
	@echo "Targets:"
	@echo "  venv           Create virtualenv"
	@echo "  install        Install server deps"
	@echo "  redis-start    Start Redis (brew)"
	@echo "  redis-stop     Stop Redis (brew)"
	@echo "  redis-restart  Restart Redis (brew)"
	@echo "  api            Run FastAPI server"
	@echo "  worker         Run RQ worker"
	@echo "  ngrok          Start ngrok tunnel on 8000"
	@echo "  kill-all       Kill api/worker/ffmpeg/ngrok"
	@echo "  restart-all    Restart redis + api + worker"
	@echo "  web            Run Next.js frontend"
	@echo "  web-install    Install web dependencies"
	@echo "  web-build      Build web app"
	@echo "  web-start      Start web app (prod)"
	@echo "  dev            Run docker compose stack"
	@echo "  compose-down   Stop docker compose stack"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r server/requirements.txt

redis-start:
	brew services start redis

redis-stop:
	brew services stop redis

redis-restart:
	brew services restart redis

api:
	source $(VENV)/bin/activate && \
	TABSERVER_PUBLIC_URL=$(TABSERVER_PUBLIC_URL) \
	uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	source $(VENV)/bin/activate && \
	python -m server.worker

web:
	cd web && pnpm dev

web-install:
	cd web && pnpm install

web-build:
	cd web && pnpm build

web-start:
	cd web && pnpm start

ngrok:
	ngrok http 8000

kill-all:
	pkill -f "uvicorn server.main:app" || true; \
	pkill -f "python -m server.worker" || true; \
	pkill -f "ffmpeg" || true; \
	pkill -f "ngrok http 8000" || true

restart-all: kill-all redis-restart
	source $(VENV)/bin/activate && \
	TABSERVER_PUBLIC_URL=$(TABSERVER_PUBLIC_URL) \
	uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload &
	sleep 1
	source $(VENV)/bin/activate && \
	python -m server.worker

dev:
	docker compose up --build

compose-down:
	docker compose down
