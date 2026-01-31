.PHONY: help venv install redis-start redis-stop redis-restart api worker ngrok restart-all kill-all

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TABSERVER_PUBLIC_URL ?= https://ungenial-easterly-beth.ngrok-free.dev

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
