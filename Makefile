.PHONY: help dev compose-down web web-install web-build web-start

SHELL := /bin/bash

help:
	@echo "Targets:"
	@echo "  web            Run Next.js frontend"
	@echo "  web-install    Install web dependencies"
	@echo "  web-build      Build web app"
	@echo "  web-start      Start web app (prod)"
	@echo "  dev            Run docker compose stack"
	@echo "  compose-down   Stop docker compose stack"

web:
	cd web && pnpm dev

web-install:
	cd web && pnpm install

web-build:
	cd web && pnpm build

web-start:
	cd web && pnpm start

dev:
	docker compose up --build

compose-down:
	docker compose down
