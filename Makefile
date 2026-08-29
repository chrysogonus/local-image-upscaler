.PHONY: setup setup-venv setup-frontend setup-browser setup-comfyui setup-model-comfyui-illustration up down logs shell dev-backend dev-frontend build package compose-config test test-frontend test-e2e lint check ci-local clean-data clean-data-force

setup: setup-venv setup-frontend

setup-venv:
	uv sync --extra dev

setup-frontend:
	pnpm --dir frontend install
	pnpm --dir frontend exec playwright install chromium

setup-browser:
	pnpm --dir frontend exec playwright install chromium

# Installs or adopts the ComfyUI that the Illustration mode drives, then records
# where it is so `make up`, `make down` and `make clean-data` need no arguments.
# An installation already on the machine is adopted rather than replaced;
# COMFYUI_DIR picks a different location for a fresh clone.
setup-comfyui:
	uv run python scripts/install-comfyui.py $(if $(COMFYUI_DIR),--dir "$(COMFYUI_DIR)")

# Reinstalls just the illustration weight into the recorded ComfyUI. Normally
# covered by setup-comfyui; kept for repairing that one file on its own.
setup-model-comfyui-illustration:
	uv run python scripts/install-weights.py --group comfyui-illustration \
		--dir "$(or $(UPSCALER_COMFYUI_UPSCALE_MODELS_DIR),$(shell sed -n 's/^COMFYUI_ROOT=//p' .upscaler/comfyui.conf 2>/dev/null)/models/upscale_models)"

# One command for the whole deployment: the ComfyUI the Illustration mode needs,
# then the app wired to it. scripts/compose.sh adds the GPU reservation when the
# host can actually satisfy one.
#
# ComfyUI is checked first and is never allowed to stop the app: a machine
# without it still runs Upscale and Sharpen, and the mode that needs it says so
# itself. UPSCALER_COMFYUI=0 skips it entirely.
up:
	@if [ "$(UPSCALER_COMFYUI)" = "0" ]; then \
		rm -f .upscaler/comfyui.env; \
		echo "up: UPSCALER_COMFYUI=0, not starting ComfyUI"; \
	elif ./scripts/comfyui-service.sh check >/dev/null 2>&1; then \
		./scripts/comfyui-service.sh start; \
	else \
		rm -f .upscaler/comfyui.env; \
		echo "up: starting without Illustration —"; \
		./scripts/comfyui-service.sh check || true; \
	fi
	./scripts/compose.sh up --build -d

# Stops everything this project started, in the order that keeps the app from
# talking to a ComfyUI that is going away.
down:
	./scripts/compose.sh down
	@if [ -f .upscaler/comfyui.conf ]; then ./scripts/comfyui-service.sh stop; fi

logs:
	./scripts/compose.sh logs -f

shell:
	./scripts/compose.sh exec upscaler bash

dev-backend:
	uv run uvicorn upscaler.app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

dev-frontend:
	pnpm --dir frontend dev

build:
	pnpm --dir frontend build

package:
	uv run python scripts/check-package.py

# Both resolutions, because the host running this only has one of them.
compose-config:
	UPSCALER_GPU=0 ./scripts/compose.sh config --quiet
	UPSCALER_GPU=1 ./scripts/compose.sh config --quiet

test:
	uv run pytest --cov=upscaler --cov-branch --cov-report=term-missing

test-frontend:
	pnpm --dir frontend test:coverage

test-e2e:
	pnpm --dir frontend test:e2e

lint:
	uv run ruff check backend scripts
	uv run ruff format --check backend scripts
	uv run mypy backend/upscaler
	uv run shellcheck scripts/*.sh
	pnpm --dir frontend lint
	pnpm --dir frontend format:check
	pnpm --dir frontend check

check: lint test test-frontend test-e2e build package

# Every gate in one run. Unlike `check` it does not stop at the first failure,
# it verifies both lockfiles, and it refuses to report a gate as passing when
# the toolchain to run it is missing. GATES=backend or GATES=frontend narrows it.
GATES ?= all

ci-local:
	@./scripts/ci-local.sh $(GATES)

# Erases every picture this app has stored: job workspaces on the host and in the
# Docker volume, and ComfyUI's input, output and temp directories, its run history
# and its saved workflows. Model weights are kept. Prompts also live in the
# browser's local storage, which no command here can reach.
#
# The ComfyUI half uses the installation recorded by `make setup-comfyui`.
# COMFYUI=/path/to/ComfyUI overrides it for one that was never recorded.
CLEAN_DATA_ARGS = $(if $(COMFYUI),--comfyui "$(COMFYUI)")

clean-data:
	uv run python scripts/clean-data.py $(CLEAN_DATA_ARGS)

clean-data-force:
	uv run python scripts/clean-data.py --yes $(CLEAN_DATA_ARGS)
