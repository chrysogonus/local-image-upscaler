.PHONY: setup setup-venv setup-frontend setup-browser setup-model setup-cuda setup-model-cuda setup-swinir setup-model-swinir setup-comfyui setup-model-comfyui-illustration up up-cuda up-comfyui down down-comfyui logs shell dev-backend dev-frontend build package compose-config test test-frontend test-e2e lint check ci-local run clean-data clean-data-force

setup: setup-venv setup-frontend

setup-venv:
	uv sync --extra dev

setup-frontend:
	pnpm --dir frontend install
	pnpm --dir frontend exec playwright install chromium

setup-browser:
	pnpm --dir frontend exec playwright install chromium

setup-model:
	./scripts/install-realesrgan-linux.sh

# Override for a different CUDA runtime, e.g. UPSCALER_TORCH_INDEX=.../whl/cu128
UPSCALER_TORCH_INDEX ?= https://download.pytorch.org/whl/cu130

setup-cuda:
	uv sync --extra dev --extra cuda --index "$(UPSCALER_TORCH_INDEX)"

setup-model-cuda:
	uv run python scripts/install-weights.py --group realesrgan

# Transformer engine for Upscale. Needs the same torch as the CUDA engine plus
# spandrel, which supplies the architecture.
setup-swinir:
	uv sync --extra dev --extra swinir --index "$(UPSCALER_TORCH_INDEX)"

setup-model-swinir:
	uv run python scripts/install-weights.py --group swinir

# Drives the Illustration graph on your own ComfyUI. There is no
# setup-model-comfyui: the weights belong to that installation.
setup-comfyui:
	uv sync --extra dev --extra comfyui

# Installs the faithful illustration model where the user's ComfyUI can load it.
# Example: make setup-model-comfyui-illustration \
#   UPSCALER_COMFYUI_UPSCALE_MODELS_DIR=/opt/ComfyUI/models/upscale_models
setup-model-comfyui-illustration:
	test -n "$(UPSCALER_COMFYUI_UPSCALE_MODELS_DIR)" || (echo "Set UPSCALER_COMFYUI_UPSCALE_MODELS_DIR to ComfyUI/models/upscale_models" >&2; exit 2)
	uv run python scripts/install-weights.py --group comfyui-illustration --dir "$(UPSCALER_COMFYUI_UPSCALE_MODELS_DIR)"

up:
	docker compose up --build -d

CUDA_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.cuda.yml

up-cuda:
	$(CUDA_COMPOSE) up --build -d

# Start the existing host ComfyUI and the containerised app as one local-only
# deployment. Override COMFYUI or COMFYUI_PORT for another installation.
COMFYUI ?= $(HOME)/comfy/ComfyUI
COMFYUI_PORT ?= 8188
COMFYUI_WORK ?= $(CURDIR)/.upscaler/comfyui-container-work
COMFYUI_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.comfyui.yml

up-comfyui:
	./scripts/comfyui-service.sh start "$(COMFYUI)" "$(COMFYUI_PORT)"
	install -d -m 0700 "$(COMFYUI_WORK)"
	COMFYUI_ROOT="$(abspath $(COMFYUI))" COMFYUI_PORT="$(COMFYUI_PORT)" \
		UPSCALER_COMFYUI_UID="$$(id -u)" UPSCALER_COMFYUI_GID="$$(id -g)" \
		UPSCALER_COMFYUI_WORK_ROOT="$(abspath $(COMFYUI_WORK))" \
		$(COMFYUI_COMPOSE) up --build -d

down:
	docker compose down

down-comfyui:
	COMFYUI_ROOT="$(abspath $(COMFYUI))" COMFYUI_PORT="$(COMFYUI_PORT)" $(COMFYUI_COMPOSE) down
	./scripts/comfyui-service.sh stop "$(COMFYUI)" "$(COMFYUI_PORT)"

logs:
	docker compose logs -f

shell:
	docker compose exec upscaler bash

dev-backend:
	uv run uvicorn upscaler.app:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

dev-frontend:
	pnpm --dir frontend dev

build:
	pnpm --dir frontend build

package:
	uv run python scripts/check-package.py

compose-config:
	docker compose config --quiet
	$(CUDA_COMPOSE) config --quiet
	COMFYUI_ROOT="$(abspath $(COMFYUI))" COMFYUI_PORT="$(COMFYUI_PORT)" \
		UPSCALER_COMFYUI_UID="$$(id -u)" UPSCALER_COMFYUI_GID="$$(id -g)" \
		UPSCALER_COMFYUI_WORK_ROOT="$(abspath $(COMFYUI_WORK))" \
		$(COMFYUI_COMPOSE) config --quiet

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

run: build
	uv run upscaler

# Erases every picture this app has stored: job workspaces on the host and in the
# Docker volume, and ComfyUI's input, output and temp directories, its run history
# and its saved workflows. Model weights are kept. Prompts also live in the
# browser's local storage, which no command here can reach.
#
# The ComfyUI half needs to know where ComfyUI is. A shell running make does not
# normally carry the variables the app's launcher sets, so name it here:
#   make clean-data COMFYUI=/path/to/ComfyUI
CLEAN_DATA_ARGS = $(if $(COMFYUI),--comfyui "$(COMFYUI)")

clean-data:
	UPSCALER_COMFYUI_WORK_ROOT="$(abspath $(COMFYUI_WORK))" uv run python scripts/clean-data.py $(CLEAN_DATA_ARGS)

clean-data-force:
	UPSCALER_COMFYUI_WORK_ROOT="$(abspath $(COMFYUI_WORK))" uv run python scripts/clean-data.py --yes $(CLEAN_DATA_ARGS)
