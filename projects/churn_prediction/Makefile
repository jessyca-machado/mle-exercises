## Carrega .env se existir (KEY=VALUE)
ifneq (,$(wildcard .env))
include .env
export
endif

## MODE=host (default) ou MODE=docker
MODE ?= host

API_SERVICE ?= api
PY_HOST ?= uv run python
PYTEST_HOST ?= uv run pytest

PY_DOCKER := uv run python
PYTEST_DOCKER := uv run pytest
RUFF_DOCKER := uv run ruff

## scripts
TRAIN_JOB := src/jobs/train.py
EXP_SKLEARN := experiments/comparison/train_sklearn.py
EXP_MLP_TORCH := experiments/deep_learning/train_mlp_torch.py
COST_TOOLKIT := experiments/selection/cost_toolkit_metrics.py
COMPARE_MODELS := experiments/selection/compare_models.py

## API host run
API_MODULE := src.api.app:app
API_HOST ?= 0.0.0.0
API_PORT_HOST ?= 8001
API_PORT_DOCKER ?= 8000

## para testes no host sem redis
RATE_LIMIT_STORAGE_URI ?= memory://
export RATE_LIMIT_STORAGE_URI

RUFF_CACHE_DIR ?= .ruff_cache

## lint tools
RUFF ?= uv run ruff

.PHONY: help lint test run up down build rebuild logs shell run-mlflow \
        train exp-sklearn exp-mlp pipeline pipeline-clean pipeline-docker pipeline-host cost compare-models stop-api

help: ## Mostra esta ajuda
	@echo "Usage:"
	@echo "  make lint [MODE=host|docker]"
	@echo "  make test [MODE=host|docker]"
	@echo "  make run  [MODE=host|docker]"
	@echo ""
	@echo "Examples:"
	@echo "  make lint"
	@echo "  make test"
	@echo "  make run"
	@echo "  make test MODE=docker"
	@echo "  make run MODE=docker"
	@echo ""
	@echo "Other:"
	@echo "  make train [MODE=host|docker]"
	@echo "  make exp-sklearn [MODE=host|docker]"
	@echo "  make exp-mlp [MODE=host|docker]"


install-dev:  ## instala dependências de desenvolvimento (host)
	uv sync --extra dev

install-deep: ## instala dependências p/ experimentos deep learning (host)
	uv sync --extra dev --extra deep

lint: ## Executa Ruff para checar estilo e erros
ifeq ($(MODE),docker)
	docker compose run --rm \
		-e RUFF_CACHE_DIR=/tmp/ruff_cache \
		$(API_SERVICE) $(RUFF_DOCKER) check src tests
else
	$(RUFF) check .
endif

test: ## Roda pytest
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PYTEST_DOCKER) -vv
else
	$(PYTEST_HOST) -vv
endif

stop-api: ## Para a API (host e docker) para evitar conflito de porta
	@echo "Parando API no Docker (se existir)..."
	@docker compose stop $(API_SERVICE) >/dev/null 2>&1 || true
	@echo "Matando processo no host que está ouvindo em :$(API_PORT_HOST)..."
	@lsof -tiTCP:$(API_PORT_HOST) -sTCP:LISTEN >/dev/null 2>&1 && \
		kill -9 $$(lsof -tiTCP:$(API_PORT_HOST) -sTCP:LISTEN) || true

run: ## Sobe a API (Docker ou host)
ifeq ($(MODE),docker)
	docker compose up -d $(API_SERVICE)
	docker compose logs -f $(API_SERVICE)
else
	$(MAKE) stop-api MODE=host
	uvicorn $(API_MODULE) --host $(API_HOST) --port $(API_PORT_HOST) --reload
endif

run-mlflow:
	mlflow server --host 0.0.0.0 --port 5001

## Docker helpers (quando MODE=docker)
up: ## Sobe todos os serviços do Docker Compose em background
	docker compose up -d

down: ## Para e remove containers/redes do Compose
	docker compose down

build: ## Faz build da imagem Docker do serviço API
	docker compose build $(API_SERVICE)

rebuild: ## Rebuild da imagem da API sem usar cache
	docker compose build --no-cache $(API_SERVICE)

logs: ## Acompanha logs da API em tempo real
	docker compose logs -f $(API_SERVICE)

shell: ## Abre shell interativo dentro do container da API
	docker compose run --rm $(API_SERVICE) /bin/sh

## Train / Experiments
exp-sklearn: ## Executa experimento scikit-learn: treino e avaliação
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PY_DOCKER) $(EXP_SKLEARN)
else
	$(PY_HOST) $(EXP_SKLEARN)
endif

exp-mlp: ## Executa experimento MLP PyTorch: treino e avaliação (host requer extra 'deep')
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PY_DOCKER) $(EXP_MLP_TORCH)
else
	$(PY_HOST) $(EXP_MLP_TORCH)
endif

train: ## Treina e registra o modelo no MLflow
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PY_DOCKER) $(TRAIN_JOB)
else
	$(PY_HOST) $(TRAIN_JOB)
endif

##Custo e teste de hipóteses
cost: ## Calcula trade-off de custo e impacto no negócio
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PY_DOCKER) $(COST_TOOLKIT)
else
	$(PY_HOST) $(COST_TOOLKIT)
endif

compare-models: ## Compara modelos candidatos e testa hipóteses estatísticas
ifeq ($(MODE),docker)
	docker compose run --rm $(API_SERVICE) $(PY_DOCKER) $(COMPARE_MODELS)
else
	$(PY_HOST) $(COMPARE_MODELS)
endif

db-reset: ## destrói e recria tudo do zero
ifeq ($(MODE),docker)
	docker compose down -v
	docker compose up -d --build
else
	@echo "db-reset só faz sentido com MODE=docker" && exit 2
endif

## Experimentos utilizando o MODE atual (default: host)
experiment: ## Executa fluxo de experimento completo: lint, exp, custo, compare
	$(MAKE) lint MODE=$(MODE)
	$(MAKE) exp-sklearn MODE=$(MODE)
	$(MAKE) exp-mlp MODE=$(MODE)
	$(MAKE) cost MODE=$(MODE)
	$(MAKE) compare-models MODE=$(MODE)

## Pipeline utilizando o MODE atual (default: host)
pipeline: ## Fluxo padrão (não destrói volumes): lint, test, train, run
	$(MAKE) stop-api MODE=$(MODE)
	$(MAKE) lint MODE=$(MODE)
	$(MAKE) test MODE=$(MODE)
	$(MAKE) train MODE=$(MODE)
	$(MAKE) run MODE=$(MODE)

pipeline-clean: ## Fluxo do zero (apaga volumes): lint, reset, test, train, run. Utilizar apenas com Docker
	$(MAKE) lint MODE=docker
	$(MAKE) db-reset MODE=docker
	$(MAKE) test MODE=docker
	$(MAKE) train MODE=docker
	$(MAKE) run MODE=docker

## Atalhos explícitos
pipeline-host: ## Executa pipeline completa no ambiente do host
	$(MAKE) pipeline MODE=host

experiment-host: ## Executa experimentos completos no ambiente do host
	$(MAKE) experiment MODE=host

pipeline-docker: ## Executa pipeline completa dentro do Docker Compose
	$(MAKE) pipeline MODE=docker

experiment-docker: ## Executa experimentos completos dentro do Docker Compose
	$(MAKE) experiment MODE=docker
