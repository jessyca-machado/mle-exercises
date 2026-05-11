# Churn Prediction (Telco customer churn IBM)

## índice

1. [Objetivo](#objetivo)
2. [Arquitetura](#arquitetura)
3. [Pré-requisitos](#pré-requisito)
4. [Configurar ambiente](#configurar-ambiente)
5. [Reproduzir o treino](#reproduzir-o-treino)
6. [Utilizar endpoints API](#utilizar-endpoints-api)
7. [Estrutura do repositório](#estrutura-do-repositório)
8. [Links úteis](#links-úteis)
9. [Vídeo STAR](#vídeo-star)
10. [Referências](#referências)

## Objetivo
Este repositório implementa um modelo de churn para identificar clientes com alta probabilidade de deixarem a empresa de telecomunicações (virarem churn). O repositório conta com:

  - **Experimentações** com baseline, trade-off de custo e teste de hipóteses.

  - **Treino** do melhor modelo end-to-end (feature engineering + preprocess + selector + estimador + registro).

  - **Model Registry e versionamento** com MLflow.

  - **API de inferência (FastAPI) com:**
    - /health.
    - /predict e /predict_batch.
    - autenticação via API Key (X-API-KEY).
    - rate limiting (SlowAPI + Redis).
    - persistência de predições em Postgres.
    - endpoint /metrics (Prometheus).

  - **Monitoramento:**
    - Prometheus.
    - Drift PSI em job batch (drift.py) persistindo em Postgres (drift_metrics).

## Arquitetura
**Tracking/Registry**
  - MLflow Tracking + Model Registry.
  - Backend store: Postgres.
  - Artifacts: MinIO (S3).
  - mlflow-proxy (nginx) para evitar problemas de Host header entre containers.

**Serving (real-time)**
  - FastAPI (Gunicorn + UvicornWorker).
  - Modelo carregado no startup via MLflow PyFunc (CHURN_MODEL_URI).
  - Predições persistidas no Postgres (churn_predictions).

**Observabilidade**
  - Prometheus scrape em /metrics.
  - Drift PSI via drift.py → salva em drift_metrics.

## Pré-requisito:
- Docker
- Docker Compose
- Make. Para Windows, instalar via WSL2, Chocolatey ou GnuWin32
- UV
- Python 3.12

**Instalação:**
- Docker: https://docs.docker.com/get-docker/. Para Windows, é preciso possuir o backend Linux WSL2, as instruções estão na mesma documentação.
- uv: https://docs.astral.sh/uv/getting-started/installation/.

### É possível trabalhar de duas formas:
  - **A. Docker-only**
    - Sobe toda a stack via Docker Compose.
    - Roda jobs (baseline/drift/treino final) dentro dos containers com docker compose.
    - Não precisa instalar Python/uv no host.

  - **B. Dev no host com uv**
    - Usa uv para instalar dependências e rodar scripts no host.
    - Útil principalmente para rodar experiments/ localmente.

### Sobre dependências, extras e lockfile (uv)
- As dependências estão em `pyproject.toml`.
- O arquivo `uv.lock` registra versões exatas para reproduzir o ambiente.
- O projeto usa **extras** para separar dependências por finalidade:
  - **Core (default)**: dependências necessárias para rodar a API e o pipeline principal (ex.: FastAPI, MLflow, XGBoost, pandas, etc.).
  - **extra `dev`**: ferramentas de desenvolvimento e testes (ex.: `ruff`, `pytest`, `ipykernel`, `pre-commit`, etc.).
  - **extra `deep`**: dependências de deep learning para os experimentos em PyTorch (ex.: `torch`, `skorch`).

#### Quando usar o extra deep?
- Para rodar o experimento de MLP em PyTorch (`experiments/deep_learning/train_mlp_torch.py`):
  - Instale **core + dev + deep**.

**Observação:**
- A imagem Docker da API inclui o extra `dev` para permitir rodar `ruff/pytest` dentro do container (targets `lint`/`test` com `MODE=docker`).
- O extra `deep` (PyTorch) é necessário para o experimento MLP. Se você rodar `make exp-mlp MODE=host`, instale com `make install-deep`.

## Configurar ambiente
```bash
cd projects/churn_prediction
# Copie o .env.example para o .env
cp .env.example .env
```

### para rodar docker (recomendável)
```bash
cd projects/churn_prediction
```
```bash
# primeira vez (ou após mudanças no código/dependências)
docker compose up -d --build
```

### para rodar no host
```bash
cd projects/churn_prediction
```
```bash
# criar venv
uv venv .venv
```
```bash
# Instalar core + dev (lint/test/notebooks)
uv sync --extra dev
```
```bash
# Instalar também o extra deep (necessário para experimentos com PyTorch/MLP):
uv sync --extra dev --extra deep
```
```bash
# Ative o venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows
```

## Reproduzir o treino
- Todas as etapas são registradas no mlflow.
- Os parâmetros do melhor modelo do experimento (`xgboost`), utilizados na pipeline, estão [nesse caminho](configs/best_model.yml).

### Nota sobre o endpoint `/ready` (warmup do modelo)
- A API carrega o modelo do MLflow **de forma assíncrona** na inicialização.
- Durante alguns segundos (enquanto o modelo ainda não foi carregado), o endpoint `/ready` pode retornar **HTTP 503**.
- Assim que o modelo termina de carregar, `/ready` passa a retornar **HTTP 200** e a API fica pronta para receber predições.

### Um único comando
#### Pipeline Docker (padrão)
Executa o fluxo padrão (lint → testes → treino → sobe API) **sem apagar volumes**.
```bash
cd projects/churn_prediction
make pipeline-docker
# OU
cd projects/churn_prediction
docker compose run --rm api uv run pytest -vv
docker compose run --rm api uv run python src/jobs/train.py
docker compose up -d api
docker compose logs -f api
```

#### Pipeline Docker (do zero)
Reinicia a stack do zero (docker compose down -v), apagando volumes do Postgres/MinIO, etc. Remove histórico do MLflow e artifacts
```bash
cd projects/churn_prediction
make pipeline-clean
# OU
cd projects/churn_prediction
docker compose down -v
docker compose up -d --build
docker compose run --rm api uv run pytest -vv
docker compose run --rm api uv run python src/jobs/train.py
docker compose up -d api
docker compose logs -f api
```

#### Experimentos Docker
Rodar experimentos sklearn + MLP com Pytorch + trade-off de custo + teste de hipóteses entre modelos.
> Observação: A imagem Docker do serviço `api` instala `--extra dev --extra deep` para permitir rodar `ruff/pytest` e o experimento MLP (PyTorch) dentro do container.
> Isso aumenta o tamanho e o tempo de build da imagem.
```bash
cd projects/churn_prediction
make experiment-docker
# OU
cd projects/churn_prediction
docker compose run --rm api uv run python experiments/comparison/train_sklearn.py
docker compose run --rm api uv run python experiments/deep_learning/train_mlp_torch.py
docker compose run --rm api uv run python experiments/selection/cost_toolkit_metrics.py
docker compose run --rm api uv run python experiments/selection/compare_models.py
```

#### Pipeline Host
Rodar pipeline: teste + treino + api
```bash
cd projects/churn_prediction
make pipeline-host
# OU
cd projects/churn_prediction
uv run pytest -vv
uv run python src/jobs/train.py
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

#### Experimentos Host
Rodar experimentos sklearn + MLP com Pytorch + trade-off de custo + teste de hipóteses entre modelos
```bash
cd projects/churn_prediction
make experiment-host
# OU
cd projects/churn_prediction
uv run python experiments/comparison/train_sklearn.py
uv run python experiments/deep_learning/train_mlp_torch.py
uv run python experiments/selection/cost_toolkit_metrics.py
uv run python experiments/selection/compare_models.py
```

### Comandos (Makefile) (cross-platform)
Os comandos aceitam `MODE=docker` para rodar dentro do Docker Compose e `MODE=host` para rodar no host.
Atalhos disponíveis: `pipeline-docker`, `pipeline-host`, `experiment-docker`, `experiment-host`.

| Comando | Tempo estimado (Docker) | Descrição |
|---|---:|---|
| `make exp-sklearn` | ~30 min | Validação cruzada estratificada com 10 folds e tuning com `RandomizedSearchCV`. |
| `make exp-mlp` | ~5 min | Validação cruzada estratificada com 10 folds e tuning (MLP). |
| `make cost` | ~1 min | Calcula trade-off de custo e impacto no negócio. |
| `make compare-models` | ~1 min | Registra a comparação de modelos e teste de hipóteses. |
| `make train` | ~1 min | Treina e registra o modelo final escolhido. |
| `make test` | ~1 min | Executa testes de integração e unitários pytest. |
| `make run` | ~1 min | Sobe a API para inferência (em outro terminal, faça o POST dos dados no caminho `cd projects/churn_prediction`). |

## Utilizar endpoints API
```bash
## Linux/MacOS
set -a; source .env; set +a
API_URL="$API_URL_DOCKER"
curl -fsS "$API_URL/health" >/dev/null 2>&1 || API_URL="$API_URL_HOST"
until curl -fsS -H "X-API-Key: $CHURN_API_KEY" "$API_URL/ready" >/dev/null; do
  echo "Aguardando modelo carregar em $API_URL ..."
  sleep 2
done
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHURN_API_KEY" \
  --data-binary @examples/payload.json
echo
```
```bash
# PowerShell (Windows)
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}
$API_URL = $env:API_URL_DOCKER
try {
  curl.exe -fsS "$API_URL/health" | Out-Null
} catch {
  $API_URL = $env:API_URL_HOST
}
Write-Host "Usando API_URL=$API_URL"
while ($true) {
  try {
    curl.exe -fsS -H "X-API-Key: $env:CHURN_API_KEY" "$API_URL/ready" | Out-Null
    break
  } catch {
    Write-Host "Aguardando modelo carregar em $API_URL ..."
    Start-Sleep -Seconds 2
  }
}
curl.exe -s -X POST "$API_URL/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:CHURN_API_KEY" `
  --data-binary "@examples/payload.json"
Write-Host ""
```

## Estrutura do repositório
```text
└── churn_prediction                            # Predição de churn
    ├── configs
    │   ├── best_model.yml                      # Hiperparâmetros do modelo xgboost utilizado no treino
    ├── docs
    │   ├── archtecture.md                      # Arquitetura
    │   ├── monitoring.md                       # Monitoramento e métricas
    │   ├── ml_canvas_exercises.py              # Arquivo de canvas inicial
    │   └── model_card.md                       # Model Card do modelo final
    ├── experiments
    │   ├── comparison
    │   │   └── train_sklearn.py                # Treina e avalia com sklearn
    │   ├── deep_learning
    │   │   └── train_mlp_torch.py              # Treina e avalia MLP em PyTorch
    │   └── selection
    │       ├── compare_models.py               # Compara modelos com teste de hipótese
    │       └── cost_toolkit_metrics.py         # Trade-off de custo e impacto no negócio
    ├── infra
    │   ├── nginx                               # Configurações do Nginx
    │   │   └── mlflow.conf                     # Proxy e rotas para o MLflow
    │   ├── postgres                            # Configurações do Postgres
    │   │   └── init                            # Scripts de inicialização do banco
    │   └── prometheus                          # Configurações do Prometheus
    │       └── prometheus.yml                  # Targets e regras de coleta de métricas
    ├── notebooks
    │   └── eda.ipynb                           # Análise exploratória de dados
    ├── src
    │   ├── api
    │   │   ├── app.py                          # Aplicação da API e endpoints
    │   │   └── metrics.py                      # Métricas exportadas pela API
    │   ├── core
    │   │   └── models                          # Estruturas centrais de modelos
    │   ├── data
    │   │   ├── feature_engineering.py          # Criação e transformação de features
    │   │   ├── load_data.py                    # Carregamento de dados
    │   │   ├── preprocess.py                   # Pré-processamento e preparação de dataset
    │   │   └── transformers.py                 # Transformers customizados
    │   ├── entrypoints
    │   │   └── cli.py                          # Interface de linha de comando
    │   ├── infra
    │   │   ├── db                              # Integração com banco de dados
    │   │   └── mlflow                          # Integração com MLflow
    │   ├── jobs
    │   │   ├── drift.py                        # Rotina de drift
    │   │   ├── generate_traffic.py             # Geração de tráfego para testes e métricas
    │   │   ├── make_baseline.py                # Criação de baseline
    │   │   ├── predict.py                      # Predição em modo job
    │   │   └── train.py                        # Treino e registro do modelo
    │   ├── ml
    │   │   ├── churn_pyfunc_mlp.py             # Wrapper PyFunc para MLP
    │   │   ├── churn_pyfunc_xgb.py             # Wrapper PyFunc para XGBoost
    │   │   ├── cost_utils.py                   # Utilitários de custo
    │   │   ├── data_utils.py                   # Utilitários de dados
    │   │   ├── logging_utils.py                # Utilitários de logging
    │   │   ├── metrics_utils.py                # Utilitários de métricas
    │   │   ├── mlflow_selection_utils.py       # Seleção do melhor run no MLflow
    │   │   ├── mlflow_utils.py                 # Helpers de MLflow
    │   │   └── persistence.py                  # Persistência de artefatos
    │   └── utils
    │       ├── constants.py                    # Constantes utilizadas
    │       └── helpers.py                      # Funções auxiliares
    ├── tests
    │   ├── conftest.py                         # Fixtures do pytest
    │   ├── integration
    │   │   ├── test_api.py                     # Testes da API
    │   │   ├── test_e2e.py                     # Teste ponta a ponta
    │   │   └── test_mlflow_logging.py          # Testes de logging no MLflow
    │   └── units
    │       ├── test_load_data.py               # Testes de carga de dados
    │       ├── test_mlflow_fetch_best_params.py# Testes de seleção de parâmetros
    │       ├── test_preprocessing.py           # Testes de pré-processamento
    │       ├── test_preprocessor_sanity.py     # Sanidade do preprocessor
    │       ├── test_pyfunc_contract_unit.py    # Contrato do PyFunc
    │       ├── test_trainer_pipeline.py        # Testes do pipeline de treino
    │       └── test_trainer_predict_pyfunc_mode.py # Predição via PyFunc em testes
    ├── .env.example                            # Exemplo de variáveis de ambiente
    ├── pyproject.toml                          # Dependências e configuração
    ├── requirements-mlflow.txt                 # Dependências do serviço MLflow
    ├── uv.lock                                 # Lockfile de dependências
    ├── .dockerignore                           # Itens excluídos do build do Docker
    ├── .pre-commit-config.yaml                 # Hooks de qualidade antes do commit
    ├── Dockerfile.api                          # Imagem da API
    ├── Dockerfile.mlflow                       # Imagem do MLflow
    ├── Makefile                                # Atalhos de comandos
    ├── docker-compose.yml                      # Sobe o stack de serviços
    └── README.md                               # ← Você está aqui
```

## Links úteis

- Veja a [documentação da arquitetura](docs/archtecture.md).
- Veja a [documentação do plano de monitoramento](docs/monitoring.md).
- Veja o [model card](docs/model_card.md).
- Veja o [canvas](docs/ml_canvas_exercicios.md).

## Vídeo STAR
[Assista o vídeo](https://drive.google.com/file/d/1gySWCibZ_yyCqUJg683hFJ_9G6r1hs3G/view?usp=sharing)

## Referências
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/2.11/index.html?_gl=1*13zs027*_up*MQ..*_ga*NzEwMjAzMDIzLjE3NzgyMDM5NTA.*_ga_469Y0W5V62*czE3NzgyMDM5NDkkbzEkZzAkdDE3NzgyMDM5NDkkajYwJGwwJGgw)
- [Scikit-Learn DOcumentation](https://scikit-learn.org/0.21/documentation.html)
- [Docker Documentation](https://docs.docker.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/release_3.2.0/)

## LICENSE
[MIT LICENSE](https://github.com/jessyca-machado/mle-exercises/blob/main/LICENSE).
