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
- Docker e Docker Compose para rodar a stack completa + make.
- Modo host: Python 3.12 + uv + make.

**Instalação:**
- Docker: https://docs.docker.com/get-docker/.
- uv: https://docs.astral.sh/uv/getting-started/installation/.

### É possível trabalhar de duas formas:
  - **A. Docker-only**
    - Sobe toda a stack via Docker Compose.
    - Roda jobs (baseline/drift/treino final) dentro dos containers com docker compose.
    - Não precisa instalar Python/uv no host.

  - **B. Dev no host com uv**
    - Usa uv para instalar dependências e rodar scripts no host.
    - Útil principalmente para rodar experiments/ localmente.


### Sobre dependências e lockfile
- As dependências estão em `pyproject.toml`.
- O arquivo `uv.lock` registra versões exatas para reproduzir o ambiente.
- Para trabalhar com notebooks, é utilizado o `ipykernel` como dependência de desenvolvimento (extra dev).
- O docker também consome as dependências do `uv.lock`.

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
# Instale as dependências conforme o `uv.lock` (ou gere/atualize o lock quando necessário)
uv sync --extra dev
```
```bash
# Ative o venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows
```

## Reproduzir o treino
- Para subir a API é necessário respeitar a ordem de execução do experimento até o treino.
- Todas as etapas são registradas no mlflow e os parâmetros do melhor modelo são filtrados por lá.

### Um único comando
#### Docker
```bash
make pipeline-docker
```

#### Host
```bash
make pipeline-host
```

### Comandos (Makefile) (cross-platform)
inclua o MODE=docker após os comandos se for rodar com Docker.

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
# Linux/macOS
# carregar variáveis do .env e utiliza o endpoint.
set -a; source .env; set +a
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHURN_API_KEY" \
  --data-binary @examples/payload.json
```
```bash
# Windows (PowerShell)
# carregar variáveis do .env e utiliza o endpoint
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}

curl.exe -s -X POST "$env:API_URL/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:CHURN_API_KEY" `
  --data-binary "@examples/payload.json"
```

## Estrutura do repositório
```text
└── churn_prediction                            # Predição de churn
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
    ├── scripts
    │   └── run_api.sh                          # Script para subir a API
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
Veja o arquivo [LICENSE](https://github.com/jessyca-machado/mle-exercises/blob/main/LICENSE).
