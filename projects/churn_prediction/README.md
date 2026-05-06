# Churn Prediction (Telco customer churn IBM)

## Objetivo
Este projeto implementa um modelo supervisionado de churn para identificar clientes com alta probabilidade de deixarem a empresa de telecomunicações (virarem churn). O projeto conta com:
  - **Treino** do modelo end-to-end (feature engineering + preprocess + selector + estimador + registro)

  - **Model Registry e versionamento** com MLflow

  - **API de inferência (FastAPI) com:**
    - /predict e /predict_batch
    - autenticação via API Key (X-API-KEY)
    - rate limiting (SlowAPI + Redis)
    - persistência de predições em Postgres
    - endpoint /metrics (Prometheus)

  - **Monitoramento:**
    - Prometheus + Grafana
    - Drift PSI em job batch (drift.py) persistindo em Postgres (drift_metrics)

## 🛠️ Arquitetura
**Serving (real-time)**
  - FastAPI (Gunicorn + UvicornWorker)
  - Modelo carregado no startup via MLflow PyFunc (CHURN_MODEL_URI)
  - Predições persistidas no Postgres (churn_predictions)

**Tracking/Registry**
  - MLflow Tracking + Model Registry
  - Backend store: Postgres
  - Artifacts: MinIO (S3)
  - mlflow-proxy (nginx) para evitar problemas de Host header entre containers

**Observabilidade**
  - Prometheus scrape em /metrics
  - Drift PSI via drift.py → salva em drift_metrics

## Modos de execução
É possível trabalhar de duas formas:
  - **A. Docker-only**
    - Sobe toda a stack via Docker Compose.
    - Roda jobs (baseline/drift/treino final) dentro dos containers com docker compose.
    - Não precisa instalar Python/uv no host.

  - **B. Dev no host com uv**
    - Usa uv para instalar dependências e rodar scripts no host.
    - Útil principalmente para rodar experiments/ localmente.

## Pré-requisito:

### Para Docker-only
#### Linux/macOS
Recomenda-se instalar via Docker Engine (Linux) ou Docker Desktop (macOS).

```bash
sudo curl -fsSL https://get.docker.com | sh
```

Depois, para rodar Docker sem `sudo`:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verificar se está instalado:
```bash
docker --version
docker compose version
```

#### Windows — Docker Desktop
Instale o Docker Desktop manualmente a partir do link https://docs.docker.com/desktop/setup/install/windows-install/.


### UV Lock
O projeto utiliza o **uv** como gerenciador oficial de pacotes e a versão estrita do **Python 3.12.x**.

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verifique a instalação:
```bash
uv --version
```

### 📦 **Sobre dependências e lockfile**
- As dependências do projeto estão em `pyproject.toml`.
- O arquivo `uv.lock` registra versões exatas para reproduzir o ambiente.
- Para trabalhar com notebooks, o projeto usa `ipykernel` como dependência de desenvolvimento (extra dev).
- O docker também consome as dependências do `uv.lock`

## 🔧 Configuração do ambiente

### para rodar docker
#### 1. Entrar na pasta do projeto:
```bash
cd projects/churn_prediction
```

### para rodar no host
#### 1. Entrar na pasta do projeto:
```bash
cd projects/churn_prediction
```

#### 2. Criar a venv:
```bash
uv venv .venv
```

#### 3. Instalar as dependências (incluindo as de desenvolvimento):
Instale as dependências conforme o `uv.lock` (ou gere/atualize o lock quando necessário):
```bash
uv sync --extra dev
```

#### 4. Ative o ambiente virtual:
##### Linux/macOS
```bash
source .venv/bin/activate
```

##### Windows (PowerShell)
```bash
.\.venv\Scripts\Activate.ps1
```

### crie o .env para rodar o projeto no host
Crie:
`projects/churn_prediction/.env`
Contendo:
```env
CHURN_MODEL_URI=models:/churn_xgb@prod
CHURN_THRESHOLD=0.5
CHURN_API_KEY=`CRIE_UM_TOKEN_PARA_RODAR_NA_API`
RATE_LIMIT_STORAGE_URI=memory://
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio12345
AWS_REGION=us-east-1
MLFLOW_S3_ENDPOINT_URL=http://localhost:9100
MLFLOW_S3_IGNORE_TLS=true
API_URL=http://localhost:8001
```

## Reproduzir o treino
- O projeto roda end-to-end em CPU por padrão (sem necessidade de GPU).
- Para subir a API é necessário respeitar a execução do projeto do experimento até o treino.
- Para rodar a pipeline abaixo é preciso que tenha o make instalado.
- É possível rodar o comandos no terminal, acessando os documentos com recomendação de uso na docstring:

### Scripts / Atalhos (linha de comando)

| Ação | Comando / Referência | Descrição |
|---|---|---|
| Experimento sklearn | `python experiments/comparison/train_sklearn.py` | Executa experimento com modelos/rotina sklearn. |
| Experimento MLP-Pytorch | `python experiments/deep_learning/train_mlp_torch.py` | Executa experimento de deep learning (MLP) em PyTorch. |
| Trade-off de custo | `python experiments/selection/cost_toolkit_metrics.py` | Calcula métricas e trade-off de custo/impacto no negócio. |
| Teste de hipótese | `python experiments/selection/compare_models.py` | Compara modelos via teste de hipótese. |
| Treino | `python src/jobs/train.py` | Treina e registra o modelo final. |
| Testes (unitário e integração) | `pytest -vv` | Executa toda a suíte de testes com verbosidade. |
| API | Docstring em `src/api/app.py` | Instruções/uso da API (endpoints, payload, execução). |

## Um único comando
### Docker
```bash
make pipeline MODE="docker"
```

### Host
```bash
make pipeline
```

## Comandos (Makefile) (cross-platform)

| Comando | Tempo estimado | Descrição |
|---|---:|---|
| `make exp-sklearn` | ~30 min | Validação cruzada estratificada com 10 folds e tuning com `RandomizedGridSearchCV`. |
| `make exp-mlp` | ~2 min | Validação cruzada estratificada com 10 folds e tuning (MLP). |
| `make cost` | ~1 min | Calcula trade-off de custo e impacto no negócio. |
| `make compare-models` | ~1 min | Treina e registra o modelo final (comparação de modelos). |
| `make train` | ~1 min | Treina e registra o modelo final. |
| `make test` | ~1 min | Executa testes de integração e unitários. |
| `make run` | ~1 min | Sobe a API para inferência (em outro terminal, faça o POST dos dados). |

## Exemplo de uso manual da API
### Linux
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $CHURN_API_KEY" \
  -d '{
    "customer_id":"0001",
    "gender":"Female",
    "SeniorCitizen":0,
    "Partner":1,
    "Dependents":0,
    "tenure":12,
    "PhoneService":1,
    "MultipleLines":"No",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":1,
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":70.35,
    "TotalCharges":"845.5"
  }'
```

Veja a [documentação da arquitetura](docs/archtecture.md).

Veja a [documentação do plano de monitoramento](docs/monitoring.md).

Veja o [model card do projeto](model_card.md).

Veja o [canvas do projeto](ml_canvas_exercicios.md).

## Estrutura do projeto

└── churn_prediction                            # Projeto de predição de churn
    ├── .dockerignore                           # Itens excluídos do build do Docker
    ├── .pre-commit-config.yaml                 # Hooks de qualidade antes do commit


    ├── Dockerfile.api                          # Imagem da API
    ├── Dockerfile.mlflow                       # Imagem do MLflow


    ├── Makefile                                # Atalhos de comandos
    ├── README.md                               # README do churn_prediction
    ├── docker-compose.yml                      # Sobe o stack de serviços


    ├── docs                                    # Documentação
    │   ├── archtecture.md                      # Arquitetura do projeto
    │   └── monitoring.md                       # Monitoramento e métricas


    ├── experiments                             # Experimentação e comparação de modelos
    │   ├── __init__.py                         # Define o pacote experiments
    │
    │   ├── comparison                          # Experimentos com abordagem sklearn
    │   │   ├── __init__.py                     # Define o pacote comparison
    │   │   └── train_sklearn.py                # Treina e avalia com sklearn
    │
    │   ├── deep_learning                       # Experimentos de deep learning
    │   │   ├── __init__.py                     # Define o pacote deep_learning
    │   │   └── train_mlp_torch.py              # Treina e avalia MLP em PyTorch
    │
    │   └── selection                           # Seleção de modelo por critérios de negócio
    │       ├── __init__.py                     # Define o pacote selection
    │       ├── compare_models.py               # Compara modelos com teste de hipótese
    │       └── cost_toolkit_metrics.py         # Trade-off de custo e impacto no negócio


    ├── infra                                   # Arquivos de infraestrutura
    │   ├── nginx                               # Configurações do Nginx
    │   │   └── mlflow.conf                     # Proxy e rotas para o MLflow
    │   ├── postgres                            # Configurações do Postgres
    │   │   └── init                            # Scripts de inicialização do banco
    │   └── prometheus                          # Configurações do Prometheus
    │       └── prometheus.yml                  # Targets e regras de coleta de métricas


    ├── ml_canvas_exercises.py                 # Arquivo de canvas inicial do projeto


    ├── model_card.md                           # Model Card do modelo final


    ├── notebooks                               # Notebooks de análise
    │   └── eda.ipynb                           # Análise exploratória de dados


    ├── pyproject.toml                          # Dependências e configuração do projeto
    ├── requirements-mlflow.txt                 # Dependências do serviço MLflow
    ├── uv.lock                                 # Lockfile de dependências


    ├── scripts                                 # Scripts utilitários
    │   ├── __init__.py                         # Define o pacote scripts
    │   └── run_api.sh                          # Script para subir a API


    ├── src                                     # Código fonte
    │   ├── __init__.py                         # Define o pacote src
    │
    │   ├── api                                 # Camada de serving
    │   │   ├── __init__.py                     # Define o pacote api
    │   │   ├── app.py                          # Aplicação da API e endpoints
    │   │   └── metrics.py                      # Métricas exportadas pela API
    │
    │   ├── core                                # Núcleo do domínio
    │   │   ├── __init__.py                     # Define o pacote core
    │   │   └── models                          # Estruturas centrais de modelos
    │
    │   ├── data                                # Pipeline de dados
    │   │   ├── __init__.py                     # Define o pacote data
    │   │   ├── feature_engineering.py          # Criação e transformação de features
    │   │   ├── load_data.py                    # Carregamento de dados
    │   │   ├── preprocess.py                   # Pré-processamento e preparação de dataset
    │   │   └── transformers.py                 # Transformers customizados
    │
    │   ├── entrypoints                         # Pontos de entrada
    │   │   ├── __init__.py                     # Define o pacote entrypoints
    │   │   └── cli.py                          # Interface de linha de comando
    │
    │   ├── infra                               # Integrações com serviços externos
    │   │   ├── __init__.py                     # Define o pacote infra
    │   │   ├── db                              # Integração com banco de dados
    │   │   └── mlflow                          # Integração com MLflow
    │
    │   ├── jobs                                # Rotinas executáveis
    │   │   ├── __init__.py                     # Define o pacote jobs
    │   │   ├── drift.py                        # Rotina de drift
    │   │   ├── generate_traffic.py             # Geração de tráfego para testes e métricas
    │   │   ├── make_baseline.py                # Criação de baseline
    │   │   ├── predict.py                      # Predição em modo job
    │   │   └── train.py                        # Treino e registro do modelo
    │
    │   ├── ml                                  # Componentes de ML
    │   │   ├── __init__.py                     # Define o pacote ml
    │   │   ├── churn_pyfunc_mlp.py             # Wrapper PyFunc para MLP
    │   │   ├── churn_pyfunc_xgb.py             # Wrapper PyFunc para XGBoost
    │   │   ├── cost_utils.py                   # Utilitários de custo
    │   │   ├── data_utils.py                   # Utilitários de dados
    │   │   ├── logging_utils.py                # Utilitários de logging
    │   │   ├── metrics_utils.py                # Utilitários de métricas
    │   │   ├── mlflow_selection_utils.py       # Seleção do melhor run no MLflow
    │   │   ├── mlflow_utils.py                 # Helpers de MLflow
    │   │   └── persistence.py                  # Persistência de artefatos
    │
    │   └── utils                               # Utilidades gerais
    │       ├── __init__.py                     # Define o pacote utils
    │       ├── constants.py                    # Constantes do projeto
    │       └── helpers.py                      # Funções auxiliares


    └── tests                                   # Testes automatizados
        ├── __init__.py                         # Define o pacote tests
        ├── conftest.py                         # Fixtures do pytest
        │
        ├── integration                         # Testes de integração
        │   ├── __init__.py                     # Define o pacote integration
        │   ├── test_api.py                     # Testes da API
        │   ├── test_e2e.py                     # Teste ponta a ponta
        │   └── test_mlflow_logging.py          # Testes de logging no MLflow
        │
        └── units                               # Testes unitários
            ├── __init__.py                     # Define o pacote units
            │
            ├── test_load_data.py               # Testes de carga de dados
            ├── test_mlflow_fetch_best_params.py# Testes de seleção de parâmetros
            ├── test_preprocessing.py           # Testes de pré-processamento
            ├── test_preprocessor_sanity.py     # Sanidade do preprocessor
            ├── test_pyfunc_contract_unit.py    # Contrato do PyFunc
            ├── test_trainer_pipeline.py        # Testes do pipeline de treino
            └── test_trainer_predict_pyfunc_mode.py # Predição via PyFunc em testes
