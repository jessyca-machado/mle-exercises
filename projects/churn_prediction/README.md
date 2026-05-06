# Churn Prediction (Telco customer churn IBM)

## ** Objetivo**
Este projeto implementa um modelo supervisionado de churn para identificar clientes com alta probabilidade de deixarem a empresa de telecomunicações (virarem churn). O projeto conta com:
  - **Treino** modelo end-to-end (feature engineering + preprocess + selector + estimador + registro)

  - **Model Registry e versionamento** com MLflow

  **API de inferência (FastAPI) com:**
    - /predict e /predict_batch
    - autenticação via API Key (X-API-KEY)
    - rate limiting (SlowAPI + Redis)
    - persistência de predições em Postgres
    - endpoint /metrics (Prometheus)

  **Monitoramento:**
    - Prometheus + Grafana
    - Drift PSI em job batch (drift.py) persistindo em Postgres (drift_metrics)

## 🛠️ Arquitetura**
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

## **Modos de execução**
É possível trabalhar de duas formas:
  **A. Docker-only**
    - Sobe toda a stack via Docker Compose.
    - Roda jobs (baseline/drift/treino final) dentro dos containers com docker compose.
    - Não precisa instalar Python/uv no host.

  **B. Dev no host com uv**
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


### Para Dev com uv
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

## 🔧 Configuração do ambiente - para rodar no host

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

#### 2. **Criar a venv:**
```bash
uv venv .venv
```

#### 3. **Instalar as dependências (incluindo as de desenvolvimento):**
Instale as dependências conforme o `uv.lock` (ou gere/atualize o lock quando necessário):
```bash
uv sync --extra dev
```

#### 4. **Ative o ambiente virtual:**
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
  - Experimento sklearn: `python experiments/comparison/train_sklearn.py`
  - Experimento MLP-Pytorch: `python experiments/deep_learning/train_mlp_torch.py`
  - Trade-off de custo: `python experiments/selection/cost_toolkit_metrics.py`
  - Teste de hipótese: `python experiments/selection/compare_models.py`
  - Treino: `python src/jobs/train.py`
  - Testes (unitário e integração): `pytest -vv`
  - API: detalhes na docstring da `/projects/churn_prediction/src/api/app.py`

## Um único comando
### Docker
```bash
make pipeline MODE="docker"
```

### Host
```bash
make pipeline
```

## OU passo a passo (cross-platform)
make exp-sklearn   # ~30min. validação cruzada estratificada com 10 folds e tuning com RandomizedGridSearchCV
make exp-mlp     # ~2min. validação cruzada estratificada com 10 folds e tuning
make cost         ## ~1min: Calcula trade-off de custo e impacto no negócio
make compare-models # ~1min: treina/ registra modelo final
make train       # ~1min: treina/ registra modelo final
make test   # ~1min: testes de integração e unitários
make run      # ~1min: sobe a API. A partir daqui já é possível fazer inferência. Abra um segundo terminal e faça o post dos dados

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

## URLs úteis
