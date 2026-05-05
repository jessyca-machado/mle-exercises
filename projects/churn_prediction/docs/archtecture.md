
---

# 1) Arquitetura de deploy escolhida (batch vs real‑time) + justificativa

## 1.1 Visão geral (decisão)
A solução adotada é **híbrida**:

- **Real‑time (online)**: a predição de churn é servida via **API FastAPI** (`/predict` e `/predict_batch`) para consumo sob demanda.
- **Batch (offline/assíncrono)**: processos auxiliares executados como jobs (não necessariamente via scheduler neste momento) para:
  - registrar e versionar modelos
  - calcular drift (sem rótulo) em janelas de tempo
  - futuramente avaliar performance com rótulo atrasado e retreinar

Essa escolha atende a dois aspectos típicos de churn:
- necessidade de **consulta imediata** quando o cliente aparece em um ponto de contato (atendimento) → real‑time;
- necessidade de **processos periódicos** de controle de qualidade, drift e atualização do modelo → batch.

---

## 1.2 Componentes (o que roda em produção)
### Serving (tempo real)
- **FastAPI** com Gunicorn/UvicornWorker
- Carregamento do modelo no startup via MLflow PyFunc:
  - `CHURN_MODEL_URI` (ex: `models:/churn_xgb/2`)
- Endpoints:
  - `/health` (liveness)
  - `/ready` (readiness: modelo carregado)
  - `/predict` (1 registro)
  - `/predict_batch` (lista)
  - `/metrics` (Prometheus scrape)

### Model Registry e artifacts
- **MLflow Tracking + Model Registry**
  - backend store: **Postgres**
  - artifacts: **MinIO (S3 compatible)**
- Observação: para evitar problemas de Host header entre containers, foi criado um **reverse proxy Nginx** (`mlflow-proxy`), e os serviços dentro da rede docker usam `http://mlflow-proxy`.

### Persistência operacional de predições
- **Postgres (DB churn)** com tabela `churn_predictions`
  - armazena `request_id`, `batch_id`, `item_index`, `model_uri`, `y_pred`, `y_pred_proba`, `threshold`, `features (JSONB)`
- Justificativa: sem persistência, não existe base para:
  - drift real com dados de produção
  - avaliação posterior com rótulo atrasado
  - auditoria / rastreabilidade de decisões do modelo

### Observabilidade
- **Prometheus** coletando métricas da API em `/metrics`
- **Grafana** para dashboards

---

## 1.3 Fluxo de ponta a ponta
1) Cliente/sistema chama `/predict` ou `/predict_batch`
2) API valida payload (Pydantic), normaliza tipos numéricos e constrói DataFrame
3) API chama modelo carregado (MLflow PyFunc) e calcula `y_pred_proba` e `y_pred` via threshold
4) API retorna resposta
5) Em background, a API grava predições no Postgres (`churn_predictions`)
6) Métricas e logs são emitidos (latência, status, etc.)

---

## 1.4 Fluxo de batch (suporte pós‑deploy)
### Drift sem rótulo - data drift
- Job `drift.py`:
  - lê baseline (ex: `baseline.json`)
  - lê amostra recente de `churn_predictions` do Postgres por janela (`window_days`)
  - calcula PSI por feature e PSI do score (`y_pred_proba`)
  - persiste resultado em tabela `drift_metrics`

### Delayed labels
- Em churn, o “rótulo” cliente churnou ou não geralmente é conhecido **após um período**. Portanto, a medição de performance real deve ser batch:
  - job junta predições históricas + tabela de churn real
  - calcula métricas por janela
  - decide retreino/promoção/rollback

---

## 1.5 Justificativa técnica
### Por que real‑time?
- Permite uso em tempo de decisão (CRM, atendimento, retenção) com baixa latência.
- Garante que a predição considere o estado mais recente do cliente.
- A API permite também batch pequeno (`/predict_batch`) sem necessidade de pipeline separado para volumes moderados.

### Por que batch complementar?
- Drift e avaliação com rótulo atrasado **não dependem de resposta imediata**.
- Jobs batch são mais baratos e simples para tarefas periódicas (monitoramento, relatórios, recalibração, retreino).
- Se a base de clientes crescer, o scoring completo mensal pode migrar para um job batch dedicado (e a API fica para consultas on-demand).

### Trade-off latência vs custo
- Online exige disponibilidade contínua, monitoramento e controle de latência.
- Batch reduz custo e complexidade para tarefas que não precisam de resposta imediata.

---
