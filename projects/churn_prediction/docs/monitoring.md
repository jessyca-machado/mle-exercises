
---

# 2) Plano de monitoramento - métricas, alertas e playbook

O plano é dividido em:
  **(A) saúde do serviço**,
  **(B) saúde do modelo sem rótulo**,
  **(C) saúde do modelo com rótulo atrasado**,
  **(D) governança/ações**.

## 2.1 Métricas a monitorar

### A) Métricas do serviço (API)
Coletadas via Prometheus:
- **Throughput**
  - `http_requests_total{path,method,status_code}`
- **Latência HTTP**
  - `http_request_latency_seconds_bucket` (p50/p95/p99 via histogram)
- **Taxa de erro**
  - % de respostas 5xx e 4xx
- **Readiness**
  - `/ready` retornando 200 no modelo carregado

**Objetivo:** garantir disponibilidade e SLA.

---

### B) Métricas do modelo sem rótulo - sem ground truth
1) **Latência de inferência**
   - `model_inference_latency_seconds_bucket{path}`
2) **Distribuição/volume de classes previstas**
   - `predictions_total{path,predicted_class}`
   - histograma de `y_pred_proba`
3) **Drift de dados (PSI)**
   - calculado em batch por `drift.py` e persistido em `drift_metrics`
   - PSI de features numéricas (ex.: tenure, MonthlyCharges, TotalCharges)
   - PSI do score (`y_pred_proba`) como proxy de mudança comportamental
4) **Qualidade do input (sanidade)**
   - taxa de payload inválido (422)
   - taxa de coerção para NaN / missing quando implementado

**Objetivo:** detectar mudanças na distribuição dos dados e comportamento do score.

---

### C) Métricas do modelo com rótulo atrasado
Como churn é atrasado, medir em batch mensal:
   - AUC-ROC, PR-AUC
   - F1, recall, precision, com threshold atual
   - taxa de churn real vs taxa prevista
   - estabilidade/calibração

**Objetivo:** detectar degradação real de performance e necessidade de retreino.

---

## 2.2 Alertas recomendados

### A) Alertas do serviço
- **API 5xx rate alto**
  - gatilho: `5xx > 1%` por 5–10 min
- **Latência p95 alta**
  - gatilho: p95 > 300ms por 10 min
- **Readiness falhando**
  - `/ready` != 200 por X tentativas

### B) Alertas de drift, sem rótulo
- **PSI alto (moderado/alto)**
  - `max_psi >= 0.1` → alerta
  - `max_psi >= 0.2` → crítico
- **Baixo volume (não alertar)**
  - `n_rows < DRIFT_MIN_ROWS` → sem drift calculado não alerta

### C) Alertas de performance, quando houver label
- queda de recall além de X% por janela
- aumento de falsos positivos

---

## 2.3 Playbook de resposta - o que fazer quando alerta dispara

### Incidente 1 — API instável (5xx /ready falhando)
1) Checar logs da API (stacktrace)
2) Checar conectividade com MLflow/MinIO/Postgres
3) Se falha ao carregar modelo:
   - validar `CHURN_MODEL_URI`
   - se necessário, rollback para versão anterior do modelo
4) Se é problema de dependência/ambiente:
   - corrigir imagem e redeploy

**Resultado esperado:** API volta a 200 e /ready ok.

---

### Incidente 2 — Latência alta (p95 acima do limite)
1) Checar uso de CPU/RAM do container
2) Verificar se persistência no Postgres está degradando
3) Mitigações:
   - aumentar workers
   - mover insert para fila
   - reduzir payload batch size recomendado

---

### Incidente 3 — Drift alto (PSI >= 0.2)
1) Confirmar se `n_rows` suficiente, não é efeito de amostra pequena
2) Investigar quais features mais mudaram (PSI por feature)
3) Verificar mudanças na origem dos dados (schema, categorias, regras de negócio)
4) Ações:
   - se drift por bug/pipeline: corrigir ETL/validação e reprocessar
   - se drift real de comportamento:
     - agendar retreino
     - comparar métricas offline
     - promover modelo novo no registry

---

### Incidente 4 — Performance real caiu, quando houver rótulo
1) Confirmar janela/segmentos afetados
2) Avaliar retreino com dados mais novos
3) Reavaliar threshold (otimizar recall)
4) Se necessário rollback para versão anterior

---

## 2.4 Estratégia de manutenção / retreino
- **Retreino por agenda** mensal E
- **Retreino por gatilho**
  - drift alto sustentado: PSI > 0.2 por 3 execuções
  - queda de performance com label

---
