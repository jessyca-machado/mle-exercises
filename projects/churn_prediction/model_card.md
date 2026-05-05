---
language:
  - pt
license: mit
library_name: xgboost
pipeline_tag: tabular-classification
tags:
  - sklearn
  - torch
  - churn
  - tabular-classification
  - mlflow
  - fastapi
  - postgresql
  - minio
  - redis
  - mlp
  - xgboost
metrics:
  - recall
  - precision
  - f1
  - accuracy
  - roc_auc
  - average_precision
  - net_value
  - roi
datasets:
  - Telco CUstomer Churn IBM
---

# Model Card — Churn XGBoost

## Model Details
- **Versão:** 1.0.0
- **Model name (MLflow Registry):** `churn_xgb`
- **Model URI (exemplo):** `models:/churn_xgb/2`
- **Tipo:** Classificação binária (churn: 1 / não churn: 0)
- **Formato de serving:** **MLflow PyFunc** (end-to-end pipeline)
- **Algoritmo principal:** XGBoost (com pré-processamento + modelo empacotado)
- **Versão de Python (serving):** **Python 3.12.x**
- **Treinado e logado por:** _jessycamachado0@gmail.com
- **Data do treino/log:** 2026-05-08
- **Licença:** MIT
- **Contato:** jessycamachado0@gmail.com

| Key | Value |
|-----|-------|
| Registered name | `churn_xgb` |
| Model URI | `models:/churn_xgb/2` |
| Task | Binary classification (churn) |
| Base model | XGBoost (`xgboost.XGBClassifier`) |
| Serving format | MLflow PyFunc (end-to-end pipeline) |
| Preprocessing | Sklearn pipeline (ex.: OHE para categóricas + imputação + scaling, se aplicável) |
| Input schema | Tabular JSON → pandas.DataFrame (19 features) |
| Output | `y_pred` (0/1) and `y_pred_proba` (float) |
| Default threshold | `CHURN_THRESHOLD=0.5` |
| Python (model logged with) | Python 3.12.x (ex.: 3.12.13) |
| MLflow | Tracking + Registry (via `MLFLOW_TRACKING_URI` / `MLFLOW_REGISTRY_URI`) |
| Artifact store | MinIO (S3-compatible) |
| Rate limiting | SlowAPI + Redis (Docker) / `memory://` (host tests) |

## Intended Use
### Uso primário
Estimar a **probabilidade de churn** para clientes ativos, independente do mês de entrada na empresa tenure >= 0, suportando ações de retenção (CRM, Customer Success).

### Usuários pretendidos
- Time de retenção / CRM
- Analistas de dados e ML engineers (monitoramento)

### Out-of-scope
- Decisões automáticas punitivas (bloqueio/cobrança) sem revisão humana
- Decisões de crédito/score
- Uso fora do domínio do dataset original sem reavaliação/ transferência de domínio

## Data & Features
### Tipos de features (exemplo)
- **Demográficas:** `gender`, `SeniorCitizen`, etc.
- **Contrato/serviço:** `Contract`, `InternetService`, `TechSupport`, etc.
- **Cobrança/uso:** `MonthlyCharges`, `TotalCharges`, `tenure`

## Metrics
As métricas abaixo devem ser reportadas no conjunto de validação/teste:
- **Recall:** capacidade de identificar corretamente as instâncias positivas. Essa é a métrica utilizada como alvo no estudo, pirorizar o modelo que melhor retorna o recall na vailidação cruzada estratificada
- **ROC-AUC:** capacidade discriminativa geral
- **Average Precision / PR-AUC:** prioritária em caso de desbalanceamento. Foi inicialmente relevante no estudo mas substituída pela recall depois.
- **net_value:** o valor líquido de cada modelo no cálculo do trade-off de custo. Custo falso positivo = 1.0, custo falso negativo = 5.0, benefício verdadeiro positivo = 10.0
- **roi:** retorno sobre investimento. Utilizado como métrica secundária no cálculo do trade-off de custo

## Evaluation Data
- Dataset, split estratificado com StratifiedKFold(treino: 80%; teste: 20%).
- Pre-processamento:
  - Conversão e coerção de numéricos (ex.: `TotalCharges` pode chegar como string/blank e é convertido para float quando possível)
  - imputação de média, mediana ou moda em dados faltantes
  - transformação das variávelis binárias YES/NO para 1/0
  - feature engineering das variáveis
  - One-Hot nas variáveis categóricas
  - StandardScale nas variáveis numéricas
  - Seleção das melhores features com SelectKBest
- **Validação:** utilização do RandomizedSearchCV com 10 folds para a validação do melhor modelo

## Training Data
- Utilização dos resultados registrados no MLFlow
- Mesmo dataset, mesma transformação, tuning do melhor modelo com os parâmetros do RandomizedSearchCV

### Config
```yaml
model__colsample_bytree: 0.908897907718663
model__gamma: 0.0008585370205937364
model__learning_rate: 0.0010296901472345186
model__max_depth: 4
model__min_child_weight: 0.28677569187725055
model__n_estimators: 591
model__reg_alpha: 0.8241925264876453
model__reg_lambda: 7.126985539523763
model__scale_pos_weight: 6.0
model__subsample: 0.9705203514053395
select_kbest_k: 20
```

## Ethical Considerations
- Ações de retenção podem afetar usuários de maneira desigual se houver viés em features sensíveis/proxies.
- Evitar uso indevido do modelo para decisões punitivas.
- Revisar periodicamente drift e performance por subgrupos.

## Caveats and Recommendations
- Reavaliar o threshold quando a matriz de custos do negócio mudar.
- Recomenda-se re-treino periódico mensal ou por critérios de drift (PSI/KS). Foi implementado o critério de drift em `/src/jobs/drift.py`

## Infrastructure Observations (Compose)
O ambiente de desenvolvimento/integração utiliza:
- MLflow Tracking/Registry via mlflow-proxy
- MinIO como backend S3 para artifacts
- PostgreSQL para persistência, inclui storage de predictions, se habilitado
- Redis para rate limiting (SlowAPI) no modo Docker

## Limitations
- Generalização limitada ao dataset/domínio: o modelo tende a funcionar bem apenas para clientes com perfil e distribuição parecidos com o dataset usado no treino. Mudanças de produto, preço, campanha, canal ou política podem degradar performance
- Sensibilidade a drift: features como MonthlyCharges, TotalCharges, Contract e tenure mudam com o tempo; sem monitoramento, o modelo pode ficar desatualizado
- Dependência do schema: a API espera um conjunto específico de campos. Mudanças no schema (novas categorias, campos faltantes) podem quebrar ou reduzir qualidade
- Rate limiting depende de Redis: fora do Docker, precisa configurar RATE_LIMIT_STORAGE_URI=memory:// ou apontar para um Redis real. Em memory://, o rate limit:
  - não é compartilhado entre processos/replicas
  - se perde ao reiniciar o processo
- API key estática: X-API-Key é simples, mas:
  - não tem expiração/rotacionamento automático
  - não identifica usuário (sem auditoria por cliente)
  - se vazar, dá acesso total aos endpoints protegidos
  - Ideal evoluir para JWT/OAuth2 no futuro
