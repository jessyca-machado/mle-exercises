"""
Pipelines - Preparação do dataset para treino e teste com:
A classe SklearnPipelineRunner é um orquestrador de pipeline, que roda para diferentes modelos, que centraliza o fluxo de:
    - Montar um pipeline (pré-processamento por tipo de coluna, opcional feature engineering, opcional seleção de features, conversão para float32 e o modelo);
    - Treinar esse pipeline (fit), com opção de tuning via GridSearchCV;
    - Utilizar o modelo treinado para predict (previsão binária) e predict_proba (probabilidade da previsão binária);
    - Avaliar no teste com métricas prontas (evaluate, incluindo AUC quando possível);
    - Rodar validação cruzada (cross_validate) nos dados de treino;
    - Persistir e recarregar o melhor estimador (save/load).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# modelo campeão (sklearn API)
from xgboost import XGBClassifier

# suas constantes
from src.utils.constants import (
    CAT_COLS,
    NUM_COLS,
    BOL_COLS,
    BIN_COLS,
    RANDOM_SEED,
)

# -----------------------------
# Config de produção
# -----------------------------
@dataclass(frozen=True)
class ChurnModelConfig:
    """
    Config do modelo para produção.

    threshold:
        - usado apenas para decisão binária (predict_label)
        - predict_proba sempre retorna probabilidade
    """
    threshold: float = 0.05  # ajuste conforme seu cost_toolkit (best_thr do sweep)
    random_state: int = RANDOM_SEED


def build_preprocessor_from_df(df: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o preprocessor usando apenas colunas existentes no DataFrame (robusto para APIs).
    Mantém a mesma lógica dos seus treinos.
    """
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in NUM_COLS if c in df.columns]
    bol = [c for c in BOL_COLS if c in df.columns]
    bin_cols = [c for c in BIN_COLS if c in df.columns]

    # escalar numéricas + binárias (se binárias forem 0/1, scaling é opcional, mas mantém consistência)
    scaled_cols = num + bin_cols
    num = [c for c in scaled_cols if c not in cat]
    bol = [c for c in bol if c not in cat and c not in num]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
            ("bol", "passthrough", bol),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_xgboost_best_estimator(config: ChurnModelConfig) -> XGBClassifier:
    """
    Hiperparâmetros do seu melhor run (ajuste conforme o best_params que você logou).
    Você mostrou algo como:
        colsample_bytree=0.8, learning_rate=0.03, max_depth=3, n_estimators=300,
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        colsample_bytree=0.8,
        subsample=0.8             # se você tiver no best_params, use o valor correto
        reg_lambda=5.0,             # idem
        objective="binary:logistic",
        eval_metric="aucpr",        # otimiza PR-AUC durante treino
        n_jobs=-1,
        random_state=config.random_state,
    )


def build_churn_pipeline(example_df: pd.DataFrame, config: Optional[ChurnModelConfig] = None) -> Pipeline:
    """
    Monta pipeline preprocess + modelo.
    example_df é usado apenas para inferir colunas presentes (para o preprocessor).
    """
    config = config or ChurnModelConfig()
    preprocessor = build_preprocessor_from_df(example_df)
    model = build_xgboost_best_estimator(config)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# -----------------------------
# Helpers de inferência (API)
# -----------------------------
def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Retorna probabilidade de churn (classe positiva=1).
    """
    proba = pipe.predict_proba(X)[:, 1]
    return proba.astype(float)


def predict_label(pipe: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    """
    Retorna rótulo binário usando um threshold (decisão operacional).
    """
    proba = predict_proba(pipe, X)
    return (proba >= threshold).astype(int)


def validate_payload_columns(X: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Payload inválido: faltando colunas: {missing}")
