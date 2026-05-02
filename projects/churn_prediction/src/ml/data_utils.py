"""
Utilitários para carregamento, pré-processamento e avaliação de dados de churn.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,  
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import (
    FEATURES_COLS,
    YES_NO_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_SEED,
    BOL_COLS,
    BIN_COLS,
    CAT_COLS,
    NUM_COLS
)


@dataclass
class ExistingColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Seleciona, no fit, a interseção entre uma lista desejada e as colunas
    presentes no DataFrame. No transform, retorna apenas essas colunas (na mesma ordem).
    """
    columns: Sequence[str]

    def fit(self, X: pd.DataFrame, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("ExistingColumnsSelector espera um pandas DataFrame como X.")
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame):
        return X.loc[:, self.columns_]


def load_and_split_churn() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carrega os dados de churn, realiza o pré-processamento e divide em conjuntos de treino e teste.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")
    X = df_clean[FEATURES_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if y.nunique() == 2 else None,
    )

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Calcula um conjunto de métricas de classificação binária.

    Args:
        y_true: Array com rótulos verdadeiros (0/1).
        y_pred: Array com predições binárias (0/1).
        y_prob: Array com probabilidades/scores da classe positiva.

    Returns:
        Dicionário com métricas:
            - accuracy
            - precision
            - recall
            - f1
            - roc_auc
            - average_precision
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }


def cat_selector(X: pd.DataFrame) -> list[str]:
    """
    Seleciona colunas categóricas presentes no DataFrame.
    """
    return [c for c in CAT_COLS if c in X.columns]


def num_selector(X: pd.DataFrame) -> list[str]:
    """
    Seleciona colunas numéricas e binadas presentes no DataFrame, excluindo as categóricas.
    """
    scaled_cols = [c for c in (NUM_COLS + BIN_COLS) if c not in set(CAT_COLS)]
    return [c for c in scaled_cols if c in X.columns]


def bol_selector(X: pd.DataFrame) -> list[str]:
    """
    Seleciona colunas booleanas presentes no DataFrame, excluindo as categóricas e as numéricas/binadas.
    """
    scaled_cols = [c for c in (NUM_COLS + BIN_COLS) if c not in set(CAT_COLS)]
    bol_cols = [c for c in BOL_COLS if c not in set(CAT_COLS) and c not in set(scaled_cols)]
    return [c for c in bol_cols if c in X.columns]


def build_preprocessor() -> ColumnTransformer:
    """
    Constrói um `ColumnTransformer` a partir das colunas presentes no DataFrame.

    As colunas são separadas em grupos:
    - categóricas: OneHotEncoder
    - numéricas (+ BIN_COLS): StandardScaler
    - booleanas: passthrough

    Returns:
        Um `ColumnTransformer` configurado para preprocessar as colunas disponíveis.
    """
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    bol_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("passthrough", "passthrough"),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_selector),
            ("num", num_pipe, num_selector),
            ("bol", bol_pipe, bol_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
