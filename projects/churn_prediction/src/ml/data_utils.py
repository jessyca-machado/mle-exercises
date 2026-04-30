from __future__ import annotations
from typing import Tuple
import pandas as pd

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

def load_and_split_churn() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carrega os dados de churn, realiza o pré-processamento e divide em conjuntos de treino e teste.
    
    Args:
        None
    
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


def build_preprocessor_from_df(df: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói um `ColumnTransformer` a partir das colunas presentes no DataFrame já após
    o feature engineering.

    As colunas são separadas em grupos:
    - categóricas: OneHotEncoder
    - numéricas (+ BIN_COLS): StandardScaler
    - booleanas: passthrough

    Args:
        df: DataFrame contendo as colunas de entrada. A função utiliza esse DataFrame
        apenas para verificar quais colunas de `CAT_COLS`, `NUM_COLS`, `BOL_COLS` e `BIN_COLS`
        existem de fato.

    Returns:
        Um `ColumnTransformer` configurado para preprocessar as colunas disponíveis.
    """
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in NUM_COLS if c in df.columns]
    bol = [c for c in BOL_COLS if c in df.columns]
    bin_cols = [c for c in BIN_COLS if c in df.columns]

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
