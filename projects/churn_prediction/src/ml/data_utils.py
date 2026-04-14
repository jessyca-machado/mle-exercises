from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import (
    FEATURES_COLS, YES_NO_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE
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
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() == 2 else None,
    )

    return X_train, X_test, y_train, y_test
