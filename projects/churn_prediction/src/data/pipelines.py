"""Pipelines - Preparação do dataset para churn (treino/teste) com:

- load -> preprocess -> feature engineering -> split -> one-hot (fit no treino, transform no teste)

Uso:
    python src/data/pipelines.py
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.feature_engineering import feature_engineering
from src.data.encoding import fit_one_hot, transform_one_hot, DummiesEncoder
from src.utils.constants import FEATURES_COLS_ENG, YES_NO_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_train_test(
    features: Sequence[str],
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
    drop_first: bool = True,
    use_feature_engineering: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DummiesEncoder]:
    """Pipeline completa de preparação do dataset para churn prediction:

    Args:
        features: lista de colunas a serem usadas como features.
        target: nome da coluna target.
        test_size: proporção do dataset a ser usado como teste.
        random_state: semente para reprodutibilidade do split.
        drop_first: se True, remove a primeira categoria no one-hot (evita multicolinearidade).
        use_feature_engineering: se True, aplica a etapa de feature engineering.

    Returns:
        Dataframe com as novas features criadas.
    """
    df = load_data_churn()

    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")

    if use_feature_engineering:
        df_model = feature_engineering(df_clean, "Dataset after feature engineering")
    else:
        df_model = df_clean

    X_raw = df_model[list(features)].copy()
    y = df_model[target].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None,
    )

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, encoder = fit_one_hot(
        X_train_raw,
        categorical_cols=categorical_cols,
        drop_first=drop_first,
        dtype="int64",
    )
    X_test = transform_one_hot(X_test_raw, encoder)

    logger.info("--- Final dataset (train/test) ---")
    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
    logger.info("y_train shape: %s | y_test shape: %s", y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, encoder


def main() -> None:
    X_train, X_test, y_train, y_test, _ = prepare_train_test(FEATURES_COLS_ENG)
    logger.info("X_train first 5 rows:\n%s", X_train.head(5).to_string(index=False))
    logger.info("X_test first 5 rows:\n%s", X_test.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
