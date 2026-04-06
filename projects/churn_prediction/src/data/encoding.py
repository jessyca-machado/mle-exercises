"""Encoding - One-hot encoding com alinhamento treino/teste usando pandas.get_dummies.

- fit: aplica get_dummies no treino e guarda as colunas resultantes.
- transform: aplica get_dummies no teste e reindexa para as colunas do treino.
- como a variável target é altamente desbalanceada no dataset, aplica o stratify no train_test_split para manter a proporção de classes.

Uso:
    python src/data/encoding.py
"""
import logging
from typing import Iterable, Optional
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.feature_engineering import feature_engineering
from src.utils.constants import YES_NO_COLS, FEATURES_COLS, TARGET_COL


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DummiesEncoder:
    columns_: pd.Index
    categorical_cols_: list[str]
    drop_first: bool = False
    dtype: str | type = "int64"


def fit_one_hot(
    df: pd.DataFrame,
    categorical_cols: Optional[Iterable[str]] = None,
    drop_first: bool = False,
    dtype: str | type = "int64",
) -> tuple[pd.DataFrame, DummiesEncoder]:
    """
    Fit no treino: aplica get_dummies e memoriza as colunas finais.

    Args:
        df: Dataframe a ser transformado.
        categorical_cols: lista de colunas categóricas para transformar. Se None, detecta automaticamente.
        drop_first: se True, remove a primeira categoria (evita multicolinearidade).
        dtype: tipo dos dummies (ex.: "int64" ou bool).

    Returns:
        Tuple com o dataframe transformado e o encoder.
    """
    X = df.copy()

    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        categorical_cols = [c for c in categorical_cols if c in X.columns]

    X_encoder = pd.get_dummies(
        X,
        columns=list(categorical_cols),
        drop_first=drop_first,
        dummy_na=False,
        dtype=dtype,
    )

    encoder = DummiesEncoder(
        columns_=X_encoder.columns,
        categorical_cols_=list(categorical_cols),
        drop_first=drop_first,
        dtype=dtype,
    )

    return X_encoder, encoder


def transform_one_hot(df: pd.DataFrame, encoder: DummiesEncoder) -> pd.DataFrame:
    """
    Transform no teste/inferência: aplica get_dummies e alinha às colunas do treino.
    - Colunas ausentes no teste -> 0
    - Colunas extras no teste (categorias novas) -> descartadas

    Args:
        df: Dataframe a ser transformado.
        encoder: DummiesEncoder com as colunas e parâmetros do treino.

    Returns:
        Dataframe transformado e alinhado às colunas do treino.
    """
    X = df.copy()

    categorical_cols = [c for c in encoder.categorical_cols_ if c in X.columns]

    X_encoder = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=encoder.drop_first,
        dummy_na=False,
        dtype=encoder.dtype,
    )

    X_encoder = X_encoder.reindex(columns=encoder.columns_, fill_value=0)

    return X_encoder


def main() -> None:
    """
    Demonstração com o dataset real do projeto (Telco Churn):
    - carrega dados
    - pre-processa + feature engineering
    - split
    - fit dummies no treino
    - transform dummies no teste (alinhado)
    """

    logger.info("--- Demo: churn dataset + get_dummies alinhado ---")

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")
    df_feature_engineering = feature_engineering(df_clean, "Dataset after feature engineering")

    X_raw = df_feature_engineering[FEATURES_COLS].copy()
    y = df_feature_engineering[TARGET_COL].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Categóricas detectadas (no treino): %s", categorical_cols)

    X_train, encoder = fit_one_hot(
        X_train_raw,
        categorical_cols=categorical_cols,
        drop_first=True,
        dtype="int64",
    )
    X_test = transform_one_hot(X_test_raw, encoder)

    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
    logger.info("y_train shape: %s | y_test shape: %s", y_train.shape, y_test.shape)

    # Checagens úteis
    missing_in_test = set(encoder.columns_) - set(X_test.columns)
    extra_in_test = set(X_test.columns) - set(encoder.columns_)
    logger.info("Colunas do treino ausentes no teste (depois do reindex): %d", len(missing_in_test))
    logger.info("Colunas extras no teste vs treino (depois do reindex): %d", len(extra_in_test))

    logger.info("Primeiras 5 linhas (X_train):\n%s", X_train.head(5).to_string(index=False))
    logger.info("Primeiras 5 linhas (X_test):\n%s", X_test.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
