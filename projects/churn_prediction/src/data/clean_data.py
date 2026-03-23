import logging
import pandas as pd
import numpy as np

from src.data.load_data import load_data_churn
from src.utils.helpers import get_null_columns, impute_missing, one_hot_encode_categoricals
from src.utils.constants import FEATURES_COLS, YES_NO_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def clean_data(
        df: pd.DataFrame,
        features: list,
        yes_no_cols: list,
        stage_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega, pré-processa o dataset Telco Customer Churn da IBM e faz a transformação dos dados.

    Aplica imputação simples e encoding de variáveis categóricas.

    Args:
        df: Dataframe a ser transformado.
        features: Features de treino.
        yes_no_cols: Features categorizadas em yes/no e que serão transformadas em tipagem inteiro no formato 1/0.
        stage_name: Nome do estágio para logging.

    Returns:
        Tupla(df_clean, X, y) com dataframe limpo, features e target.
    """
    df_clean = df.copy()

    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    df_clean = df_clean.dropna(subset=["customerID"])

    for col in yes_no_cols:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    cat_cols_null, num_cols_null, bool_cols_null = get_null_columns(df_clean)

    df_clean = impute_missing(df_clean, cat_cols_null, num_cols_null, bool_cols_null)

    X_raw = df_clean[features].copy()
    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    X = one_hot_encode_categoricals(X_raw, categorical_cols, drop_first=True)

    y = df_clean["Churn"]

    logger.info("\n--- %s ---", stage_name)
    logger.info("First 10 rows of df_clean:\n%s", df_clean.head(10).to_string(index=False))

    logger.info("X shape: %s", X.shape)
    logger.info("X first 10 rows:\n%s", X.head(10).to_string(index=False))

    logger.info("y shape: %s", y.shape)
    logger.info("y first 10 values:\n%s", y.head(10).to_string(index=False))
    
    return df_clean, X, y


def main() -> None:
    df = load_data_churn()

    df_clean, X, y = clean_data(df, FEATURES_COLS, YES_NO_COLS, "Cleaned dataset and features")


if __name__ == "__main__":
    main()
