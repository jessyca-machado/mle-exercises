import pandas as pd
import numpy as np

from utils.helpers import get_null_columns, impute_missing
from utils.constants import URL, FEATURES_COLS, YES_NO_COLS


def clean_data(
        df: pd.DataFrame,
        features: list,
        yes_no_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega, pré-processa o dataset Telco Customer Churn da IBM e faz a transformação dos dados.

    Aplica imputação simples e encoding de variáveis categóricas.

    Returns:
        Tupla(df_clean, X, y) com dataframe limpo, features e target.
    """

    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=["customerID"])

    for col in yes_no_cols:
        if df_clean[col].dtype == "object":
            df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Identifica colunas com valores nulos
    cat_cols_null, num_cols_null, bool_cols_null = get_null_columns(df_clean)

    # Imputa missing values
    df_clean = impute_missing(df_clean, cat_cols_null, num_cols_null, bool_cols_null)

    X = df_clean[features]
    y = df_clean["Churn"]
    
    return df_clean, X, y
