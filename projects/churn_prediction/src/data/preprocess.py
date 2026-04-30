"""Pre processamento — transformação dos dados nulos, vazios ou em branco por meio de métodos estatísticos.

- Identifica colunas nulas/vazias por tipo.
- Imputa mediana/moda para valores faltantes.
- Pré-processa dataset (conversões, mapeamentos, imputação, logging).

Uso:
    python src/data/preprocess.py

"""
import logging
from statistics import median
import pandas as pd
import numpy as np

from src.data.load_data import load_data_churn
from src.utils.constants import YES_NO_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_null_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Retorna listas de colunas com valores nulos, vazios ou em branco, separadas por tipo.

    Args:
        df: Dataframe a ser transformado.

    Returns:
        Tupla(cat_cols_null, num_cols_null, bool_cols_null)
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    cat_cols_null = []
    for col in cat_cols:
        s = df[col]
        null_mask = s.isna().to_numpy()

        arr = s.astype("string").to_numpy(dtype="U")
        blank_mask = np.char.str_len(np.char.strip(arr)) == 0

        if np.any(null_mask | blank_mask):
            cat_cols_null.append(col)

    num_cols_null = []
    for col in num_cols:
        s = df[col]
        null_mask = s.isna().to_numpy()

        arr = s.astype("string").to_numpy(dtype="U")
        blank_mask = np.char.str_len(np.char.strip(arr)) == 0

        if np.any(null_mask | blank_mask):
            num_cols_null.append(col)

    bool_cols_null = []
    for col in bool_cols:
        s = df[col]
        null_mask = s.isna().to_numpy()

        arr = s.astype("string").to_numpy(dtype="U")
        blank_mask = np.char.str_len(np.char.strip(arr)) == 0

        if np.any(null_mask | blank_mask):
            bool_cols_null.append(col)

    return cat_cols_null, num_cols_null, bool_cols_null


def impute_missing(
        df: pd.DataFrame,
        cat_cols_null: list,
        num_cols_null: list,
        bool_cols_null: list
) -> pd.DataFrame:
    """Imputa valores faltantes: median para numéricas, mode para categóricas e booleanas.

    Args:
        df: Dataframe a ser transformado.
        cat_cols_null: Features categóricas a serem transformadas.
        num_cols_null: Features numéricas a serem transformadas.
        bool_cols_null: Features booleanas a serem transformadas.
    
    Returns:
        Dataframe com a transformação final
    """
    # Regra específica para TotalCharges. Os clientes que possuem a TotalCharges não fecharam o 1º mês de faturamento, então o valor será igual ao MonthlyCharges.
    if 'TotalCharges' in num_cols_null and 'TotalCharges' in df.columns:
        if 'MonthlyCharges' in df.columns:
            mask = df['TotalCharges'].isna()
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']

    for col in num_cols_null:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    all_cat_boll_cols = cat_cols_null + bool_cols_null

    for col in all_cat_boll_cols:
        if col in df.columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

    return df


def pre_processing(
    df: pd.DataFrame,
    yes_no_cols: list,
    stage_name: str,
) -> pd.DataFrame:
    """Carrega, pré-processa o dataset Telco Customer Churn da IBM e faz a transformação dos dados.

    Aplica imputação simples e encoding de variáveis categóricas.

    Args:
        df: Dataframe a ser transformado.
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

    logger.info("\n--- %s ---", stage_name)
    logger.info("First 10 rows of df_clean:\n%s", df_clean.head(10).to_string(index=False))
    
    return df_clean


def main() -> None:
    df = load_data_churn()

    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")


if __name__ == "__main__":
    main()
