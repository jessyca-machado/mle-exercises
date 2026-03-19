import pandas as pd
import numpy as np

def get_null_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """Retorna listas de colunas com valores nulos, separadas por tipo.

    Returns:
        Tupla(cat_cols_null, num_cols_null, bool_cols_null)
    """

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    bool_cols = df.select_dtypes(include='bool').columns

    cat_cols_null = [col for col in cat_cols if df[col].isnull().sum() > 0]
    num_cols_null = [col for col in num_cols if df[col].isnull().sum() > 0]
    bool_cols_null = [col for col in bool_cols if df[col].isnull().sum() > 0]

    return cat_cols_null, num_cols_null, bool_cols_null

def impute_missing(
        df: pd.DataFrame,
        cat_cols_null: list,
        num_cols_null: list,
        bool_cols_null: list
) -> pd.DataFrame:
    """Imputa valores faltantes: median para numéricas, mode para categóricas e booleanas.
    
    Returns:
        Dataframe com a transformação final
    """

    # Numéricas
    for col in num_cols_null:
        if col in df.columns:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    all_cat_boll_cols = cat_cols_null + bool_cols_null

    # Categóricas
    for col in all_cat_boll_cols:
        if col in df.columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

    return df
