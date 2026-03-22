import pandas as pd
import numpy as np

def get_null_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """Retorna listas de colunas com valores nulos, separadas por tipo.

    Args:
        df: Dataframe a ser transformado.

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

    Args:
        df: Dataframe a ser transformado.
        cat_cols_null: Features categóricas a serem transformadas.
        num_cols_null: Features numéricas a serem transformadas.
        bool_cols_null: Features booleanas a serem transformadas.
    
    Returns:
        Dataframe com a transformação final
    """

    for col in num_cols_null:
        if col in df.columns:
            median = df[col].median()
            df[col].fillna(median, inplace=True)

    all_cat_boll_cols = cat_cols_null + bool_cols_null

    for col in all_cat_boll_cols:
        if col in df.columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

    return df


def one_hot_encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    drop_first: bool = False,
    dtype: str | type = "int64",
) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding nas colunas categóricas informadas.
    
    Args:
        df: DataFrame de entrada.
        categorical_cols: lista de colunas categóricas para transformar.
        drop_first: se True, remove a primeira categoria (evita multicolinearidade).
        dtype: tipo dos dummies (ex.: "int64" ou bool).
    
    Returns:
        DataFrame com as colunas categóricas transformadas em dummies.
    """
    df_out = df.copy()

    categorical_cols = [c for c in categorical_cols if c in df_out.columns]

    df_out = pd.get_dummies(
        df_out,
        columns=categorical_cols,
        drop_first=drop_first,
        dummy_na=False,
        dtype=dtype,
    )

    return df_out
