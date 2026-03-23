import pandas as pd
import numpy as np


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


def log_class_distribution(
        y: pd.Series,
        name: str
) -> Tuple[str, pd.Series, pd.Series]:
    """
    Calcula a distribuição de classes de uma série (target), retornando contagens
    absolutas e proporções (frequências relativas).

    Args:
        y: Série com os rótulos/classes (pode conter NaN).
        name: Nome do conjunto (ex.: "train", "test" ou nome da variável).

    Returns:
        Tupla (name, counts, ratio), onde:
            - name: o mesmo identificador fornecido.
            - counts: contagem absoluta por classe (inclui NaN se existir).
            - ratio: proporção por classe (normalizada para somar 1), arredondada em 4 casas (inclui NaN se existir).
    """

    counts = y.value_counts(dropna=False)
    ratios = y.value_counts(normalize=True, dropna=False).round(4)

    return name, counts, ratios
