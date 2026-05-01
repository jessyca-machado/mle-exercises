import pandas as pd

def test_pre_processing_basic_properties(df_clean) -> None:
    """
    Testa as propriedades básicas do DataFrame após o pré-processamento.
    """
    assert df_clean["customerID"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])

def test_pre_processing_yes_no_cols_mapped(df_clean) -> None:
    """
    Testa se as colunas de sim/não foram mapeadas corretamente.
    """
    for col in ["Partner", "Dependents", "PaperlessBilling"]:
        if col in df_clean.columns:
            assert not df_clean[col].astype(str).isin(["Yes", "No"]).any()


def test_raw_df_fixture_is_dataframe(raw_df) -> None:
    """
    Testa se o fixture `raw_df` retorna um DataFrame.
    Também verifica se o DataFrame tem o número esperado de linhas para o teste e se contém a coluna "Churn",
    utilizada como target.
    """
    assert isinstance(raw_df, pd.DataFrame)
    assert raw_df.shape[0] == 180
    assert "Churn" in raw_df.columns


def test_preprocessing_converts_totalcharges_to_numeric(df_clean) -> None:
    """
    Verifica se a coluna "TotalCharges" foi convertida para numérica após o pré-processamento,
    e se não há valores não numéricos.
    """
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
