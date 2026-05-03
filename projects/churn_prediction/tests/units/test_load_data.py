import pandas as pd
import pytest

from src.data import load_data
from src.utils.constants import TARGET_COL


def test_load_data_churn_returns_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Testa se a função `load_data_churn` retorna um DataFrame com a coluna de target "Churn".
    Utiliza monkeypatch para substituir a função de carregamento por uma versão fake que retorna um
        DataFrame simples.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture do pytest para mock de atributos e métodos.
    """

    def fake_loader():
        return pd.DataFrame({"customerID": ["A"], "Churn": [0]})

    monkeypatch.setattr(load_data, "load_data_churn", fake_loader)

    df = load_data.load_data_churn()
    assert isinstance(df, pd.DataFrame)
    assert TARGET_COL in df.columns
