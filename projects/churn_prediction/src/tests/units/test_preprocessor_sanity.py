import numpy as np
from src.data.feature_engineering import TelcoFeatureEngineeringBins
from src.ml.data_utils import build_preprocessor


def test_feature_engineering_and_preprocessor_fit_transform(X_y) -> None:
    """
    Testa se a combinação de feature engineering e pré-processamento funciona sem erros e retorna um array numérico
    adequado para treino de modelos. O teste faz o seguinte:
        - Aplica a feature engineering nas colunas.
        - Aplica o pré-processamento, tratamento de colunas yes/no, encoding, etc.
        - Verifica se o resultado é um array numérico sem valores infinitos ou NaN, e se o número de linhas é o mesmo do input.

    Args:
        X_y: fixture que retorna os dados de treino (X, y)
    """
    X, y = X_y

    fe = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    X_fe = fe.fit_transform(X)

    pre = build_preprocessor()
    X_pp = pre.fit_transform(X_fe)

    assert X_pp.shape[0] == X.shape[0]
    assert np.isfinite(X_pp).all()
