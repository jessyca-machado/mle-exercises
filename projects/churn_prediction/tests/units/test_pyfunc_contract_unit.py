import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.data.feature_engineering import TelcoFeatureEngineeringBins
from src.ml.data_utils import build_preprocessor
from src.ml.churn_pyfunc_xgb import ChurnModelXGB


def test_pyfunc_contract_returns_dataframe_with_y_pred_proba(X_y, X_example) -> None:
    """
    Testa se o contrato do pyfunc é respeitado, ou seja, se a função predict retorna um DataFrame com a coluna 'y_pred_proba'
    contendo as probabilidades previstas, e se essas probabilidades estão no intervalo [0, 1].

    Args:
        X_y: fixture que retorna os dados de treino (X, y)
        X_example: fixture que retorna um exemplo de input para o modelo (DataFrame)
    """
    X, y = X_y

    fe = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    pre = build_preprocessor()
    selector = SelectKBest(score_func=mutual_info_classif, k="all")

    X_fe = fe.fit_transform(X)
    X_pp = pre.fit_transform(X_fe)
    X_sel = selector.fit_transform(X_pp, y)

    est = LogisticRegression(max_iter=200).fit(X_sel, y)

    pyfunc = ChurnModelXGB()
    pyfunc.feature_engineering = fe
    pyfunc.preprocessor = pre
    pyfunc.selector = selector
    pyfunc.model = est

    out = pyfunc.predict(context=None, model_input=X_example)

    assert isinstance(out, pd.DataFrame)
    assert "y_pred_proba" in out.columns
    assert len(out) == len(X_example)

    probs = out["y_pred_proba"].to_numpy()
    assert probs.dtype.kind in {"f"}  # float
    assert np.all((probs >= 0.0) & (probs <= 1.0))
