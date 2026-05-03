import mlflow
import pandas as pd
import pytest

from src.infra.mlflow.params import fetch_best_xgb_params_from_mlflow


def test_fetch_best_xgb_params_from_mlflow_parses_model_and_kbest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Procurar o melhor resultado de uma busca randomizada no MLflow e verificar se os parâmetros do
    modelo XGBoost e do SelectKBest são corretamente extraídos e convertidos para os tipos
    adequados.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture do pytest para mock de atributos e métodos.
    """

    class FakeExp:
        experiment_id = "123"

    class FakeClient:
        def get_experiment_by_name(self, name) -> FakeExp:
            return FakeExp()

    df = pd.DataFrame(
        [
            {
                "run_id": "RUN123",
                "metrics.best_cv_score": 0.97,
                "params.search_type": "randomized",
                "params.model__max_depth": "4",
                "params.model__n_estimators": "200",
                "params.model__learning_rate": "0.05",
                "params.select_kbest__k": "15",
            }
        ]
    )

    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: df)

    best = fetch_best_xgb_params_from_mlflow(
        experiment_name="exp",
        tracking_uri="sqlite:///fake.db",
        metric_key="best_cv_score",
        search_type_value="randomized",
        client=FakeClient(),
    )

    assert best.run_id == "RUN123"
    assert abs(best.best_cv_score - 0.97) < 1e-9

    assert best.xgb_params["max_depth"] == 4
    assert best.xgb_params["n_estimators"] == 200
    assert abs(best.xgb_params["learning_rate"] - 0.05) < 1e-12

    assert best.select_kbest_k == 15
