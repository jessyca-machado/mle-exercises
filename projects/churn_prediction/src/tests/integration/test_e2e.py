import mlflow
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.core.models.trainer import ChurnModelTrainer
from src.jobs.train import log_xgb_end_to_end_pyfunc

@pytest.mark.integration
def test_e2e_train_log_load_predict(tmp_path, X_y, request) -> None:
    """
    PIpeline end-to-end de treino, log e predict. O teste faz o seguinte:
        - treina pipeline
        - loga como pyfunc
        - carrega o modelo pelo MLflow
        - roda predict no input_example

    Args:
        tmp_path: fixture do pytest para criar um diretório temporário
        X_y: fixture que retorna os dados de treino (X, y)
        request: fixture do pytest para acessar informações do teste, como o caminho do projeto
    """
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("e2e-exp")

    X, y = X_y

    trainer = ChurnModelTrainer(n_folds=2, seed=42)
    trainer.build(X, y, LogisticRegression(max_iter=200))
    summary = trainer.train()

    input_example = X.head(6).copy()
    int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns
    if len(int_cols) > 0:
        input_example[int_cols] = input_example[int_cols].astype("float64")

    root = request.config.rootpath
    pyfunc_path = (root / "src" / "ml" / "churn_pyfunc_xgb.py").resolve()
    assert pyfunc_path.exists()

    with mlflow.start_run():
        model_uri = log_xgb_end_to_end_pyfunc(
            fitted_pipeline=summary.fitted_pipeline,
            name="e2e_model",
            pyfunc_code_path=str(pyfunc_path),
            pip_requirements=None,
            input_example=input_example,
        )

    loaded = mlflow.pyfunc.load_model(model_uri)
    out = loaded.predict(input_example)

    if isinstance(out, pd.DataFrame):
        assert "y_pred_proba" in out.columns
        assert len(out) == len(input_example)
    else:
        assert len(out) == len(input_example)
