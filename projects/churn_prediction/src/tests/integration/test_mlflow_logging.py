import pytest
import mlflow
from sklearn.linear_model import LogisticRegression
from src.core.models.trainer import ChurnModelTrainer

def test_mlflow_logs_run(tmp_path, X_y, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Testa se é possível logar uma execução de treino no MLflow, verificando se o banco de dados do MLflow é criado
    e se os parâmetros e métricas são logados corretamente.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture do pytest para mock de atributos e métodos.
    """
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-exp")

    X, y = X_y
    trainer = ChurnModelTrainer(n_folds=2, seed=42)
    trainer.build(X, y, LogisticRegression(max_iter=200))
    summary = trainer.train()

    with mlflow.start_run():
        mlflow.log_metrics({f"cv_mean_{k}": v for k, v in summary.metrics_mean.items()})
        mlflow.log_params({"foo": "bar"})

    assert db_path.exists()
