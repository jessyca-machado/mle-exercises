# tests/test_trainer_pipeline.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.core.models.trainer import ChurnModelTrainer

def test_trainer_train_and_predict(X_y) -> None:
    """
    Testa a pipeline de treino e predição do ChurnModelTrainer. O teste faz o seguinte:
        - treina o modelo usando os dados de treino (X, y)
        - verifica se as métricas de avaliação estão presentes no resumo do treino
        - roda a predição no mesmo conjunto de treino e verifica se as formas dos arrays de predição estão corretas e se as classes previstas são 0 ou 1.

    Args:
        X_y: fixture que retorna os dados de treino (X, y)
    """
    X, y = X_y
    trainer = ChurnModelTrainer(n_folds=2, seed=42)

    trainer.build(X, y, LogisticRegression(max_iter=200))
    summary = trainer.train()

    assert "recall" in summary.metrics_mean
    assert trainer.final_model is not None

    y_pred, y_proba = trainer.predict(X)
    assert y_pred.shape[0] == len(X)
    assert y_proba.shape[0] == len(X)
    assert set(np.unique(y_pred)).issubset({0, 1})
