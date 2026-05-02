import numpy as np
import pandas as pd

from src.core.models.trainer import ChurnModelTrainer


class FakePyfuncModel:
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns:
            Retorna probabilidades determinísticas para testar threshold
        """
        return pd.DataFrame({"y_pred_proba": np.linspace(0.0, 1.0, len(X))})


def test_trainer_predict_with_pyfunc_dataframe(X_example) -> None:
    """
    Checa se o método predict do ChurnModelTrainer funciona corretamente quando o modelo final é um pyfunc que retorna um DataFrame
    com a coluna 'y_pred_proba'. O teste faz:
        - Injeta um modelo pyfunc fake que retorna probabilidades lineares de 0 a 1.
        - Roda o método predict do trainer com um exemplo de input.
        - Verifica se as formas dos arrays de predição estão corretas, se as classes

    Args:
        X_example: fixture que retorna um exemplo de input para o modelo (DataFrame)
    """
    trainer = ChurnModelTrainer()
    trainer.final_model = FakePyfuncModel()

    y_pred, y_proba = trainer.predict(X_example, threshold=0.5, proba_col="y_pred_proba")

    assert y_pred.shape == (len(X_example),)
    assert y_proba.shape == (len(X_example),)

    assert set(np.unique(y_pred)).issubset({0, 1})
    assert (y_pred == (y_proba >= 0.5).astype(int)).all()
