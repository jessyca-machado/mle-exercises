from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import skops.io as sio

from src.utils.constants import TRUSTED_TYPES


class ChurnModelXGB(mlflow.pyfunc.PythonModel):
    """
    PyFunc end-to-end (feature engineering -> preprocess -> select -> XGBoost).

    Contrato de 1/O:
        - Entrada: pd.DataFrame cru
        - Saída: pd.DataFrame com coluna `y_pred_proba`
    """

    def load_context(self, context):
        fe_path = context.artifacts["feature_engineering"]
        preproc_path = context.artifacts["preprocessor"]
        selector_path = context.artifacts["selector"]
        model_path = context.artifacts["estimator"]
        self.feature_engineering = sio.load(fe_path, trusted=TRUSTED_TYPES)
        self.preprocessor = sio.load(preproc_path, trusted=TRUSTED_TYPES)
        self.selector = sio.load(selector_path, trusted=TRUSTED_TYPES)
        self.model = sio.load(model_path, trusted=TRUSTED_TYPES)

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna probabilidade de churn (classe positiva).

        Args:
            context (Any): Contexto do MLflow, não utilizado aqui.
            model_input (pd.DataFrame): DataFrame de entrada, sem as colunas de features
                engineering.

        Returns:
            pd.DataFrame: DataFrame com uma coluna `y_pred_proba` contendo as probabilidades de
                churn.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X_fe = self.feature_engineering.transform(model_input)
        X_pp = self.preprocessor.transform(X_fe).astype(np.float32)
        X_sel = self.selector.transform(X_pp).astype(np.float32)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_sel)[:, 1].astype(float)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_sel)
            proba = (1.0 / (1.0 + np.exp(-scores))).astype(float)
        else:
            pred = self.model.predict(X_sel)
            proba = np.asarray(pred, dtype=float)

        return pd.DataFrame({"y_pred_proba": proba})


mlflow.models.set_model(ChurnModelXGB())
