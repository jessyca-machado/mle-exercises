import numpy as np
import torch
import mlflow
import mlflow.pyfunc
import skops.io as sio
from typing import Any
import pandas as pd
from src.utils.constants import TRUSTED_TYPES    

class ChurnModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        fe_path = context.artifacts["feature_engineering"]
        preproc_path = context.artifacts["preprocessor"]
        selector_path = context.artifacts["selector"]
        model_path = context.artifacts["torchscript_model"]
        self.feature_engineering = sio.load(fe_path, trusted=TRUSTED_TYPES)
        self.preprocessor = sio.load(preproc_path, trusted=TRUSTED_TYPES)
        self.selector = sio.load(selector_path, trusted=TRUSTED_TYPES)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidade de churn (classe positiva).

        Args:
            context (Any): Contexto do MLflow, não utilizado aqui.
            model_input (pd.DataFrame): DataFrame de entrada, sem as colunas de features engenheiradas.
        
        Returns:
            np.ndarray: Array de probabilidades de churn para cada linha do DataFrame de entrada.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X_fe = self.feature_engineering.transform(model_input)

        X_pp = self.preprocessor.transform(X_fe).astype(np.float32)

        X_sel = self.selector.transform(X_pp).astype(np.float32)

        X_t = torch.from_numpy(X_sel)
        with torch.no_grad():
            logits = self.model(X_t).reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

mlflow.models.set_model(ChurnModel())
