"""Pipeline de inferência: carrega modelo do MLflow e gera predições para um DataFrame de features.
- O modelo deve ser um MLflow PyFunc que retorna um DataFrame com coluna de probabilidade (ex: "y_pred_proba").
- O código é modular para facilitar uso em batch offline, FastAPI, ou testes unitários.

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import pytz
from mlflow.tracking import MlflowClient

from src.ml.mlflow_utils import setup_mlflow_sqlite
from src.utils.constants import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.core.models.trainer import ChurnModelTrainer


@dataclass(frozen=True)
class PredictConfig:
    """
    Configuração para inferência.
    """
    registry_uri: str = MLFLOW_TRACKING_URI
    model_name: Optional[str] = None
    model_version: Optional[Union[int, str]] = None
    model_uri: Optional[str] = None
    timezone: str = "America/Sao_Paulo"
    threshold: float = 0.5
    proba_col: str = "y_pred_proba"


def get_latest_model_uri(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"Nenhuma versão encontrada para o modelo '{model_name}'.")
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{model_name}/{latest.version}"


def resolve_model_uri(config: PredictConfig, client: Optional[MlflowClient] = None) -> str:
    """
    Resolve a URI do modelo seguindo prioridade:
    1) config.model_uri
    2) config.model_name + config.model_version
    3) config.model_name + latest
    """
    if config.model_uri:
        return config.model_uri

    if not config.model_name:
        raise ValueError("Você deve fornecer `model_uri` ou `model_name`.")

    if config.model_version is not None:
        return f"models:/{config.model_name}/{config.model_version}"

    if client is None:
        client = MlflowClient()

    return get_latest_model_uri(client, config.model_name)


def load_pyfunc_model(model_uri: str, registry_uri: str):
    """
    Carrega um modelo MLflow PyFunc.
    """
    mlflow.set_registry_uri(registry_uri)
    return mlflow.pyfunc.load_model(model_uri)


def _as_1d_float_array(raw: Any) -> np.ndarray:
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] == 1:
            arr = raw.iloc[:, 0].to_numpy()
        else:
            arr = raw.iloc[:, 0].to_numpy()
    elif isinstance(raw, pd.Series):
        arr = raw.to_numpy()
    else:
        arr = np.asarray(raw)

    return np.asarray(arr, dtype=float).reshape(-1)


def predict_proba_pyfunc(model, X: pd.DataFrame, proba_col: str = "y_pred_proba") -> np.ndarray:
    """
    Extrai probabilidade da classe positiva a partir de um PyFunc.
    """
    raw = model.predict(X)

    if isinstance(raw, pd.DataFrame) and proba_col in raw.columns:
        y_prob = raw[proba_col].to_numpy(dtype=float)
    else:
        y_prob = _as_1d_float_array(raw)

    if y_prob.ndim != 1:
        raise ValueError(f"Saída do modelo deveria ser 1D, mas veio com shape={y_prob.shape}")

    return y_prob


def predict_df(
    trainer: ChurnModelTrainer,
    X: pd.DataFrame,
    threshold: float = 0.5,
    proba_col: str = "y_pred_proba",
    timezone: str = "America/Sao_Paulo",
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retorna DataFrame de predições.
    Utiliza `trainer.final_model` como pyfunc, e chama `predict_proba_pyfunc`.
    """
    if trainer.final_model is None:
        raise RuntimeError("trainer.final_model não foi definido.")

    y_prob = predict_proba_pyfunc(trainer.final_model, X, proba_col=proba_col)
    y_pred = (y_prob >= float(threshold)).astype(int)

    tz = pytz.timezone(timezone)
    now_dt = now or datetime.now(tz)
    prediction_month = now_dt.strftime("%Y-%m-01")

    return pd.DataFrame(
        {
            "prediction_month": prediction_month,
            "y_pred": y_pred,
            "y_pred_proba": y_prob.astype(float),
        }
    )


def predict_from_model_uri(
    X: pd.DataFrame,
    config: PredictConfig,
    client: Optional[MlflowClient] = None,
) -> pd.DataFrame:
    """
    Carrega o modelo (pyfunc) e retorna predições para X.

    Usos principais:
    - batch offline
    - FastAPI, com X vindo de request
    - testes, mock de mlflow.load_model / client

    Returns:
        DataFrame com prediction_month, y_pred, y_pred_proba.
    """
    setup_mlflow_sqlite(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        artifact_root=MLFLOW_ARTIFACT_ROOT,
    )

    model_uri = resolve_model_uri(config, client=client)
    model = load_pyfunc_model(model_uri, registry_uri=config.registry_uri)

    trainer = ChurnModelTrainer()
    trainer.final_model = model

    return predict_df(
        trainer=trainer,
        X=X,
        threshold=config.threshold,
        proba_col=config.proba_col,
        timezone=config.timezone,
    )
