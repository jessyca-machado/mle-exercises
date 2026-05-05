from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.jobs.predict import PredictConfig, load_pyfunc_model, resolve_model_uri
from src.utils.constants import FEATURES_COLS, YES_NO_COLS


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def predict_proba_from_pyfunc(
    model, X: pd.DataFrame, proba_col: str = "y_pred_proba"
) -> np.ndarray:
    """
    Compatível com o contrato: PyFunc retorna DataFrame com coluna y_pred_proba
    ou array/Series.
    """
    raw = model.predict(X)
    if isinstance(raw, pd.DataFrame) and proba_col in raw.columns:
        arr = raw[proba_col].to_numpy(dtype=float)
    elif isinstance(raw, pd.Series):
        arr = raw.to_numpy(dtype=float)
    else:
        arr = np.asarray(raw, dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)


def main() -> None:
    out_path = Path(os.getenv("BASELINE_OUT_PATH", "artifacts/baseline.json"))
    _ensure_dir(out_path)

    # Prioridade: BASELINE_MODEL_URI > CHURN_MODEL_URI > model_name/version
    model_uri = os.getenv("BASELINE_MODEL_URI") or os.getenv("CHURN_MODEL_URI")
    model_name = os.getenv("BASELINE_MODEL_NAME")
    model_version = os.getenv("BASELINE_MODEL_VERSION")

    # MLflow endpoints (no docker: use http://mlflow-proxy; no host: http://localhost:5001)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    cfg = PredictConfig(
        registry_uri=registry_uri,
        model_uri=model_uri,
        model_name=model_name,
        model_version=model_version,
        threshold=float(os.getenv("CHURN_THRESHOLD", "0.5")),
        proba_col="y_pred_proba",
    )
    resolved_uri = resolve_model_uri(cfg)
    model = load_pyfunc_model(resolved_uri, registry_uri=cfg.registry_uri)

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")

    X = df_clean[FEATURES_COLS].copy()

    numeric_cols = [
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
    ]
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    # Colunas numéricas para drift (pegue do X já coerced)
    tenure = _to_float_series(X["tenure"])
    monthly = _to_float_series(X["MonthlyCharges"])
    total = _to_float_series(X["TotalCharges"])

    # Score baseline (com o modelo atual)
    y_pred_proba = predict_proba_from_pyfunc(model, X, proba_col="y_pred_proba")

    baseline: dict[str, Any] = {
        "meta": {
            "model_uri": resolved_uri,
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri,
            "n_rows": int(X.shape[0]),
        },
        "tenure": np.asarray(tenure, dtype=float).tolist(),
        "MonthlyCharges": np.asarray(monthly, dtype=float).tolist(),
        "TotalCharges": np.asarray(total, dtype=float).tolist(),
        "y_pred_proba": np.asarray(y_pred_proba, dtype=float).tolist(),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False)

    print(f"Baseline salvo em: {out_path.resolve()}")
    print(f"Model URI usado: {resolved_uri}")
    print(f"N linhas: {baseline['meta']['n_rows']}")


if __name__ == "__main__":
    main()
