"""
Treina e rastreia modelos de machine learning para previsão de churn.
Este script carrega os dados, pré-processa, divide em treino e teste, e treina modelos especificados.
Os resultados são rastreados usando MLflow, incluindo métricas e artefatos do modelo

Uso:
    python src/infra/jobs/train.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

import os
import logging
from typing import Any
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.pipelines import SklearnPipelineRunner
from src.utils.constants import (
    FEATURES_COLS, YES_NO_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, CAT_COLS, NUM_COLS, BOL_COLS
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_models() -> dict[str, Any]:
    """Define os modelos a comparar."""
    return {
        "logreg": LogisticRegression(random_state=RANDOM_STATE),
    }


def train_and_track(models: dict[str, object] | None = None) -> None:
    models = models or get_models()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-model-comparison")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")

    X_raw = df_clean[FEATURES_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() == 2 else None,
    )

    num_cols = NUM_COLS + BOL_COLS

    parent_run_name = os.getenv("MLFLOW_RUN_NAME", "compare_models")

    best = {"model_name": None, "accuracy": float("-inf")}

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags({
            "project": "churn",
            "task": "binary_classification",
            "job": "train_churn_mlflow_job",
        })

        mlflow.log_params({
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "categorical_cols_n": len(CAT_COLS),
            "numerical_cols_n": len(num_cols),
            "n_models": len(models),
            "optimize_metric": "accuracy",
        })

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.set_tag("model_name", model_name)
                mlflow.log_param("estimator_class", model.__class__.__name__)

                runner = SklearnPipelineRunner(
                    model=model,
                    categorical_cols=CAT_COLS,
                    numerical_cols=num_cols,
                    use_feature_engineering=False,
                    feature_engineering_transformer=None,
                    use_feature_selection=False,
                    use_grid_search=False,
                    param_grid={},
                    cv=5,
                    scoring="accuracy",
                    pos_label=1,
                )

                logger.info("Treinando modelo=%s (%s)", model_name, model.__class__.__name__)
                runner.fit(X_train, y_train)

                metrics = runner.evaluate(X_test, y_test, include_auc=True)
                mlflow.log_metrics(metrics)

                try:
                    mlflow.sklearn.log_model(
                        sk_model=runner.best_model,
                        name="model",
                        serialization_format="skops",
                        pip_requirements="requirements.txt",
                    )
                except TypeError:
                    mlflow.sklearn.log_model(
                        sk_model=runner.best_model,
                        artifact_path="model",
                        serialization_format="skops",
                        pip_requirements="requirements.txt",
                    )

                acc = metrics.get("accuracy", float("-inf"))
                if acc > best["accuracy"]:
                    best = {"model_name": model_name, "accuracy": acc}

        mlflow.log_params({
            "best_model_name": best["model_name"],
        })
        mlflow.log_metric("best_accuracy", best["accuracy"])

        logger.info("Melhor modelo: %s | accuracy=%.4f", best["model_name"], best["accuracy"])
        logger.info("Parent run_id: %s", parent_run.info.run_id)


if __name__ == "__main__":
    train_and_track()
