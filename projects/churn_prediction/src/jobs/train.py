"""
Treina e rastreia modelos de machine learning para previsão de churn.
Este script carrega os dados, pré-processa, divide em treino e teste, e treina modelos especificados.
Os resultados são rastreados usando MLflow, incluindo métricas e artefatos do modelo

Uso:
    python src/jobs/train.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

import logging
import os
from typing import Any

from sklearn.linear_model import LogisticRegression

from src.data.pipelines import SklearnPipelineRunner
from src.ml.data_utils import load_and_split_churn
from src.ml.experiment_runner import ExperimentSpec, run_experiment
from src.ml.mlflow_utils import setup_mlflow, end_active_run
from src.utils.constants import (
    CAT_COLS, NUM_COLS, BOL_COLS, BIN_COLS,
    RANDOM_STATE, TEST_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_models() -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(random_state=RANDOM_STATE),
    }

def build_runner(model: object) -> SklearnPipelineRunner:
    num_cols = NUM_COLS + BOL_COLS  # mantenha sua regra aqui se for “do job”
    return SklearnPipelineRunner(
        model=model,
        categorical_cols=CAT_COLS,
        numerical_cols=num_cols,
        boolean_cols=[],         # ou remova se seu runner nem precisa
        binned_cols=BIN_COLS,    # se aplicável
        use_feature_engineering=False,
        feature_engineering_transformer=None,
        use_feature_selection=False,
        k_best=None,
        use_grid_search=False,
        param_grid={},
        cv=5,
        scoring="accuracy",
        pos_label=1,
    )

def train_and_track(models: dict[str, object] | None = None) -> None:
    models = models or get_models()

    setup_mlflow(default_experiment="churn-model-comparison")
    end_active_run()

    X_train, X_test, y_train, y_test = load_and_split_churn()

    parent_run_name = os.getenv("MLFLOW_RUN_NAME", "compare_models")

    spec = ExperimentSpec(
        parent_run_name=parent_run_name,
        job_tag="train_churn_mlflow_job",
        best_metric="accuracy",
        log_model=True,
    )

    df = run_experiment(
        logger=logger,
        spec=spec,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        models=models,
        build_runner=build_runner,
        parent_tags={"project": "churn", "task": "binary_classification"},
        parent_params={"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
    )

    logger.info("Resumo:\n%s", df.to_string(index=False))

if __name__ == "__main__":
    train_and_track()