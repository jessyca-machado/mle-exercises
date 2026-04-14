"""
Baseline model — treinamento de modelos baseline (Dummy) e LogisticRegression usando a pipeline robusta do projeto.

O que este script faz:
- load_data_churn -> pre_processing -> split
- Treina 3 modelos com o mesmo pipeline (SklearnPipelineRunner):
    1) DummyClassifier(most_frequent) - baseline "ingênuo" que pode prever só a classe 0
    2) DummyClassifier(stratified) - baseline que sorteia classes conforme a proporção, evitando prever só 0
    3) LogisticRegression
- Loga métricas no MLflow (1 run pai + runs filhos aninhados)
- Loga modelo (pipeline completo) como artefato
- (Opcional) salva o melhor modelo em joblib

Uso:
    python experiments/baselines/baseline_model.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Tuple

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.pipelines import SklearnPipelineRunner
from src.ml.data_utils import load_and_split_churn
from src.ml.experiment_runner import ExperimentSpec, run_sklearn_models_experiment
from src.ml.logging_utils import get_logger
from src.ml.mlflow_utils import setup_mlflow, end_active_run
from src.utils.constants import CAT_COLS, NUM_COLS, BOL_COLS, BIN_COLS, RANDOM_STATE, TEST_SIZE, TRUSTED

logger = get_logger(__name__)

def get_models():
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logreg": LogisticRegression(random_state=RANDOM_STATE),
    }


def build_runner(model):
    return SklearnPipelineRunner(
        model=model,
        categorical_cols=CAT_COLS,
        numerical_cols=NUM_COLS,
        boolean_cols=BOL_COLS,
        binned_cols=BIN_COLS,
        use_feature_engineering=False,
        feature_engineering_transformer=None,
        use_feature_selection=False,
        k_best=None,
        use_optuna_search=False,
        optuna_param_distributions={},
        optuna_n_trials=0,
        optuna_timeout=None,
        optuna_n_jobs=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        pos_label=1,
    )


def run_baseline_experiment(save_best: bool | None = None) -> pd.DataFrame:
    setup_mlflow(default_experiment="churn-baselines")
    end_active_run()

    X_train, X_test, y_train, y_test = load_and_split_churn()
    models = get_models()

    if save_best is None:
        save_best = os.getenv("SAVE_BEST_JOBLIB", "0") == "1"

    tofloat32_fqn = "src.data.transformers.ToFloat32"
    base_trusted = list(TRUSTED or ())
    if tofloat32_fqn not in base_trusted:
        base_trusted.append(tofloat32_fqn)
    trusted_types = tuple(base_trusted)

    spec = ExperimentSpec(
        parent_run_name="baseline_model",
        job_tag="baseline_model",
        best_metric="f1",
        save_best=save_best,
        best_output_path_tpl="models/baseline_best_{model}.joblib",
        log_best_confusion_matrix=True,
        cm_normalize=None,
        cm_labels=(0, 1),
        log_model=True,
        model_artifact_name="model",
        skops_trusted_types=trusted_types,
    )

    df = run_sklearn_models_experiment(
        logger=logger,
        spec=spec,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models,
        build_runner=build_runner,
        parent_tags={"project": "churn", "task": "binary_classification"},
        parent_params={
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "use_feature_engineering": False,
            "use_feature_selection": False,
            "k_best": None,
            "do_cv": True,
            "select_by_cv": True,
            "tuning_mode": "none",
            "skops_trusted_types_added": tofloat32_fqn,
        },
        tuning_mode="none",
        do_cv=True,               # roda cross_validate em cada modelo
        select_by_cv=True,        # seleciona melhor pelo mean_cv
        cv_refit_only_best=True,  # comportamento do seu runner/orquestrador (mantido)
    )

    if "mean_cv" in df.columns:
        df = df.sort_values("mean_cv", ascending=False)

    logger.info("Resumo:\n%s", df.to_string(index=False))

    return df


def main() -> None:
    run_baseline_experiment(save_best=True)


if __name__ == "__main__":
    main()
