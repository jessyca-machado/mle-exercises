"""model_comparison — Comparação de modelos (DecisionTree, RandomForest, SVC) com opção de seleção de features.

Executar um experimento comparativo entre três modelos clássicos e 1 modelo complexo:
    - DecisionTreeClassifier
    - RandomForestClassifier
    - SVC
    - GradientBoostingClassifier

Métricas
    - accuracy
    - f1 (classe positiva = 1)
    - precision (classe positiva = 1)
    - recall (classe positiva = 1)
    - auc_roc (se houver predict_proba)

Uso
    python experiments/comparison/comparison_model.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Tuple

import pandas as pd
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from src.data.pipelines import SklearnPipelineRunner
from src.ml.data_utils import load_and_split_churn
from src.ml.experiment_runner import ExperimentSpec, run_sklearn_models_experiment
from src.ml.logging_utils import get_logger
from src.ml.mlflow_utils import setup_mlflow, end_active_run
from src.utils.constants import CAT_COLS, NUM_COLS, BOL_COLS, BIN_COLS, RANDOM_STATE, TEST_SIZE, TRUSTED
from src.data.feature_engineering import TelcoFeatureEngineeringBins

from src.data.transformers import ToFloat32

logger = get_logger(__name__)

def get_models():
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

fe_transformer = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)

optuna_param_distributions: Dict[str, Dict[str, optuna.distributions.BaseDistribution]] = {
    "GradientBoosting": {
        "model__n_estimators": IntDistribution(100, 600, step=50),
        "model__learning_rate": FloatDistribution(0.01, 0.2, log=True),
        "model__max_depth": IntDistribution(2, 4),
        "model__subsample": FloatDistribution(0.8, 1.0),
    },
    "RandomForest": {
        "model__n_estimators": IntDistribution(200, 600, step=50),
        "model__max_depth": CategoricalDistribution([None, 8, 16]),
        "model__min_samples_split": IntDistribution(2, 30),
        "model__min_samples_leaf": IntDistribution(1, 10),
        "model__max_features": CategoricalDistribution(["sqrt", "log2"]),
    },
    "SVC_rbf": {
        "model__C": FloatDistribution(1e-1, 10.0, log=True),
        "model__gamma": CategoricalDistribution(["scale", 1e-3, 1e-2, 1e-1]),
        "model__class_weight": CategoricalDistribution([None, "balanced"]),
    },
    "DecisionTree": {
        "model__max_depth": CategoricalDistribution([None, 3, 5, 8, 12]),
        "model__min_samples_split": IntDistribution(2, 30),
        "model__min_samples_leaf": IntDistribution(1, 10),
        "model__criterion": CategoricalDistribution(["gini", "entropy"]),
    },
}


def build_runner(model_name: str, model):
    return SklearnPipelineRunner(
        model=model,
        categorical_cols=CAT_COLS,
        numerical_cols=NUM_COLS,
        boolean_cols=BOL_COLS,
        binned_cols=BIN_COLS,
        use_feature_engineering=True,
        feature_engineering_transformer=fe_transformer,
        use_feature_selection=True,
        use_optuna_search=True,
        optuna_param_distributions=optuna_param_distributions.get(model_name, {}),
        optuna_n_trials=30,
        optuna_timeout=None,
        optuna_n_jobs=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1",
        pos_label=1,
    )


def run_comparison_experiment(save_best: bool | None = None) -> pd.DataFrame:
    setup_mlflow(default_experiment="churn-model-comparison")
    end_active_run()

    X_train, X_test, y_train, y_test = load_and_split_churn()
    models = get_models()

    if save_best is None:
        save_best = os.getenv("SAVE_BEST_JOBLIB", "0") == "1"

    base_trusted = list(TRUSTED or ())
    tofloat32_fqn = "src.data.transformers.ToFloat32"
    if tofloat32_fqn not in base_trusted:
        base_trusted.append(tofloat32_fqn)
    trusted_types = tuple(base_trusted)

    spec = ExperimentSpec(
        parent_run_name="comparison_model",
        job_tag="comparison_model",
        best_metric="f1",
        save_best=save_best,
        best_output_path_tpl="models/comparison_best_{model}.joblib",
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
        build_runner=build_runner,  # <-- passa a função, não chama
        parent_tags={"project": "churn", "task": "binary_classification"},
        parent_params={
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "use_feature_engineering": True,
            "use_feature_selection": True,
            "k_best": 15,
            "do_cv": True,
            "select_by_cv": True,
            "tuning_mode": "best_only",
            "optuna_n_trials": 30,
            "skops_trusted_types_added": "ToFloat32",
        },
        tuning_mode="best_only",   # <-- GS só no melhor
        do_cv=True,                   # <-- roda CV (fase 1)
        select_by_cv=True,            # <-- seleciona vencedor pelo mean_cv
        cv_refit_only_best=True,      # <-- CV em todos (necessário pra comparar por CV)
    )


    if 'mean_cv' in df.columns:
        df = df.sort_values("mean_cv", ascending=False)

    logger.info("Resumo:\n%s", df.to_string(index=False))
    return df


def main() -> None:
    run_comparison_experiment(save_best=True)


if __name__ == "__main__":
    main()
