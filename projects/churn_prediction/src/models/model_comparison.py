"""model_comparison — Comparação de modelos (DecisionTree, RandomForest, SVC) com opção de seleção de features.

Executar um experimento comparativo entre três modelos clássicos e 1 modelo complexo:
    - DecisionTreeClassifier
    - RandomForestClassifier
    - SVC
    - MLPClassifier (implementação customizada usando PyTorch)

O script utiliza o pipeline de dados do projeto para garantir:
    - split treino/teste adequado
    - one-hot encoding alinhado (fit no treino, transform no teste)
    - (opcional) feature engineering
    - (opcional) seleção de features por feature_importances_ ou Anova (fit no treino)

Métricas
    - accuracy
    - f1 (classe positiva = 1)
    - auc_roc (se houver predict_proba)
    - pr_auc (Average Precision, se houver predict_proba)

Uso
    python src/models/model_comparison.py

Para visualizar:
    mlflow ui # Inicia UI em http://localhost:5000
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

import mlflow

from src.data.pipelines import prepare_train_test
from src.models.feature_selection import (
    fit_feature_selector,
    apply_selector,
    ImportanceSelector,
    AnovaSelector,
)
from src.utils.constants import (
    RANDOM_STATE,
    TEST_SIZE,
    FEATURES_COLS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
)
from src.models.torch_mlp import TorchMLPClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """
    Estrutura simples para armazenar resultados de avaliação de um modelo.

    Atributos:
        name: Nome do modelo.
        metrics: Dicionário com métricas calculadas (accuracy, f1, auc_roc, pr_auc).
    """
    name: str
    cv_metrics: Dict[str, float]
    holdout_metrics: Dict[str, float]


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tag("artifact_root_hint", MLFLOW_ARTIFACT_ROOT)
    mlflow.set_tag("mlflow_backend_store", MLFLOW_TRACKING_URI)


def roc_auc_scorer(estimator, X, y_true):
    y_proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y_true, y_proba)


def pr_auc_scorer(estimator, X, y_true):
    y_proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y_true, y_proba)


def cross_validate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "acc": "accuracy",
        "f1": "f1",
        "roc_auc": roc_auc_scorer,
        "pr_auc": pr_auc_scorer,
    }

    out = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )

    return {
        "cv_acc_mean": float(out["test_acc"].mean()),
        "cv_acc_std": float(out["test_acc"].std()),
        "cv_f1_mean": float(out["test_f1"].mean()),
        "cv_f1_std": float(out["test_f1"].std()),
        "cv_auc_mean": float(out["test_roc_auc"].mean()),
        "cv_auc_std": float(out["test_roc_auc"].std()),
        "cv_pr_auc_mean": float(out["test_pr_auc"].mean()),
        "cv_pr_auc_std": float(out["test_pr_auc"].std()),
    }


def train_and_evaluate(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> Tuple[Dict[str, float], str]:
    """
    Treina um modelo e calcula métricas no conjunto de teste.

    Centralizar lógica de treino/avaliação e padronizar o logging de métricas.

    Args:
        model: Pipeline sklearn (ou estimador compatível com .fit/.predict).
        X_train: Features de treino.
        X_test: Features de teste (mesmas colunas e ordem do treino).
        y_train: Target de treino.
        y_test: Target de teste.
        model_name: Nome do modelo para logs.

    Returns:
        Dicionário com:
        - accuracy
        - f1
        - auc_roc (NaN se não houver predict_proba)
        - pr_auc (NaN se não houver predict_proba)
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    report = classification_report(y_test, y_pred_test)

    logger.info("\n--- %s ---", model_name)
    logger.info(report)

    metrics: Dict[str, float] = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_f1": float(f1_score(y_test, y_pred_test, pos_label=1)),
    }
    metrics["overfitting_gap"] = float(metrics["train_accuracy"] - metrics["test_accuracy"])

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    logger.info(
        "Metrics | train_acc=%.4f test_acc=%.4f f1=%.4f auc_roc=%.4f pr_auc=%.4f gap=%.4f",
        metrics["train_accuracy"],
        metrics["test_accuracy"],
        metrics["test_f1"],
        metrics["auc_roc"],
        metrics["pr_auc"],
        metrics["overfitting_gap"],
    )
    return metrics, report


def main(
    use_feature_engineering: bool = True,
    use_feature_selection: bool = True,
    feature_selection: str = "feature_importance",
    top_k: int = 10,
    log_models: bool = False,
    parent_run_name: Optional[str] = "model_comparison",  # mantido na assinatura, mas não usado
    cv_splits: int = 5,
) -> None:
    """
    Executa comparação entre DecisionTree, RandomForest e SVC, com seleção de features opcional.
    Preparar os dados via pipeline do projeto, opcionalmente reduzir a dimensionalidade
    por importâncias e comparar os modelos com as mesmas features finais.
    Args:
        use_feature_engineering:
            Se True, aplica feature_engineering dentro do pipeline de dados.
            Se False, usa apenas preprocess + encoding.
        use_feature_selection:
            Se True, aplica seleção de features Top-K usando importâncias de RandomForest.
        top_k: Quantidade de features a manter quando use_feature_selection=True.
    Returns:
        None: Função de execução (efeitos colaterais: logs e impressão de summary).
    """
    _setup_mlflow()

    if mlflow.active_run() is not None:
        mlflow.end_run()

    X_train, X_test, y_train, y_test, encoder = prepare_train_test(
        features=FEATURES_COLS,
        target="Churn",
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        drop_first=True,
        use_feature_engineering=use_feature_engineering,
    )

    n_features_before = int(X_train.shape[1])
    logger.info("Antes da seleção: %d features", n_features_before)

    selector = None
    if use_feature_selection:
        selector = fit_feature_selector(
            X_train,
            y_train,
            top_k=top_k,
            random_state=RANDOM_STATE,
            feature_selection=feature_selection,
        )
        X_train = apply_selector(X_train, selector)
        X_test = apply_selector(X_test, selector)

    n_features_after = int(X_train.shape[1])
    logger.info("Depois da seleção: %d features", n_features_after)

    models: List[Tuple[str, Pipeline]] = [
        (
            "DecisionTree",
            Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        ),
        (
            "RandomForest",
            Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
        ),
        (
            "SVC_rbf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "MLP_torch",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", TorchMLPClassifier(random_state=RANDOM_STATE)),
                ]
            ),
        ),
    ]

    results: List[ModelResult] = []

    for name, model in models:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("drop_first", True)
            mlflow.log_param("use_feature_engineering", bool(use_feature_engineering))
            mlflow.log_param("use_feature_selection", bool(use_feature_selection))
            mlflow.log_param("feature_selection", feature_selection if use_feature_selection else "none")
            mlflow.log_param("top_k", int(top_k) if use_feature_selection else 0)
            mlflow.log_param("n_features_before_selection", n_features_before)
            mlflow.log_param("n_features_after_selection", n_features_after)
            mlflow.log_param("cv_splits", int(cv_splits))

            try:
                mlflow.log_param("encoder_num_columns", int(len(encoder.columns_)))
            except Exception:
                pass

            if use_feature_selection and selector is not None:
                mlflow.log_text("\n".join(selector.selected_features_), "selected_features.txt")
                if isinstance(selector, ImportanceSelector):
                    mlflow.log_text(selector.importances_.to_string(), "feature_importances_full.txt")
                    mlflow.log_text(selector.importances_.head(50).to_string(), "feature_importances_top50.txt")
                elif isinstance(selector, AnovaSelector):
                    mlflow.log_text(selector.scores_.to_string(), "anova_scores_full.txt")
                    mlflow.log_text(selector.pvalues_.to_string(), "anova_pvalues_full.txt")
                    mlflow.log_text(selector.scores_.head(50).to_string(), "anova_scores_top50.txt")

            if name == "MLP_torch":
                clf = model.named_steps["clf"]
                for attr in ["hidden_dims", "dropout", "lr", "batch_size", "epochs", "weight_decay", "device"]:
                    if hasattr(clf, attr):
                        mlflow.log_param(f"mlp_{attr}", str(getattr(clf, attr)))

            n_jobs_cv = 1 if name == "MLP_torch" else -1
            logger.info("Running CV for %s (n_jobs=%s)...", name, n_jobs_cv)

            cv_metrics = cross_validate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                cv_splits=cv_splits,
                n_jobs=n_jobs_cv,
            )

            for k, v in cv_metrics.items():
                mlflow.log_metric(k, v)

            holdout_metrics, report = train_and_evaluate(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=name,
            )

            logger.info(
                "CV (%s) | acc=%.4f±%.4f f1=%.4f±%.4f auc=%.4f±%.4f pr_auc=%.4f±%.4f",
                name,
                cv_metrics["cv_acc_mean"], cv_metrics["cv_acc_std"],
                cv_metrics["cv_f1_mean"], cv_metrics["cv_f1_std"],
                cv_metrics["cv_auc_mean"], cv_metrics["cv_auc_std"],
                cv_metrics["cv_pr_auc_mean"], cv_metrics["cv_pr_auc_std"],
            )

            mlflow.log_metric("train_accuracy", holdout_metrics["train_accuracy"])
            mlflow.log_metric("test_accuracy", holdout_metrics["test_accuracy"])
            mlflow.log_metric("test_f1", holdout_metrics["test_f1"])
            mlflow.log_metric("overfitting_gap", holdout_metrics["overfitting_gap"])

            if not np.isnan(holdout_metrics["auc_roc"]):
                mlflow.log_metric("auc_roc", holdout_metrics["auc_roc"])
            if not np.isnan(holdout_metrics["pr_auc"]):
                mlflow.log_metric("pr_auc", holdout_metrics["pr_auc"])

            mlflow.log_text(report, "classification_report.txt")

            if log_models:
                mlflow.sklearn.log_model(model, artifact_path="model")

            results.append(ModelResult(name=name, cv_metrics=cv_metrics, holdout_metrics=holdout_metrics))

    summary = (
        pd.DataFrame(
            [
                {
                    "model": r.name,
                    "cv_pr_auc_mean": r.cv_metrics["cv_pr_auc_mean"],
                    "cv_pr_auc_std": r.cv_metrics["cv_pr_auc_std"],
                    "cv_auc_mean": r.cv_metrics["cv_auc_mean"],
                    "cv_f1_mean": r.cv_metrics["cv_f1_mean"],
                    "cv_acc_mean": r.cv_metrics["cv_acc_mean"],
                    "test_pr_auc": r.holdout_metrics["pr_auc"],
                    "test_auc_roc": r.holdout_metrics["auc_roc"],
                    "test_f1": r.holdout_metrics["test_f1"],
                    "test_accuracy": r.holdout_metrics["test_accuracy"],
                    "train_accuracy": r.holdout_metrics["train_accuracy"],
                    "overfitting_gap": r.holdout_metrics["overfitting_gap"],
                }
                for r in results
            ]
        )
        .sort_values(by="cv_pr_auc_mean", ascending=False)
    )

    logger.info("\n=== Summary (sorted by CV PR-AUC mean) ===\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main(
        use_feature_engineering=True,
        use_feature_selection=False,
        feature_selection="feature_importance",
        top_k=10,
        log_models=False,
        parent_run_name="model_comparison",
        cv_splits=5,
    )
