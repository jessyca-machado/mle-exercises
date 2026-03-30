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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    metrics: Dict[str, float]


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tag("artifact_root_hint", MLFLOW_ARTIFACT_ROOT)
    mlflow.set_tag("mlflow_backend_store", MLFLOW_TRACKING_URI)


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
    parent_run_name: Optional[str] = "model_comparison",
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

    parent_ctx = (
        mlflow.start_run(run_name=parent_run_name)
        if parent_run_name is not None
        else None
    )

    try:
        X_train, X_test, y_train, y_test, encoder = prepare_train_test(
            features=FEATURES_COLS,
            target="Churn",
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            drop_first=True,
            use_feature_engineering=use_feature_engineering,
        )

        logger.info("Antes da seleção: %d features", X_train.shape[1])

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
                        ("clf", SVC(
                            kernel="rbf",
                            probability=True,
                            random_state=RANDOM_STATE,
                        )),
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
            with mlflow.start_run(run_name=name, nested=(parent_ctx is not None)):

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

                if name == "MLP_torch":
                    clf = model.named_steps["clf"]
                    mlflow.log_param("mlp_hidden_dims", str(clf.hidden_dims))
                    mlflow.log_param("mlp_dropout", float(clf.dropout))
                    mlflow.log_param("mlp_lr", float(clf.lr))
                    mlflow.log_param("mlp_batch_size", int(clf.batch_size))
                    mlflow.log_param("mlp_epochs", int(clf.epochs))
                    mlflow.log_param("mlp_weight_decay", float(clf.weight_decay))
                    mlflow.log_param("mlp_device", str(clf.device))

                try:
                    mlflow.log_param("encoder_num_columns", int(len(encoder.columns_)))
                except Exception:
                    pass

                if use_feature_selection and selector is not None:
                    mlflow.log_text("\n".join(selector.selected_features_), "selected_features.txt")

                    if isinstance(selector, ImportanceSelector):
                        mlflow.log_text(
                            selector.importances_.to_string(),
                            "feature_importances_full.txt",
                        )
                        mlflow.log_text(
                            selector.importances_.head(50).to_string(),
                            "feature_importances_top50.txt",
                        )
                    elif isinstance(selector, AnovaSelector):
                        mlflow.log_text(selector.scores_.to_string(), "anova_scores_full.txt")
                        mlflow.log_text(selector.pvalues_.to_string(), "anova_pvalues_full.txt")
                        mlflow.log_text(selector.scores_.head(50).to_string(), "anova_scores_top50.txt")

                metrics, report = train_and_evaluate(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_name=name,
                )

                mlflow.log_metric("train_accuracy", metrics["train_accuracy"])
                mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
                mlflow.log_metric("test_f1", metrics["test_f1"])
                mlflow.log_metric("overfitting_gap", metrics["overfitting_gap"])

                if not np.isnan(metrics["auc_roc"]):
                    mlflow.log_metric("auc_roc", metrics["auc_roc"])
                if not np.isnan(metrics["pr_auc"]):
                    mlflow.log_metric("pr_auc", metrics["pr_auc"])

                mlflow.log_text(report, "classification_report.txt")

                if log_models:
                    mlflow.sklearn.log_model(model, artifact_path="model")

                results.append(ModelResult(name=name, metrics=metrics))

        summary = (
            pd.DataFrame(
                [
                    {
                        "model": r.name,
                        "train_accuracy": r.metrics["train_accuracy"],
                        "test_accuracy": r.metrics["test_accuracy"],
                        "test_f1": r.metrics["test_f1"],
                        "auc_roc": r.metrics["auc_roc"],
                        "pr_auc": r.metrics["pr_auc"],
                        "overfitting_gap": r.metrics["overfitting_gap"],
                    }
                    for r in results
                ]
            )
            .sort_values(by="pr_auc", ascending=False)
        )

        logger.info("\n=== Summary (sorted by PR-AUC) ===\n%s", summary.to_string(index=False))

        if parent_ctx is not None:
            mlflow.log_text(summary.to_string(index=False), "summary.txt")
            mlflow.log_dict(
                {
                    "use_feature_engineering": use_feature_engineering,
                    "use_feature_selection": use_feature_selection,
                    "feature_selection": feature_selection if use_feature_selection else None,
                    "top_k": top_k if use_feature_selection else None,
                    "n_features_before_selection": n_features_before,
                    "n_features_after_selection": n_features_after,
                },
                "experiment_config.json",
            )

    finally:
        if parent_ctx is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main(
        use_feature_engineering=False,
        use_feature_selection=True,
        feature_selection="anova",  # ou "anova"
        top_k=10,
        log_models=False,
        parent_run_name="model_comparison",
    )
