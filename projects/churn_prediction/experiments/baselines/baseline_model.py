"""Baseline model — treinamento de modelos heurístico e de regressão logística.

- Demonstra como estabelecer baselines sólidas antes de modelos complexos.
- Inclui logging de hiperparâmetros e métricas para comparação.
- Loga distribuição de classes (para evidenciar desbalanceamento).
- Usa dois baselines:
    1) most_frequent (baseline "ingênuo" que pode prever só a classe 0)
    2) stratified (baseline que sorteia classes conforme a proporção, evitando prever só 0)
- Usa StandardScaler para normalização dos dados na pipeline LogisticRegression.

MLflow:
- Cria 1 run por modelo (Dummy MF, Dummy Strat, Logistic Regression)
- Loga parâmetros, métricas, artefatos (classification report)
- Loga o modelo (apenas LogisticRegression pipeline)

Uso:
    python experiments/baselines/baseline_model.py

Para visualizar:
    mlflow ui # Inicia UI em http://localhost:5000
"""
import logging

import numpy as np
import pandas as pd
import mlflow

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.pipelines import prepare_train_test
from src.utils.helpers import log_class_distribution
from src.utils.constants import (
    RANDOM_STATE,
    TEST_SIZE,
    FEATURES_COLS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """Treina e avalia um modelo, retornando métricas.

    Args:
        model: Pipeline sklearn a ser treinado.
        X_train: Features de treino.
        X_test: Features de teste.
        y_train: Target de treino.
        y_test: Target de teste.
        model_name: Nome do modelo para logging.

    Returns:
        Dicionário com métricas: accuracy, f1_score, auc_roc, pr_auc.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info("\n--- %s ---", model_name)
    logger.info(classification_report(y_test, y_pred))

    metrics: dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["f1"] = float(f1_score(y_test, y_pred, pos_label=1))
    logger.info("F1 (classe 1): %.4f", metrics["f1"])

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        logger.info("AUC-ROC: %.4f", metrics["auc_roc"])
        logger.info("PR-AUC (Average Precision): %.4f", metrics["pr_auc"])
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique_preds.tolist(), pred_counts.tolist()))
    logger.info("Distribuição das predições (classe -> contagem): %s", pred_dist)

    return metrics


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tag("artifact_root_hint", MLFLOW_ARTIFACT_ROOT)
    mlflow.set_tag("mlflow_backend_store", MLFLOW_TRACKING_URI)


def main() -> None:
    _setup_mlflow()
    if mlflow.active_run() is not None:
        mlflow.end_run()

    X_train, X_test, y_train, y_test, encoder = prepare_train_test(
        features=FEATURES_COLS,
        target="Churn",
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        drop_first=True,
        use_feature_engineering=False,
    )

    logger.info("Treino: %d | Teste: %d", len(X_train), len(X_test))

    name, counts, ratios = log_class_distribution(y_train, "y_train")
    logger.info("%s | contagens:\n%s", name, counts.to_string())
    logger.info("%s | proporções:\n%s", name, ratios.to_string())

    name, counts, ratios = log_class_distribution(y_test, "y_test")
    logger.info("%s | contagens:\n%s", name, counts.to_string())
    logger.info("%s | proporções:\n%s", name, ratios.to_string())

    dummy_mf_pipeline = Pipeline(
        [("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))]
    )
    dummy_mf_metrics = train_and_evaluate(
        dummy_mf_pipeline, X_train, X_test, y_train, y_test, "DummyClassifier (most_frequent)"
    )

    dummy_strat_pipeline = Pipeline(
        [("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))]
    )
    dummy_strat_metrics = train_and_evaluate(
        dummy_strat_pipeline, X_train, X_test, y_train, y_test, "DummyClassifier (stratified)"
    )

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("drop_first", True)
        mlflow.log_param("n_features_after_encoding", int(X_train.shape[1]))
        mlflow.log_param("encoder_num_columns", int(len(encoder.columns_)))

        lr_metrics = train_and_evaluate(
            lr_pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            "LogisticRegression",
        )

        y_pred_train = lr_pipeline.predict(X_train)
        y_pred_test = lr_pipeline.predict(X_test)

        mlflow.log_text(classification_report(y_test, y_pred_test), "classification_report.txt")

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        mlflow.log_metric("train_accuracy", float(train_accuracy))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_f1_score", float(f1_score(y_test, y_pred_test, pos_label=1)))
        mlflow.log_metric("test_precision", float(precision_score(y_test, y_pred_test, pos_label=1)))
        mlflow.log_metric("test_recall", float(recall_score(y_test, y_pred_test, pos_label=1)))
        mlflow.log_metric("overfitting_gap", float(train_accuracy - test_accuracy))

        if "auc_roc" in lr_metrics and not np.isnan(lr_metrics["auc_roc"]):
            mlflow.log_metric("auc_roc", lr_metrics["auc_roc"])
        if "pr_auc" in lr_metrics and not np.isnan(lr_metrics["pr_auc"]):
            mlflow.log_metric("pr_auc", lr_metrics["pr_auc"])

    logger.info("\n=== Comparação (Accuracy) ===")
    logger.info("Acc Dummy(most_frequent): %.4f", dummy_mf_metrics["accuracy"])
    logger.info("Acc Dummy(stratified):    %.4f", dummy_strat_metrics["accuracy"])
    logger.info("Acc LogisticRegression:   %.4f", lr_metrics["accuracy"])

    logger.info("\n=== Comparação (F1 classe 1) ===")
    logger.info("F1 Dummy(most_frequent): %.4f", dummy_mf_metrics["f1"])
    logger.info("F1 Dummy(stratified):    %.4f", dummy_strat_metrics["f1"])
    logger.info("F1 LogisticRegression:   %.4f", lr_metrics["f1"])

    logger.info("\n=== Comparação (PR-AUC) ===")
    logger.info("PR-AUC Dummy(most_frequent): %.4f", dummy_mf_metrics["pr_auc"])
    logger.info("PR-AUC Dummy(stratified):    %.4f", dummy_strat_metrics["pr_auc"])
    logger.info("PR-AUC LogisticRegression:   %.4f", lr_metrics["pr_auc"])

    logger.info("\n=== Comparação geral (LogisticRegression vs. DummyClassifier(most_frequent)) ===")
    logger.info(
        "Ganho de acurácia vs baseline: +%.2f%%",
        (lr_metrics["accuracy"] - dummy_mf_metrics["accuracy"]) * 100,
    )
    logger.info(
        "Ganho de F1 (classe 1) vs baseline: +%.2f%%",
        (lr_metrics["f1"] - dummy_mf_metrics["f1"]) * 100,
    )
    logger.info(
        "Ganho de PR-AUC vs baseline: +%.2f%%",
        (lr_metrics["pr_auc"] - dummy_mf_metrics["pr_auc"]) * 100,
    )

    logger.info("\n=== Comparação geral (LogisticRegression vs. DummyClassifier(stratified)) ===")
    logger.info(
        "Ganho de acurácia vs baseline: +%.2f%%",
        (lr_metrics["accuracy"] - dummy_strat_metrics["accuracy"]) * 100,
    )
    logger.info(
        "Ganho de F1 (classe 1) vs baseline: +%.2f%%",
        (lr_metrics["f1"] - dummy_strat_metrics["f1"]) * 100,
    )
    logger.info(
        "Ganho de PR-AUC vs baseline: +%.2f%%",
        (lr_metrics["pr_auc"] - dummy_strat_metrics["pr_auc"]) * 100,
    )


if __name__ == "__main__":
    main()
