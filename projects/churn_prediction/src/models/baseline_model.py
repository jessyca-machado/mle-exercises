"""Baseline model — treinamento de modelos heurístico e de regressão logística.

- Demonstra como estabelecer baselines sólidas antes de modelos complexos.
- Inclui logging de hiperparâmetros e métricas para comparação.
- Corrige unpack do clean_data (retorna df_clean, X, y).
- Loga distribuição de classes (para evidenciar desbalanceamento).
- Usa dois baselines:
    1) most_frequent (baseline "ingênuo" que pode prever só a classe 0)
    2) stratified (baseline que sorteia classes conforme a proporção, evitando prever só 0)
- Usa StandardScaler para normalização dos dados na pipeline LogisticRegression.

Uso:
    python src/models/baseline_model.py
"""
import logging
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_data_churn
from src.data.clean_data import clean_data
from src.utils.helpers import log_class_distribution
from src.utils.constants import RANDOM_STATE, TEST_SIZE, FEATURES_COLS, YES_NO_COLS

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
        Dicionário com métricas: accuracy, auc_roc.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info("\n--- %s ---", model_name)
    logger.info(classification_report(y_test, y_pred))

    metrics: dict[str, float] = {}
    metrics["accuracy"] = float((y_pred == y_test).mean())

    metrics["f1"] = float(f1_score(y_test, y_pred, pos_label=1))
    logger.info("F1 (classe 1): %.4f", metrics["f1"])

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
        logger.info("AUC-ROC: %.4f", metrics["auc_roc"])

        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        logger.info("PR-AUC (Average Precision): %.4f", metrics["pr_auc"])

    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
    logger.info(
        "Distribuição das predições (classe -> contagem): %s",
        dict(zip(unique_preds.tolist(), pred_counts.tolist())),
    )

    return metrics


def main() -> None:
    """Executa comparação entre baselines e regressão logística."""
    df = load_data_churn()

    df_clean, X, y = clean_data(df, FEATURES_COLS, YES_NO_COLS, "Cleaned dataset and features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("Treino: %d | Teste: %d", len(X_train), len(X_test))
    name, counts, ratios = log_class_distribution(y_train, "y_train")
    logger.info("%s | contagens:\n%s", name, counts.to_string())
    logger.info("%s | proporções:\n%s", name, ratios.to_string())


    name, counts, ratios = log_class_distribution(y_test, "y_test")
    logger.info("%s | contagens:\n%s", name, counts.to_string())
    logger.info("%s | proporções:\n%s", name, ratios.to_string())

    dummy_mf_pipeline = Pipeline(
        [
            ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]
    )
    dummy_mf_metrics = train_and_evaluate(
        dummy_mf_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        "DummyClassifier (most_frequent)",
    )

    dummy_strat_pipeline = Pipeline(
        [
            ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
        ]
    )
    dummy_strat_metrics = train_and_evaluate(
        dummy_strat_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        "DummyClassifier (stratified)",
    )

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    lr_metrics = train_and_evaluate(
        lr_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        "LogisticRegression (balanced)",
    )

    logger.info("\n=== Comparação (Accuracy) ===")
    logger.info("F1 DummyClassifier(most_frequent): %.4f", dummy_mf_metrics["accuracy"])
    logger.info("F1 DummyClassifier(stratified):   %.4f", dummy_strat_metrics["accuracy"])
    logger.info("F1 LogisticRegression:  %.4f", lr_metrics["accuracy"])

    logger.info("\n=== Comparação (F1 classe 1) ===")
    logger.info("F1 DummyClassifier(most_frequent): %.4f", dummy_mf_metrics["f1"])
    logger.info("F1 DummyClassifier(stratified):   %.4f", dummy_strat_metrics["f1"])
    logger.info("F1 LogisticRegression:  %.4f", lr_metrics["f1"])

    logger.info("\n=== Comparação (PR-AUC) ===")
    logger.info("PR-AUC DummyClassifier(most_frequent): %.4f", dummy_mf_metrics["pr_auc"])
    logger.info("PR-AUC DummyClassifier(stratified):    %.4f", dummy_strat_metrics["pr_auc"])
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
