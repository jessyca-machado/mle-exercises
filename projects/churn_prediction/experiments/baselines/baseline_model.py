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
import logging
from pathlib import Path
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.pipelines import SklearnPipelineRunner

from src.utils.constants import (
    FEATURES_COLS,
    YES_NO_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    CAT_COLS,
    NUM_COLS,
    BOL_COLS,
    BIN_COLS,
)


class EnsureModelNameFilter(logging.Filter):
    """Garante que todo LogRecord tenha o campo model_name para o formatter não quebrar."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "model_name"):
            record.model_name = "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(model_name)s] %(name)s: %(message)s",
)
logging.getLogger().addFilter(EnsureModelNameFilter())
logger = logging.getLogger(__name__)


class ModelLogger(logging.LoggerAdapter):
    """Injecta model_name no campo extra do log (não prefixa msg manualmente)."""
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["model_name"] = self.extra["model_name"]
        return msg, kwargs


def _setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-baselines")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def main() -> None:
    _setup_mlflow()
    if mlflow.active_run() is not None:
        mlflow.end_run()

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")
    X_raw = df_clean[FEATURES_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() == 2 else None,
    )

    logger.info("Treino: %d | Teste: %d", len(X_train), len(X_test))

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logreg": LogisticRegression(random_state=RANDOM_STATE),
    }

    best = {"model_name": None, "accuracy": float("-inf")}
    rows: list[dict] = []

    with mlflow.start_run(run_name="baseline_comparison") as parent_run:
        mlflow.set_tags({
            "project": "churn",
            "task": "binary_classification",
            "job": "baseline_models",
        })

        mlflow.log_params({
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "use_feature_engineering": False,
            "use_feature_selection": False,
            "k_best": None,
            "n_models": len(models),
        })

        for model_name, model in models.items():
            mlog = ModelLogger(logger, {"model_name": model_name})

            with mlflow.start_run(run_name=model_name, nested=True):
                mlog.info("Treinando %s", model.__class__.__name__)

                mlflow.set_tag("model_name", model_name)
                mlflow.log_param("estimator_class", model.__class__.__name__)

                runner = SklearnPipelineRunner(
                    model=model,
                    categorical_cols=CAT_COLS,
                    numerical_cols=NUM_COLS,
                    boolean_cols=BOL_COLS,
                    binned_cols=BIN_COLS,
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

                runner.fit(X_train, y_train)

                mlog.info("Avaliando no teste...")
                metrics = runner.evaluate(X_test, y_test, include_auc=True)
                mlog.info("Métricas: %s", metrics)

                rows.append({"model": model_name, **metrics})

                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    sk_model=runner.best_model,
                    name="model",
                    serialization_format="skops",
                    pip_requirements="requirements.txt",
                )

                acc = metrics.get("accuracy", float("-inf"))
                if acc > best["accuracy"]:
                    best = {"model_name": model_name, "accuracy": acc}

        results_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        logger.info("Resumo métricas por modelo:\n%s", results_df.to_string(index=False))

        mlflow.log_params({"best_model_name": best["model_name"]})
        mlflow.log_metric("best_accuracy", best["accuracy"])
        logger.info("Melhor modelo: %s | accuracy=%.4f", best["model_name"], best["accuracy"])
        logger.info("Parent run_id: %s", parent_run.info.run_id)

    if os.getenv("SAVE_BEST_JOBLIB", "0") == "1":
        best_name = best["model_name"]
        if best_name is None:
            return

        best_model = models[best_name]
        mlog = ModelLogger(logger, {"model_name": best_name})
        mlog.info("Re-treinando melhor modelo para salvar em joblib...")

        runner = SklearnPipelineRunner(
            model=best_model,
            categorical_cols=CAT_COLS,
            numerical_cols=NUM_COLS,
            boolean_cols=BOL_COLS,
            binned_cols=BIN_COLS,
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
        runner.fit(X_train, y_train)

        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"baseline_best_{best_name}.joblib"
        joblib.dump(runner.best_model, path)
        mlog.info("Melhor modelo salvo em: %s", path)

if __name__ == "__main__":
    main()
