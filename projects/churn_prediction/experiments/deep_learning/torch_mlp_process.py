"""
torch_mlp_process.py

Executa o MLP (PyTorch) dentro de um Pipeline scikit-learn usando SklearnPipelineRunner:
    load_data -> pre_processing -> split -> pipeline(preprocess + TorchMLPClassifier) -> evaluate -> save
Uso:
    python experiments/deep_learning/torch_mlp_process.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import numpy as np
import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import (
    FEATURES_COLS, YES_NO_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE,
    CAT_COLS, NUM_COLS, BOL_COLS, BIN_COLS
)

from src.data.pipelines import SklearnPipelineRunner
from experiments.deep_learning.torch_mlp import TorchMLPClassifier
from src.data.feature_engineering import TelcoFeatureEngineeringBins

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ModelLogger(logging.LoggerAdapter):
    """Injecta model_name no campo extra do log (opcional)."""
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"]["model_name"] = self.extra.get("model_name", "TorchMLP")
        return msg, kwargs


def _setup_mlflow() -> None:
    """
    Configura tracking URI e experiment name via variáveis de ambiente.
    Args:
        None
    Returns:
        None
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-deep-learning")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configurado: tracking_uri=%s experiment=%s", tracking_uri, experiment_name)


def build_mlp_model(y_train=None) -> TorchMLPClassifier:
    """
    Constrói e configura o classificador MLP em PyTorch (wrapper sklearn).
        - Define arquitetura, hiperparâmetros de treino e early stopping
        - (Opcional) calcula pos_weight para lidar com desbalanceamento
    Args:
        y_train: Série/array de rótulos do treino (opcional) para calcular pos_weight.
    Returns:
        TorchMLPClassifier: modelo configurado.
    """
    model = TorchMLPClassifier(random_state=RANDOM_STATE)

    # arquitetura / treino
    model.hidden_dims = (128, 64)
    model.dropout = 0.2
    model.lr = 1e-3
    model.batch_size = 128
    model.epochs = 50

    # early stopping
    model.val_size = 0.2
    model.patience = 7
    model.min_delta = 0.0

    # logs do treino
    model.verbose = 1

    if y_train is not None:
        y_arr = np.asarray(y_train)
        n_pos = np.sum(y_arr == 1)
        n_neg = np.sum(y_arr == 0)
        if n_pos > 0:
            model.pos_weight = float(n_neg / n_pos)
            logger.info("pos_weight configurado: %.4f (n_neg=%d, n_pos=%d)", model.pos_weight, n_neg, n_pos)

    logger.info(
        "MLP configurado: hidden_dims=%s dropout=%.3f lr=%g batch_size=%d epochs=%d patience=%d",
        model.hidden_dims, model.dropout, model.lr, model.batch_size, model.epochs, model.patience
    )

    return model


def _log_mlp_params_to_mlflow(model: TorchMLPClassifier) -> None:
    """
    Loga hiperparâmetros relevantes do TorchMLPClassifier no MLflow.
    Args:
        model: TorchMLPClassifier configurado.
    Returns:
        None
    """
    mlflow.log_params({
        "mlp_hidden_dims": str(model.hidden_dims),
        "mlp_dropout": model.dropout,
        "mlp_lr": model.lr,
        "mlp_batch_size": model.batch_size,
        "mlp_epochs": model.epochs,
        "mlp_weight_decay": model.weight_decay,
        "mlp_val_size": model.val_size,
        "mlp_patience": model.patience,
        "mlp_min_delta": model.min_delta,
        "mlp_pos_weight": model.pos_weight,
        "mlp_device": model.device,
    })


def _log_pipeline_params_to_mlflow(runner: SklearnPipelineRunner, use_fe: bool, use_fs: bool) -> None:
    """
    Loga parâmetros do pipeline (pré-processamento/FE/FS) no MLflow.
    Args:
        runner: SklearnPipelineRunner já instanciado.
        use_fe: bool indicando feature engineering.
        use_fs: bool indicando feature selection.
    Returns:
        None
    """
    mlflow.log_params({
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "use_feature_engineering": use_fe,
        "use_feature_selection": use_fs,
        "k_best": runner.k_best if use_fs else None,
        "n_cat_cols": len(runner.categorical_cols),
        "n_num_cols": len(runner.numerical_cols),
        "n_bool_cols": len(runner.boolean_cols),
        "n_bin_cols": len(runner.binned_cols),
        "scoring": runner.scoring,
        "cv": runner.cv,
    })


def main() -> None:
    """
    Executa o processo completo com MLflow:
        - Configura MLflow
        - Carrega e limpa dados
        - Split treino/teste
        - Treina pipeline com TorchMLPClassifier
        - Avalia e registra métricas
        - Registra/salva modelo
    Args:
        None
    Returns:
        None
    """
    _setup_mlflow()
    if mlflow.active_run() is not None:
        mlflow.end_run()

    model_name = "TorchMLP"
    mlog = ModelLogger(logger, {"model_name": model_name})

    mlog.info("Iniciando torch_mlp_process (com MLflow)")

    use_fe = bool(int(os.getenv("USE_FEATURE_ENGINEERING", "1")))
    use_fs = bool(int(os.getenv("USE_FEATURE_SELECTION", "0")))

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

    mlog.info(
        "Dataset: %d features | Treino: %d | Teste: %d",
        X_train.shape[1], len(X_train), len(X_test)
    )

    fe_transformer = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    mlp_model = build_mlp_model(y_train=y_train)

    runner = SklearnPipelineRunner(
        model=mlp_model,
        categorical_cols=CAT_COLS,
        numerical_cols=NUM_COLS,
        boolean_cols=BOL_COLS,
        binned_cols=BIN_COLS,
        use_feature_engineering=use_fe,
        feature_engineering_transformer=fe_transformer if use_fe else None,
        use_feature_selection=use_fs,
        use_grid_search=False,
        param_grid=None,
        cv=5,
        scoring="accuracy",
        pos_label=1,
    )

    run_name = os.getenv("MLFLOW_RUN_NAME", "torch_mlp_pipeline")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": "churn",
            "task": "binary_classification",
            "job": "deep_learning",
            "model_name": model_name,
        })

        _log_pipeline_params_to_mlflow(runner, use_fe=use_fe, use_fs=use_fs)
        _log_mlp_params_to_mlflow(mlp_model)

        mlog.info("Treinando pipeline...")
        runner.fit(X_train, y_train)

        mlog.info("Avaliando no teste...")
        metrics = runner.evaluate(X_test, y_test, include_auc=True)
        mlog.info("Métricas: %s", metrics)
        mlflow.log_metrics(metrics)

        out_path = Path(os.getenv("MODEL_OUT_PATH", "churn_torch_mlp_pipeline.joblib"))
        joblib.dump(runner.best_model, out_path)
        mlog.info("Pipeline salvo localmente em: %s", out_path)

        mlflow.log_artifact(str(out_path), artifact_path="artifacts")

        try:
            mlflow.sklearn.log_model(
                sk_model=runner.best_model,
                artifact_path="model",
                pip_requirements="requirements.txt",
            )
            mlog.info("Modelo registrado no MLflow em artifact_path=model")
        except Exception as e:
            mlog.info("Falha ao registrar mlflow.sklearn.log_model (seguindo só com artifact joblib). Erro: %s", e)

        mlog.info("MLflow run_id: %s", run.info.run_id)

    mlog.info("Finalizado torch_mlp_process")


if __name__ == "__main__":
    main()
