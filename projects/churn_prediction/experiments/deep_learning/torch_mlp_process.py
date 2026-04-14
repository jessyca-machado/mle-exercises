"""
torch_mlp_process.py
Executa um MLP (PyTorch via skorch) dentro de um Pipeline scikit-learn e
otimiza hiperparâmetros com OptunaSearchCV, usando validação cruzada e scoring F1.

Uso:
    python experiments/deep_learning/torch_mlp_process.py

Observações importantes:
- Como você está usando SelectKBest(k=15), o input_dim do MLP é 15.
- Para reduzir atrito com skorch, recomenda-se que o preprocessor retorne numpy (não pandas).
    (Se o seu pipeline atual usa set_output(transform="pandas"), pode funcionar, mas se der erro
    de tipo, remova o set_output ou converta para numpy antes do skorch.)
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.pipelines import SklearnPipelineRunner
from src.ml.logging_utils import get_logger, ModelLogger
from src.ml.metrics_utils import save_confusion_matrix_artifacts
from src.ml.mlflow_utils import setup_mlflow, end_active_run
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
    TRUSTED_SKORCH_TORCH,
)

from src.data.feature_engineering import TelcoFeatureEngineeringBins

# IMPORTANTE: agora importamos o builder skorch (e a arquitetura) do torch_mlp.py modificado
from experiments.deep_learning.torch_mlp import build_skorch_mlp

import optuna

logger = get_logger(__name__)


def _log_pipeline_params_to_mlflow(runner: SklearnPipelineRunner, use_fe: bool, use_fs: bool) -> None:
    mlflow.log_params(
        {
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
        }
    )


def _log_search_params_to_mlflow(runner: SklearnPipelineRunner) -> None:
    # loga info geral do search; best_params e best_score logaremos após o fit
    mlflow.log_params(
        {
            "optuna_n_trials": getattr(runner, "n_trials", None),
            "optuna_timeout": getattr(runner, "timeout", None),
            "optuna_n_jobs": getattr(runner, "n_jobs", None),
            "optuna_cv": str(getattr(runner, "cv", None)),
            "optuna_scoring": getattr(runner, "scoring", None),
        }
    )


def _ensure_binary_int_y(y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Garante y no formato 0/1 int (skorch lida bem com isso; scoring f1 também).
    Se y já for 0/1, só converte dtype.
    Se for string/bool, tenta mapear.
    """
    y_arr = np.asarray(y)
    # casos comuns: {0,1}, {False,True}, {"No","Yes"} etc.
    if y_arr.dtype == bool:
        return y_arr.astype(np.int64)

    # se já for numérico
    if np.issubdtype(y_arr.dtype, np.number):
        return y_arr.astype(np.int64)

    # caso string/categorico: mapear por ordem de valores únicos
    uniq = pd.unique(y_arr)
    if len(uniq) != 2:
        raise ValueError(f"y precisa ser binário; encontrei {len(uniq)} classes: {uniq}")

    # tenta mapear algo tipo "No"/"Yes" -> 0/1
    # fallback: ordena e mapeia
    uniq_sorted = sorted(list(uniq))
    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    return np.vectorize(mapping.get)(y_arr).astype(np.int64)


def main() -> None:
    setup_mlflow(default_experiment="churn-deep-learning")
    end_active_run()

    model_name = "SkorchTorchMLP_Optuna"
    mlog = ModelLogger(logger, {"model_name": model_name})
    mlog.info("Iniciando torch_mlp_process com skorch + OptunaSearchCV (F1)")

    # 1) dados
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")

    X_raw = df_clean[FEATURES_COLS].copy()
    y_raw = df_clean[TARGET_COL].copy()
    y = _ensure_binary_int_y(y_raw).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) == 2 else None,
    )

    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    mlog.info("Dataset: %d features | Treino: %d | Teste: %d", X_train.shape[1], len(X_train), len(X_test))

    # 2) feature engineering transformer (opcional)
    fe_transformer = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)

    # 3) input_dim do MLP
    # Como você usa SelectKBest(k=15), a dimensão final que entra no MLP será 15.
    # Se você desligar FS, aí precisará calcular input_dim após OHE+scaling.
    k_best = 15
    input_dim = k_best

    # 4) cria o estimador skorch
    # sem early stopping (train_split=None) para evitar "holdout dentro do fold"
    net = build_skorch_mlp(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        dropout=0.2,
        lr=1e-3,
        batch_size=128,
        max_epochs=30,
        weight_decay=0.0,
        train_split=None,
        callbacks=None,
        verbose=0,
    )

    optuna_param_distributions = {
        "model__module__hidden_dims": optuna.distributions.CategoricalDistribution(
            choices=["64,32", "128,64", "256,128"]
        ),
        "model__module__dropout": optuna.distributions.FloatDistribution(0.0, 0.5),
        "model__lr": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),
        "model__batch_size": optuna.distributions.CategoricalDistribution(choices=[64, 128, 256]),
        "model__max_epochs": optuna.distributions.CategoricalDistribution(choices=[20, 30, 50]),
        "model__optimizer__weight_decay": optuna.distributions.FloatDistribution(1e-8, 1e-3, log=True),
    }

    # 6) monta o runner com pipeline + OptunaSearchCV "setado" no runner (MODIFICADO)
    runner = SklearnPipelineRunner(
        model=net,
        categorical_cols=CAT_COLS,
        numerical_cols=NUM_COLS,
        boolean_cols=BOL_COLS,
        binned_cols=BIN_COLS,
        use_feature_engineering=True,
        feature_engineering_transformer=fe_transformer,
        use_feature_selection=True,
        k_best=k_best,
        # MODIFICADO: habilita Optuna no runner (em vez de OptunaSearchCV manual)
        use_optuna_search=True,
        optuna_param_distributions=optuna_param_distributions,
        optuna_n_trials=30,
        optuna_timeout=None,
        optuna_n_jobs=1,  # recomendado com torch
        cv=5,
        scoring="f1",
        pos_label=1,
    )

    run_name = os.getenv("MLFLOW_RUN_NAME", "skorch_optuna_mlp_pipeline")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "churn",
                "task": "binary_classification",
                "job": "deep_learning",
                "model_name": model_name,
                "tuner": "optuna_search_cv",
            }
        )

        _log_pipeline_params_to_mlflow(runner, use_fe=True, use_fs=True)
        _log_search_params_to_mlflow(runner)

        mlog.info("Rodando OptunaSearchCV via runner (cv=%s, scoring=%s)...", runner.cv, runner.scoring)
        runner.fit(X_train, y_train)

        # MODIFICADO: best info agora vem do runner (que encapsula OptunaSearchCV)
        best_params = getattr(runner, "best_params_", None)
        cv_best_score = getattr(runner, "cv_best_score_", None)

        if best_params is not None:
            mlog.info("Best params: %s", best_params)
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        if cv_best_score is not None:
            mlog.info("Best CV score (mean): %.6f", float(cv_best_score))
            mlflow.log_metric("cv_best_f1", float(cv_best_score))

        best_pipeline = runner.best_model

        # avalia no teste (holdout)
        mlog.info("Avaliando no teste...")
        y_pred = best_pipeline.predict(X_test)

        # métricas básicas (reaproveita runner.evaluate, mas runner.best_model precisa ser setado)
        metrics = runner.evaluate(X_test, y_test, include_auc=True)
        mlog.info("Métricas teste: %s", metrics)
        mlflow.log_metrics(metrics)

        # confusion matrix
        mlog.info("Logging confusion matrix artifacts...")
        try:
            cm_dir = Path("artifacts/tmp") / f"cm_{model_name}"
            paths = save_confusion_matrix_artifacts(
                y_true=y_test,
                y_pred=y_pred,
                out_dir=cm_dir,
                labels=(0, 1),
                normalize=None,
                prefix="confusion_matrix",
            )
            mlflow.log_artifact(str(paths["csv"]), artifact_path="best/confusion_matrix")
            mlflow.log_artifact(str(paths["png"]), artifact_path="best/confusion_matrix")
        except Exception as e:
            mlog.warning("Falha ao gerar/logar confusion matrix: %s", e)

        # salvar pipeline
        out_path = Path(os.getenv("MODEL_OUT_PATH", "churn_skorch_optuna_mlp_pipeline.joblib"))
        joblib.dump(best_pipeline, out_path)
        mlog.info("Modelo salvo localmente em: %s", out_path)
        mlflow.log_artifact(str(out_path), artifact_path="artifacts")

        # registrar no MLflow como sklearn model (pipeline sklearn)
        try:
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                name="model",
                serialization_format="skops",
                pip_requirements="requirements.txt",
                skops_trusted_types=TRUSTED_SKORCH_TORCH,
            )
            mlog.info("Modelo registrado no MLflow em artifact_path=model")
        except Exception as e:
            mlog.info(
                "Falha ao registrar mlflow.sklearn.log_model (seguindo só com artifact joblib). Erro: %s",
                e,
            )

        mlog.info("MLflow run_id: %s", run.info.run_id)

        final_df = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "mean_cv": float(cv_best_score) if cv_best_score is not None else None,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "auc_roc": metrics.get("auc_roc"),
                }
            ]
        )
        mlog.info("Resumo final:\n%s", final_df.to_string(index=False))

    mlog.info("Finalizado torch_mlp_process (skorch + optuna)")


if __name__ == "__main__":
    main()
