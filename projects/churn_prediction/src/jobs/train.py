"""Treino — pipeline de treino para XGBoost + registro via MLflow PyFunc,
utilizando hiperparâmetros provenientes do melhor run de RandomizedSearchCV no MLflow.

Características:
- Treina via ChurnModelTrainer.
- Busca best params do XGB no MLflow (função `fetch_best_xgb_params_from_mlflow`).
- Loga e registra um modelo END-TO-END como PyFunc (transformações + estimador).
- Preparado para pytest (funções pequenas, dependências injetáveis).

Pré-requisitos:
- ChurnModelTrainer deve montar um pipeline com steps nomeados:
    "feature_engineering", "preprocess", "select_kbest", "model".
- Pyfunc compatível com XGB:
    `src/ml/churn_pyfunc_xgb.py`, que espera artifacts:
    - feature_engineering
    - preprocessor
    - selector
    - estimator
    e retorna DataFrame com coluna `y_pred_proba`.

Uso:
    python src/jobs/train.py

Para visualizar:
    mlflow ui # Inicia UI em http://localhost:5000
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
import pandas as pd
import pytz
import skops.io as sio
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.core.models.trainer import ChurnModelTrainer
from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.entrypoints.cli import parse_args
from src.ml.mlflow_utils import setup_mlflow_sqlite
from src.utils.constants import (
    FEATURES_COLS,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    PRIMARY_METRIC,
    RANDOM_SEED,
    TARGET_COL,
    YES_NO_COLS,
)

BEST_MODEL_CONFIG_PATH_DEFAULT = Path("configs/best_model.yml")


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str
    registered_model_name: str
    tracking_uri: str = MLFLOW_TRACKING_URI
    registry_uri: str = MLFLOW_TRACKING_URI
    timezone: str = "America/Sao_Paulo"
    n_folds: int = N_FOLDS
    random_seed: int = RANDOM_SEED
    primary_metric: str = PRIMARY_METRIC
    params_prefix: str = "model__"
    pyfunc_code_path: str = "src/ml/churn_pyfunc_xgb.py"
    pip_requirements: Optional[Union[str, list[str]]] = "requirements-mlflow.txt"
    best_model_config_path: Path = BEST_MODEL_CONFIG_PATH_DEFAULT


def load_best_model_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de best model não encontrado: {path}. "
            "Crie `configs/best_model.yml` ou ajuste o caminho."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido em {path}: esperado dict no topo.")
    return data


def log_xgb_end_to_end_pyfunc(
    *,
    fitted_pipeline,
    name: str,
    pyfunc_code_path: str,
    pip_requirements: Optional[Union[str, list[str]]],
    input_example: pd.DataFrame,
) -> str:
    """
    Loga o modelo como MLflow PyFunc com artifacts compatíveis com `src/ml/churn_pyfunc_xgb.py`.

    Espera steps no pipeline:
        - feature_engineering
        - preprocess
        - select_kbest
        - model

    Args:
        fitted_pipeline: pipeline completo (com steps nomeados) treinado.
        name: nome do modelo registrado no MLflow.
        pyfunc_code_path: caminho para o código do pyfunc (ex: `src/ml/churn_pyfunc_xgb.py`).
        pip_requirements: requisitos pip para o ambiente do pyfunc.
        input_example: exemplo de DataFrame de features para inferir a assinatura.

    Returns:
        model_uri retornado pelo MLflow.
    """
    fe = fitted_pipeline.named_steps["feature_engineering"]
    pre = fitted_pipeline.named_steps["preprocess"]
    sel = fitted_pipeline.named_steps.get("select_kbest", None)
    if sel is None:
        sel = FunctionTransformer(lambda x: x)

    est = fitted_pipeline.named_steps["model"]

    tmp = Path("mlflow_artifacts_tmp")
    tmp.mkdir(exist_ok=True)

    fe_path = tmp / "feature_engineering.skops"
    pre_path = tmp / "preprocessor.skops"
    sel_path = tmp / "selector.skops"
    est_path = tmp / "estimator.skops"

    sio.dump(fe, fe_path)
    sio.dump(pre, pre_path)
    sio.dump(sel, sel_path)
    sio.dump(est, est_path)

    input_example = input_example.copy()
    int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns
    if len(int_cols) > 0:
        input_example[int_cols] = input_example[int_cols].astype("float64")

    y_example = fitted_pipeline.predict_proba(input_example)[:, 1]
    output_example = pd.DataFrame({"y_pred_proba": y_example.astype(float)})

    signature = infer_signature(input_example, output_example)

    model_info = mlflow.pyfunc.log_model(
        artifact_path=name,
        python_model=str(Path(pyfunc_code_path).resolve()),
        artifacts={
            "feature_engineering": str(fe_path.resolve()),
            "preprocessor": str(pre_path.resolve()),
            "selector": str(sel_path.resolve()),
            "estimator": str(est_path.resolve()),
        },
        pip_requirements=pip_requirements,
        input_example=input_example,
        signature=signature,
    )

    return model_info.model_uri


def run_train_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainConfig,
    trainer: Optional[ChurnModelTrainer] = None,
    client: Optional[MlflowClient] = None,
) -> dict[str, Any]:
    """
    Treina o modelo final XGB usando o melhor resultado do RandomizedSearchCV no MLflow,
    roda CV (via ChurnModelTrainer) e registra via MLflow PyFunc.

    Args:
        X: DataFrame de features.
        y: Series de target.
        config: TrainConfig com configurações de treino e MLflow.
        trainer: ChurnModelTrainer opcional (para injeção em testes).
        client: MlflowClient opcional (para injeção em testes).

    Returns:
        Dicionário com informações do run, modelo e melhores parâmetros.
    """
    if str(config.tracking_uri).startswith("sqlite:"):
        setup_mlflow_sqlite(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            artifact_root=MLFLOW_ARTIFACT_ROOT,
        )
    else:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_registry_uri(config.registry_uri)
        mlflow.set_experiment(config.experiment_name)

    best_yaml = load_best_model_yaml(config.best_model_config_path)

    model_name = best_yaml.get("model_name")
    if model_name not in ("xgboost", "xgb"):
        raise ValueError(
            f"best_model.yml: model_name='{model_name}' não suportado. "
            "Este train.py espera 'xgboost'."
        )

    xgb_params = best_yaml.get("xgb_params", {})
    if not isinstance(xgb_params, dict):
        raise ValueError("best_model.yml: xgb_params deve ser um dict.")

    select_kbest_k = best_yaml.get("select_kbest__k", "all")
    meta = best_yaml.get("meta", {}) if isinstance(best_yaml.get("meta", {}), dict) else {}

    source_run_id = meta.get("run_id")
    metric_key = meta.get("metric_key")
    best_cv_score = meta.get("best_cv_score")
    experiment_name = meta.get("experiment_name")
    source = meta.get("source")

    estimator = XGBClassifier(
        random_state=config.random_seed,
        n_jobs=-1,
        eval_metric="aucpr",
        verbosity=0,
        **xgb_params,
    )

    trainer = trainer or ChurnModelTrainer(
        n_folds=config.n_folds,
        seed=config.random_seed,
        k_best=select_kbest_k,
    )
    trainer.build(X=X, y=y, model=estimator)
    cv_summary = trainer.train()

    tz = pytz.timezone(config.timezone)
    run_name = f"{datetime.now(tz).strftime('%Y%m%d%H%M%S')}__xgb__final_train"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics({f"cv_mean_{k}": v for k, v in cv_summary.metrics_mean.items()})
        mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_summary.metrics_std.items()})

        mlflow.set_tag("run_type", "final_train")
        mlflow.set_tag("params_source", "configs/best_model.yml")
        mlflow.log_param("run_type", "final_train")
        mlflow.log_param("model_family", "xgboost")
        mlflow.log_param("cv_folds", config.n_folds)
        mlflow.log_param("random_seed", config.random_seed)
        mlflow.log_param("primary_metric", config.primary_metric)

        if source_run_id:
            mlflow.log_param("source_search_run_id", source_run_id)
        if metric_key is not None:
            mlflow.log_param("source_search_metric_key", str(metric_key))
        if best_cv_score is not None:
            mlflow.log_metric("source_search_best_cv_score", float(best_cv_score))
        if experiment_name is not None:
            mlflow.log_param("source_search_experiment_name", str(experiment_name))
        if source is not None:
            mlflow.log_param("source_search_source", str(source))

        mlflow.log_param("select_kbest__k", str(select_kbest_k))
        mlflow.log_params({f"{config.params_prefix}{k}": v for k, v in xgb_params.items()})

        if config.best_model_config_path.exists():
            mlflow.log_artifact(str(config.best_model_config_path), artifact_path="config")

        input_example = X.head(50).copy()
        model_uri = log_xgb_end_to_end_pyfunc(
            fitted_pipeline=cv_summary.fitted_pipeline,
            name=config.registered_model_name,
            pyfunc_code_path=config.pyfunc_code_path,
            pip_requirements=config.pip_requirements,
            input_example=input_example,
        )

        mv = mlflow.register_model(model_uri=model_uri, name=config.registered_model_name)
        mlf_client = client or MlflowClient()
        mlf_client.set_registered_model_alias(config.registered_model_name, "prod", mv.version)

        mlflow.set_tag("registered_model_version", str(mv.version))
        mlflow.set_tag("registered_model_alias", "prod")

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model_uri": model_uri,
            "registered_name": config.registered_model_name,
            "best_params": xgb_params,
            "k_best": select_kbest_k,
            "source_search_run_id": source_run_id,
            "source_search_best_cv_score": best_cv_score,
            "cv_mean": cv_summary.metrics_mean,
            "cv_std": cv_summary.metrics_std,
        }


def main() -> None:
    args = parse_args()

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features", verbose=False)

    y = df_clean[TARGET_COL].astype(int)
    X = df_clean[FEATURES_COLS].copy()

    cfg = TrainConfig(
        experiment_name=args.experiment_name,
        registered_model_name=args.model_name,
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        primary_metric=args.primary_metric,
        pyfunc_code_path=args.pyfunc_code_path,
        pip_requirements=args.pip_requirements,
    )

    run_train_pipeline(X=X, y=y, config=cfg)


if __name__ == "__main__":
    main()
