from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import mlflow


def setup_mlflow(default_experiment: str) -> None:
    """
    Configura o MLflow para usar o tracking URI e o nome do experimento definidos nas variáveis de ambiente.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default_experiment)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")
    try:
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def setup_mlflow_sqlite(
    *,
    tracking_uri: str,
    experiment_name: str,
    artifact_root: str,
) -> None:
    """
    Configura MLflow para uso local com:
        - backend store em SQLite (tracking_uri = "sqlite:///mlflow.db")
        - artifact store em um diretório local controlado (artifact_root = "./mlartifacts")

    Observação:
    - `artifact_root` é usado apenas ao criar o experimento. Se o experimento já existir,
        o MLflow mantém o artifact_location existente.
    """
    mlflow.set_tracking_uri(tracking_uri)

    artifact_uri = Path(artifact_root).resolve().as_uri()
    exp = mlflow.get_experiment_by_name(experiment_name)

    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )

    mlflow.set_experiment(experiment_name)


def end_active_run() -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _safe_log_params(logger, params: Dict[str, Any]) -> None:
    """
    Loga parâmetros garantindo:
        - conversão para string
        - não logar valores None (MLflow rejeita ou fica inconsistente)
    """
    logger.debug("Logging param no MLflow (%d keys)", len(params or {}))
    for k, v in (params or {}).items():
        if v is None:
            continue
        try:
            mlflow.log_param(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        except Exception as e:
            logger.warning("Falha ao logar param '%s': %s", k, e)


def _safe_log_metrics(logger, metrics: Dict[str, Any]) -> None:
    """
    Loga métricas garantindo:
        - conversão para float
        - não logar NaN/Inf (MLflow frequentemente rejeita ou fica inconsistente)
        - não engolir silenciosamente erros (marca tag no run)
    """
    logger.debug("Logging metrics no MLflow (%d keys)", len(metrics or {}))
    for k, v in (metrics or {}).items():
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                logger.warning("Métrica '%s' é NaN/Inf (%r). Pulando log.", k, fv)
                try:
                    mlflow.set_tag(f"metric_skipped_{k}", "nan_or_inf")
                except Exception:
                    pass
                continue

            mlflow.log_metric(k, fv)

        except Exception as e:
            logger.exception(
                "Falha ao logar metric '%s' (valor=%r tipo=%s): %s",
                k, v, type(v).__name__, e
            )
            try:
                mlflow.set_tag(f"metric_error_{k}", f"{type(v).__name__}: {str(e)[:180]}")
            except Exception:
                pass


def _mlflow_set_tags(logger, tags: Dict[str, str]) -> None:
    """
    Configura tags no MLflow.
    """
    logger.debug("Setting MLflow tags (%d keys)", len(tags or {}))
    for k, v in (tags or {}).items():
        try:
            mlflow.set_tag(k, v)
        except Exception as e:
            logger.warning("Falha ao setar tag '%s': %s", k, e)


def _cleanup_dir(logger, d: Path) -> None:
    """
    Remove um diretório temporário, logando falhas mas sem interromper o fluxo.
    """
    try:
        if d.exists():
            shutil.rmtree(d)
            logger.debug("Diretório temporário removido: %s", d)
    except Exception as e:
        logger.warning("Falha ao limpar diretório temporário %s: %s", d, e)
