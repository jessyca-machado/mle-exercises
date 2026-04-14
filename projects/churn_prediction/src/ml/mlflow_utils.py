from __future__ import annotations
import os
import mlflow
import shutil

def setup_mlflow(default_experiment: str) -> None:
    """
    Configura o MLflow para usar o tracking URI e o nome do experimento definidos nas variáveis de ambiente.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", default_experiment)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def end_active_run() -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _safe_log_params(logger, params: Dict[str, Any]) -> None:
    logger.debug("Logging param no MLflow (%d keys)", len(params or {}))
    for k, v in (params or {}).items():
        if v is None:
            continue
        try:
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)
            else:
                mlflow.log_param(k, str(v))
        except Exception as e:
            logger.warning("Falha ao logar param '%s': %s", k, e)


def _safe_log_metrics(logger, metrics: Dict[str, Any]) -> None:
    logger.debug("Logging metrics no MLflow (%d keys)", len(metrics or {}))
    for k, v in (metrics or {}).items():
        if v is None:
            continue
        try:
            mlflow.log_metric(k, float(v))
        except Exception as e:
            pass
            logger.warning("Falha ao logar metric '%s': %s", k, e)


def _mlflow_set_tags(logger, tags: Dict[str, str]) -> None:
    logger.debug("Setting MLflow tags (%d keys)", len(tags or {}))
    for k, v in (tags or {}).items():
        try:
            mlflow.set_tag(k, v)
        except Exception as e:
            logger.warning("Falha ao setar tag '%s': %s", k, e)


def _cleanup_dir(logger, d: Path) -> None:
    try:
        if d.exists():
            shutil.rmtree(d)
            logger.debug("Diretório temporário removido: %s", d)
    except Exception as e:
        logger.warning("Falha ao limpar diretório temporário %s: %s", d, e)
