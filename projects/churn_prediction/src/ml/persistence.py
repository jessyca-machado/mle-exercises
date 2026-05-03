from __future__ import annotations

from pathlib import Path

import joblib


def save_joblib(model, path: str | Path) -> Path:
    """Salva um modelo usando joblib, garantindo que o diretório exista.

    Args:
        model: O modelo a ser salvo.
        path: O caminho onde o modelo será salvo.

    Returns:
        O caminho onde o modelo foi salvo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_and_log_joblib(
    model,
    path: str | Path,
    mlflow_module,
    artifact_path: str = "saved_model_joblib",
    logger=None,
    log_prefix: str = "Persistindo modelo via joblib",
) -> Path:
    """
    Salva com joblib e loga como artifact no MLflow.
    - mlflow_module: passe o módulo mlflow (para facilitar testes/evitar import circular)
    - logger: opcional (pode ser ModelLogger ou logger normal)
    """
    p = Path(path)
    if logger:
        logger.info("%s: %s", log_prefix, p)

    saved_path = save_joblib(model, p)
    mlflow_module.log_artifact(str(saved_path), artifact_path=artifact_path)

    if logger:
        logger.info("Modelo salvo e logado no MLflow: %s", saved_path)

    return saved_path
