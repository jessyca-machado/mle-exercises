from __future__ import annotations
from random import uniform
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient



def _cast_param_value(name: str, value: Any) -> Any:
    """
    MLflow armazena params como string. Converte tipos comuns do XGBoost.
    Ajuste conforme o seu espaço de busca.

    Se não reconhecer, retorna o valor original.

    Args:
        - name: nome do parâmetro (ex.: "max_depth")
        - value: valor do parâmetro (ex.: "5" ou "0.1")
    """
    if value is None:
        return None

    if isinstance(value, str) and value.lower() == "none":
        return None

    int_params = {
        "max_depth",
        "n_estimators",
        "select_kbest__k",
    }
    float_params = {
        "min_child_weight",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "scale_pos_weight",
    }

    try:
        if name in int_params:
            return int(float(value))
        if name in float_params:
            return float(value)
    except Exception:
        return value

    return value


def _cast_kbest(value: Any) -> Union[int, str]:
    """
    Trata o valor de k do SelectKBest, permitindo "all" ou um número inteiro.

    Args:
        - value: valor do parâmetro (ex.: "10", "all", None)
    
    Returns:
        Total de features a serem selecionadas, ou "all" para manter todas.
    """
    if value is None:
        return "all"
    if isinstance(value, str):
        if value.lower() == "all":
            return "all"
        try:
            return int(float(value))
        except Exception:
            return value
    if isinstance(value, (int, float)):
        return int(value)
    return value


@dataclass(frozen=True)
class BestXGBFromSearch:
    """
    Resultado pronto para uso do melhor run do RandomizedSearchCV.
    """
    run_id: str
    best_cv_score: float
    metric_key: str
    xgb_params: Dict[str, Any]
    select_kbest_k: Union[int, str]


def fetch_best_xgb_params_from_mlflow(
    *,
    experiment_name: str,
    tracking_uri: str,
    metric_key: str = "best_cv_score",
    search_type_value: str = "randomized",
    search_type_field: str = "params.search_type",
    client: Optional[MlflowClient] = None,
    ) -> BestXGBFromSearch:
    """
    Busca o melhor run do RandomizedSearchCV no MLflow e extrai:

    - `xgb_params`: params do step "model" (model__<param>) para passar no XGBClassifier
    - `select_kbest_k`: valor do select_kbest__k para configurar o trainer

    Não depende de `param_prefix`, porque entende o padrão do sklearn Pipeline:
        - model__*         -> XGBClassifier kwargs
        - select_kbest__k  -> k do SelectKBest

    Requisitos:
        - no run do search, você logou:
            mlflow.log_param("search_type", "randomized")
            mlflow.log_metric("best_cv_score", rs.best_score_)
            mlflow.log_params(rs.best_params_)

    Args:
        - experiment_name: nome do experimento no MLflow onde o search foi logado
        - tracking_uri: URI do MLflow Tracking Server (ex.: "http://localhost:5000")
        - metric_key: nome da métrica onde o score do search está logado (ex.: "best_cv_score")
        - search_type_value: valor do param "search_type" que identifica o run do search (ex.: "randomized")
        - search_type_field: nome do param que identifica o tipo do search (ex.: "params.search_type")
        - client: instância opcional do MlflowClient para evitar criar uma nova conexão
    
    Returns:
        - BestXGBFromSearch: dataclass com run_id, best_cv_score, xgb_params e select_kbest_k
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = client or MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experimento não encontrado: {experiment_name}")

    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"{search_type_field} = '{search_type_value}'",
        order_by=[f"metrics.{metric_key} DESC"],
        max_results=1,
    )
    if df.empty:
        raise ValueError(
            f"Nenhum run encontrado com {search_type_field}='{search_type_value}' "
            f"no experimento '{experiment_name}'."
        )

    row = df.iloc[0].to_dict()

    run_id = str(row.get("run_id") or row.get("info.run_id") or "")
    best_score = float(row.get(f"metrics.{metric_key}", float("nan")))

    xgb_params: Dict[str, Any] = {}
    k_best: Union[int, str] = "all"

    for key, val in row.items():
        if not key.startswith("params."):
            continue
        p = key[len("params.") :]

        if p.startswith("model__"):
            clean = p[len("model__") :]
            xgb_params[clean] = _cast_param_value(clean, val)

        elif p == "select_kbest__k":
            k_best = _cast_kbest(val)

    if not xgb_params:
        raise ValueError(
            "Não encontrei parâmetros do XGBoost no run (esperado params.model__*). "
            "Verifique se você logou rs.best_params_ do RandomizedSearchCV."
        )

    return BestXGBFromSearch(
        run_id=run_id,
        best_cv_score=best_score,
        metric_key=metric_key,
        xgb_params=xgb_params,
        select_kbest_k=k_best,
    )
