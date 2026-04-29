"""
Uso:
    python experiments/selection/compare_models.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


Objective = Literal["net_value", "total_cost"]


@dataclass(frozen=True)
class CostSpec:
    cost_fp: float = 1.0 #paga cost_fp por cada falso positivo (FP)
    cost_fn: float = 5.0 #paga cost_fn por cada falso negativo (FN)
    benefit_tp: float = 10.0 #recebe benefit_tp por cada verdadeiro positivo (TP)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    require_both_classes: bool = True,
) -> dict[str, float]:
    """Estima intervalo de confiança da métrica via bootstrap.

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        n_bootstrap: Número de amostras bootstrap.
        confidence: Nível de confiança (e.g., 0.95 para 95%).

    Returns:
        Dicionário com a métrica média, std, lower e upper bound.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    vals: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if require_both_classes and len(np.unique(yt)) < 2:
            continue
        vals.append(float(metric_fn(yt, ys)))

    vals_arr = np.asarray(vals, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(vals_arr)),
        "std": float(np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else 0.0,
        "lower": float(np.percentile(vals_arr, 100 * alpha)),
        "upper": float(np.percentile(vals_arr, 100 * (1 - alpha))),
        "n_samples": float(len(vals_arr)),
    }


def make_threshold_grid(
    mode: str = "linspace",
    n: int = 201,
    fixed: Optional[Iterable[float]] = None,
) -> np.ndarray:
    if fixed is not None:
        return np.asarray(list(fixed), dtype=float)

    if mode == "toolkit":
        return np.asarray([0.3, 0.5, 0.7], dtype=float)

    # default: linspace
    return np.linspace(0.0, 1.0, int(n))


def sweep_thresholds_cost(
    y_true,
    y_score,
    spec: CostSpec,
    thresholds: Optional[Iterable[float]] = None,
    objective: Objective = "net_value",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Análise custo-benefício para classificação binária.

    Args:
        y_true: Labels verdadeiros.
        y_score: Probabilidades preditas.
        spec: dicionátio com:
            - cost_fp: Custo de um falso positivo.
            - cost_fn: Custo de um falso negativo.
            - benefit_tp: Benefício de um verdadeiro positivo.
        threshold: Limiar de classificação.
        objective: Escolher o objetivo do trade-off
            - net_value: maximizar retorno financeiro líquido.
            - total_cost: minimizar prejuízo causado por erros.

    Returns:
        Dicionário com TP, FP, FN, TN, custo total e valor líquido.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if thresholds is None:
        thresholds = make_threshold_grid(mode="linspace", n=201)

    rows: list[dict[str, Any]] = []
    for t in thresholds:
        t = float(t)
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        total_cost = fp * spec.cost_fp + fn * spec.cost_fn
        net_value = (tp * spec.benefit_tp) - (fp * spec.cost_fp) - (fn * spec.cost_fn)

        rows.append(
            {
                "threshold": t,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "total_cost": float(total_cost),
                "net_value": float(net_value),
            }
        )

    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    if df.empty:
        return df, {}

    if objective == "net_value":
        best = df.loc[df["net_value"].idxmax()].to_dict()
    elif objective == "total_cost":
        best = df.loc[df["total_cost"].idxmin()].to_dict()
    else:
        raise ValueError(f"objective inválido: {objective}")

    return df, best
