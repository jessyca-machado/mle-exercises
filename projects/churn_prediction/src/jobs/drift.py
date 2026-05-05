from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))

    if cuts.size < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1.0)
    act_perc = np.clip(act_perc, eps, 1.0)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


@dataclass(frozen=True)
class DriftConfig:
    dsn: str
    baseline_path: str
    window_days: int = 7
    n_bins: int = 10
    psi_alert_threshold: float = 0.2
    min_rows: int = 200


def load_baseline(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_production_features_sqlalchemy(dsn: str, window_days: int) -> pd.DataFrame:
    engine = create_engine(dsn)

    sql = text(
        """
        SELECT created_at, features, y_pred_proba
        FROM churn_predictions
        WHERE created_at >= now() - (:window_days || ' days')::interval
        ORDER BY created_at DESC
    """
    )

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"window_days": window_days})

    if df.empty:
        return df

    feats = pd.json_normalize(df["features"])
    out = pd.concat([df.drop(columns=["features"]), feats], axis=1)
    return out


def persist_drift_metrics(
    *,
    dsn: str,
    window_days: int,
    n_rows: int,
    psi_by_feature: dict[str, float],
    max_psi: float,
    baseline_model_uri: Optional[str],
    model_uri_current: Optional[str],
) -> None:
    engine = create_engine(dsn)

    sql = text(
        """
        INSERT INTO drift_metrics
            (window_days, n_rows, psi_json, max_psi, baseline_model_uri, model_uri_current)
        VALUES
            (
                :window_days,
                :n_rows, CAST(:psi_json AS jsonb),
                :max_psi,
                :baseline_model_uri,
                :model_uri_current
            )
    """
    )

    payload_json = json.dumps(psi_by_feature)

    with engine.begin() as conn:  # begin() = commit automático
        conn.execute(
            sql,
            {
                "window_days": int(window_days),
                "n_rows": int(n_rows),
                "psi_json": payload_json,
                "max_psi": float(max_psi),
                "baseline_model_uri": baseline_model_uri,
                "model_uri_current": model_uri_current,
            },
        )


def main() -> int:
    dsn = os.getenv("PREDICTIONS_DB_DSN")
    if not dsn:
        print("ERROR: env var PREDICTIONS_DB_DSN não definida.", file=sys.stderr)
        return 2

    baseline_path = os.getenv("DRIFT_BASELINE_PATH", "artifacts/baseline.json")
    window_days = int(os.getenv("DRIFT_WINDOW_DAYS", "7"))
    n_bins = int(os.getenv("DRIFT_N_BINS", "10"))
    psi_thr = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    min_rows = int(os.getenv("DRIFT_MIN_ROWS", "200"))

    # opcional: salvar no drift_metrics qual modelo estava servindo
    model_uri_current = os.getenv("MODEL_URI_CURRENT") or os.getenv("CHURN_MODEL_URI")

    cfg = DriftConfig(
        dsn=dsn,
        baseline_path=baseline_path,
        window_days=window_days,
        n_bins=n_bins,
        psi_alert_threshold=psi_thr,
        min_rows=min_rows,
    )

    baseline = load_baseline(cfg.baseline_path)
    baseline_model_uri = None
    if isinstance(baseline.get("meta"), dict):
        baseline_model_uri = baseline["meta"].get("model_uri")

    df_prod = load_production_features_sqlalchemy(cfg.dsn, cfg.window_days)

    if df_prod.empty:
        print(f"OK: sem dados de produção na janela de {cfg.window_days} dias.")
        return 0

    if len(df_prod) < cfg.min_rows:
        print(f"OK: poucos dados na janela ({len(df_prod)} < {cfg.min_rows}). Drift não calculado.")
        return 0

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "y_pred_proba"]

    results: dict[str, float] = {}
    for col in numeric_cols:
        if col not in df_prod.columns or col not in baseline:
            continue

        actual = pd.to_numeric(df_prod[col], errors="coerce").to_numpy(dtype=float)
        expected = np.asarray(baseline[col], dtype=float)
        results[col] = psi(expected, actual, n_bins=cfg.n_bins)

    n_rows = int(len(df_prod))
    max_psi = float(np.nanmax(list(results.values()))) if results else 0.0

    # imprime no stdout (mantém seu comportamento)
    payload = {"window_days": cfg.window_days, "n_rows": n_rows, "psi": results}
    print(json.dumps(payload, indent=2))

    # persiste no Postgres
    persist_drift_metrics(
        dsn=cfg.dsn,
        window_days=cfg.window_days,
        n_rows=n_rows,
        psi_by_feature=results,
        max_psi=max_psi,
        baseline_model_uri=baseline_model_uri,
        model_uri_current=model_uri_current,
    )

    # alerta via exit code
    if np.isfinite(max_psi) and max_psi >= cfg.psi_alert_threshold:
        msg = f"ALERT: PSI alto detectado " f"(max_psi={max_psi:.4f} >= {cfg.psi_alert_threshold})."
        print(msg, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
