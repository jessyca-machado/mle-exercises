from __future__ import annotations

import os
import random
from typing import Any

import pandas as pd
import requests

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import FEATURES_COLS, YES_NO_COLS


def row_to_payload(row: pd.Series, customer_id: str) -> dict[str, Any]:
    d = row.to_dict()

    # garantir tipos que a API espera
    for k in ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[k] = float(d[k])

    d["tenure"] = float(d["tenure"])
    d["MonthlyCharges"] = float(d["MonthlyCharges"])

    # TotalCharges pode vir string/num; mande como string para evitar edge cases
    d["TotalCharges"] = None if pd.isna(d["TotalCharges"]) else str(d["TotalCharges"])

    d["customer_id"] = customer_id
    return d


def main() -> int:
    api_url = os.getenv("API_URL", "http://localhost:8000")
    n_rows = int(os.getenv("N_ROWS", "200"))
    batch_size = int(os.getenv("BATCH_SIZE", "50"))
    threshold = float(os.getenv("THRESHOLD", "0.5"))

    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")

    X = df_clean[FEATURES_COLS].copy()
    idxs = list(range(len(X)))
    random.shuffle(idxs)
    idxs = idxs[:n_rows]

    items = []
    for i, idx in enumerate(idxs):
        row = X.iloc[idx]
        items.append(row_to_payload(row, customer_id=f"TST_{i:05d}"))

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        payload = {"threshold": threshold, "items": chunk}
        r = requests.post(f"{api_url}/predict_batch", json=payload, timeout=30)
        r.raise_for_status()
        print(f"sent {start+len(chunk)}/{len(items)}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
