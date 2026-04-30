import logging

import numpy as np
import pandas as pd

from src.utils.constants import URL, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data_churn() -> pd.DataFrame:
    """Carrega e pré-processa o dataset Telco Customer Churn da IBM.

    Returns:
        DataFrame com os dados do dataset
    """

    url = URL
    random_state = RANDOM_SEED

    try:
        df = pd.read_csv(url)
        logger.info("Dataset carregado: %d linhas", len(df))
    except Exception:
        logger.warning("Falha ao baixar dados. Usando dados sintéticos.")
        rng = np.random.default_rng(random_state)
        df = pd.DataFrame({
            "customerID": [f"CUST-{i:05d}" for i in range(1, 201)],
            "SeniorCitizen": rng.integers(0, 2, 200),
            "gender": rng.choice([0, 1], 200),
            "Partner": rng.choice([0, 1], 200),
            "Dependents": rng.choice([0, 1], 200),
            "PhoneService": rng.choice([0, 1], 200),
            "MultipleLines": rng.choice([0, 1], 200),
            "InternetService": rng.choice([0, 1, 2], 200),
            "OnlineSecurity": rng.choice([0, 1], 200),
            "OnlineBackup": rng.choice([0, 1], 200),
            "DeviceProtection": rng.choice([0, 1], 200),
            "TechSupport": rng.choice([0, 1], 200),
            "StreamingTV": rng.choice([0, 1], 200),
            "StreamingMovies": rng.choice([0, 1], 200),
            "Contract": rng.choice([0, 1, 2], 200),
            "PaperlessBilling": rng.choice([0, 1], 200),
            "PaymentMethod": rng.choice([0, 1, 2, 3], 200),
            "tenure": rng.integers(0, 72, 200),
            "MonthlyCharges": rng.normal(70, 30, 200).clip(15, 120),
            "TotalCharges": rng.normal(2000, 1500, 200).clip(20, 9000),
            "Churn": rng.integers(0, 2, 200),
        })

    return df

if __name__ == "__main__":
    load_data_churn()
