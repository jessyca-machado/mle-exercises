"""Dataset — load do dataset que será utilizado em todo o projeto.

Carrega o dataset da web para ser utilizado posteriormente.

Uso:
    python telco_customer_churn.py
"""

import logging

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE=42


def load_telco_customer_churn_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carrega e pré-processa o dataset Telco Customer Churn da IBM.

    Aplica imputação simples e encoding de variáveis categóricas.

    Returns:
        Tupla (df, X, y) com features e target.
    """
    url = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )

    try:
        df = pd.read_csv(url)
        logger.info("Dataset carregado: %d linhas", len(df))
    except Exception:
        logger.warning("Falha ao baixar dados. Usando dados sintéticos.")
        rng = np.random.default_rng(RANDOM_STATE)
        df = pd.DataFrame({
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

    feature_cols = [
        "SeniorCitizen",
        "gender",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    df = df.dropna(subset=["customerID"])

    yes_no_cols = [
            "Partner","Dependents","PhoneService","MultipleLines",
            "OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies",
            "PaperlessBilling","Churn"
        ]

    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df["InternetService"] = df["InternetService"].map({
        "No": 0,
        "DSL": 1,
        "Fiber optic": 2
    })

    df["Contract"] = df["Contract"].map({
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    })

    df["PaymentMethod"] = df["PaymentMethod"].map({
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    })

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df[feature_cols]
    y = df["Churn"]
    
    return df, X, y
