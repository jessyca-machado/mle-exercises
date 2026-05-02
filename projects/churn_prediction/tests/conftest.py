import numpy as np
import pandas as pd
import pytest

from src.utils.constants import YES_NO_COLS, TARGET_COL, FEATURES_COLS
from src.data.preprocess import pre_processing

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """
    Cria um DataFrame sintético de churn, baseado no dataset original, aplicando um pouco de ruído.
    """
    base = pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D", "E", "F"],
            "gender": ["Female", "Male", "Male", "Female", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
            "tenure": [1, 5, 10, 2, 20, 12],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "No", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No", "Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "No", "Yes", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No", "Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "No", "Yes", "No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 80.0],
            "TotalCharges": ["29.85", "1889.50", "108.15", "1840.75", "151.65", " "],
            "Churn": [0, 0, 1, 0, 1, 0],
        }
    )

    n_repeat = 30
    df = pd.concat([base.assign(_rep=i) for i in range(n_repeat)], ignore_index=True)

    df["customerID"] = df["customerID"] + "_" + df["_rep"].astype(str)

    rng = np.random.default_rng(42)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float) + rng.normal(0, 1.0, size=len(df))
    df["tenure"] = (df["tenure"].astype(int) + rng.integers(0, 3, size=len(df))).clip(lower=0)

    df = df.drop(columns=["_rep"])
    return df

@pytest.fixture
def df_clean(raw_df):
    """
    Aplica o pré-processamento do projeto no DataFrame sintético, para ser usado nos testes.
    """
    return pre_processing(raw_df, YES_NO_COLS, "test", verbose=False)

@pytest.fixture
def X_y(df_clean):
    """
    Seleciona as colunas de features e target do DataFrame pré-processado, para ser usado nos testes.
    """
    X = df_clean[FEATURES_COLS].copy()
    y = df_clean[TARGET_COL].astype(int)
    return X, y

@pytest.fixture
def X_example(df_clean):
    """
    Cria uma amostra do DataFrame de features para ser usada nos testes.
    """
    return df_clean[FEATURES_COLS].head(12).copy()
