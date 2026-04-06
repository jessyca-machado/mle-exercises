URL: str  = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )

RANDOM_STATE: int = 42

TEST_SIZE: int = 0.2

TARGET_COL: str = "Churn"

FEATURES_COLS: list[str] = [
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

YES_NO_COLS: list[str] = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]

MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT_NAME: str = "churn_baselines"

MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"
