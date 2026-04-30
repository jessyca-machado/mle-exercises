from pathlib import Path
from typing import Dict, Union, List, Any
from scipy.stats import randint, uniform, loguniform

URL: str  = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

TEST_SIZE: int = 0.2

TARGET_COL: str = "Churn"

RANDOM_SEED: int = 42

N_FOLDS: int = 10

ALPHA: float = 0.05

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
        "PaperlessBilling",
        "Churn",
]

CAT_COLS: list[str] = [
        "gender",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
]

NUM_COLS: list[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "ltv",
        "TotalChargesPerMonth",
        "MonthlyCharges_squared",
]

BOL_COLS: list[str] = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "SeniorCitizen",
        "onePlusYearCustomer",
        "MultipleLines_flag",
        "InternetService_Flag",
        "OnlineSecurity_Flag",
        "OnlineBackup_Flag",
        "DeviceProtection_Flag",
        "TechSupport_Flag",
        "StreamingTV_Flag",
        "StreamingMovies_Flag",
]

BIN_COLS: list[str] = [
        "MonthlyCharges_group",
        "TotalCharges_group",
]

TRUSTED_TYPES = [
        "src.data.feature_engineering.TelcoFeatureEngineeringBins",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.feature_selection._univariate_selection.SelectKBest",
        "sklearn.feature_selection._univariate_selection.f_classif",
]

MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT_NAME: str = "churn-model-comparison"

MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

PRIMARY_METRIC = "recall"

MLP_GRID = [
        {
                "hidden_layers": [64, 32],
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "batch_size": 256,
                "max_epochs": 100,
                "patience": 10,
                "k_best": 15,
        },
        {
                "hidden_layers": [64, 32],
                "dropout": 0.3,
                "learning_rate": 1e-3,
                "batch_size": 256,
                "max_epochs": 100,
                "patience": 10,
                "k_best": 'all',
        },
        {
                "hidden_layers": [128, 64],
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 128,
                "max_epochs": 100,
                "patience": 10,
                "k_best": 15,
        },
        {
                "hidden_layers": [128, 64],
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 128,
                "max_epochs": 100,
                "patience": 10,
                "k_best": 'all',
        },
        {
                "hidden_layers": [128, 64, 32],
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 128,
                "max_epochs": 150,
                "patience": 15,
                "k_best": 15,
        },
        {
                "hidden_layers": [128, 64, 32],
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 128,
                "max_epochs": 150,
                "patience": 15,
                "k_best": 'all',
        },
]

GridSpec = Union[Dict[str, Any], List[Dict[str, Any]]]

PARAM_DISTS: Dict[str, GridSpec] = {
        "GradientBoosting": {
                "model__n_estimators": randint(100, 801),
                "model__learning_rate": loguniform(1e-3, 0.2),
                "model__max_depth": randint(2, 5),
                "model__subsample": uniform(0.7, 0.3),
                "model__min_samples_split": randint(2, 31),
                "model__min_samples_leaf": randint(1, 11),
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },

        "RandomForest": {
                "model__n_estimators": randint(200, 801),
                "model__max_depth": [None, 6, 8, 12, 16],
                "model__min_samples_split": randint(5, 51),
                "model__min_samples_leaf": randint(2, 21),
                "model__max_features": ["sqrt", "log2"],
                "model__bootstrap": [True],
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
        "SVC_rbf": {
                "model__C": loguniform(1e-2, 1e2),
                "model__gamma": loguniform(1e-4, 1e-1),
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },

        "DecisionTree": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 2, 3, 4, 5, 8, 12, 16, 24],
                "model__min_samples_split": randint(2, 51),
                "model__min_samples_leaf": randint(1, 21),
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
        "logreg": {
                "model__C": loguniform(1e-3, 1e2),
                "model__l1_ratio": uniform(0.0, 1.0),
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
        "dummy_most_frequent": {},
        "dummy_stratified": {},
        "xgboost": {
                "model__n_estimators": randint(200, 1501),
                "model__learning_rate": loguniform(1e-3, 0.2),
                "model__max_depth": randint(2, 9),
                "model__min_child_weight": loguniform(1e-1, 20.0),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_lambda": loguniform(1e-2, 50.0),
                "model__reg_alpha": loguniform(1e-3, 10.0),
                "model__gamma": loguniform(1e-4, 5.0),
                "model__scale_pos_weight": [1.0, 2.0, 3.0, 4.0, 6.0],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
}

N_ITER_BY_MODEL = {
        "DecisionTree": 80,
        "RandomForest": 120,
        "GradientBoosting": 120,
        "SVC_rbf": 60,
        "logreg": 80,
        "xgboost": 200,
        "dummy_most_frequent": 1,
        "dummy_stratified": 1,
}
