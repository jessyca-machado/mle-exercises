import os
from pathlib import Path
from typing import Dict, Union, List, Any
import numpy as np
from scipy.stats import randint, uniform, loguniform

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
        # "ltv"
        # "MonthlyCharges_group",
        # "TotalCharges_group",
        # "onePlusYearCustomer",
        # "MonthlyCharges_squared",
        # "MultipleLines_flag",
        # "InternetService_flag",
        # "OnlineSecurity_flag",
        # "OnlineBackup_flag",
        # "DeviceProtection_flag",
        # "TechSupport_flag",
        # "StreamingTV_flag",
        # "StreamingMovies_flag"
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

TRUSTED = [
        "src.data.feature_engineering.TelcoFeatureEngineeringBins",
        "experiments.deep_learning.torch_mlp.TorchMLPClassifier",
        "sklearn.feature_selection._univariate_selection.f_classif",
        "sklearn._loss.link.Interval",
        "sklearn._loss.link.LogitLink",
        "sklearn._loss.loss.HalfBinomialLoss",
]

TRUSTED_SKORCH_TORCH = TRUSTED + [
        "experiments.deep_learning.torch_mlp.TorchMLP",
        "skorch.classifier.NeuralNetBinaryClassifier",
        "skorch.history.History",
        "skorch.dataset.Dataset",
        "skorch.callbacks.logging.EpochTimer",
        "skorch.callbacks.logging.PrintLog",
        "skorch.callbacks.scoring.EpochScoring",
        "skorch.callbacks.scoring.PassthroughScoring",
        "torch.nn.modules.loss.BCEWithLogitsLoss",
        "torch.optim.adam.Adam",
        "torch.utils.data.dataloader.DataLoader",
        "src.data.transformers.ToFloat32",
        "functools.partial",
        "skorch.setter.optimizer_setter",
        "skorch.utils._indexing_other",
        "skorch.utils.to_numpy",
        "builtins.print",
]

MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT_NAME: str = "churn-model-comparison"

MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"

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

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Reproducibilidade
RANDOM_SEED = 42
N_FOLDS = 5

# Dataset
DATASET_KAGGLE = "uciml/default-of-credit-card-clients-dataset"
DATASET_FILENAME = "credit_default.csv"

# Metricas para avaliacao
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
PRIMARY_METRIC = "average_precision"

GridSpec = Union[Dict[str, Any], List[Dict[str, Any]]]

PARAM_DISTS: Dict[str, GridSpec] = {
        "GradientBoosting": {
                "model__n_estimators": randint(100, 801),          # 100..800
                "model__learning_rate": loguniform(1e-3, 0.2),     # 0.001..0.2
                "model__max_depth": randint(2, 5),                 # 2..4
                "model__subsample": uniform(0.7, 0.3),             # 0.7..1.0
                "model__min_samples_split": randint(2, 31),        # 2..30
                "model__min_samples_leaf": randint(1, 11),         # 1..10
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },

        "RandomForest": {
                "model__n_estimators": randint(200, 801),         # 200..1000
                "model__max_depth": [None, 6, 8, 12, 16],
                "model__min_samples_split": randint(5, 51),        # 2..50
                "model__min_samples_leaf": randint(2, 21),         # 1..20
                "model__max_features": ["sqrt", "log2"],
                "model__bootstrap": [True],
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
        "SVC_rbf": {
                # em geral, SVC é sensível: loguniform ajuda muito mais que lista fixa
                "model__C": loguniform(1e-2, 1e2),                 # 0.01..100
                "model__gamma": loguniform(1e-4, 1e-1),            # 1e-4..1e-1
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

        # LogisticRegression tem “espaço discreto”; aqui fica melhor como list[dict]
        "logreg": {
                "model__C": loguniform(1e-3, 1e2),
                "model__l1_ratio": uniform(0.0, 1.0),
                "model__class_weight": [None, "balanced"],
                "select_kbest__k": [15, 20, 25, 30, "all"],
        },
        "dummy_most_frequent": {},   # não faz sentido random search
        "dummy_stratified": {},
        "xgboost": {
                "model__n_estimators": randint(200, 1501),         # 200..1500
                "model__learning_rate": loguniform(1e-3, 0.2),     # 0.001..0.2
                "model__max_depth": randint(2, 9),                 # 2..8
                "model__min_child_weight": loguniform(1e-1, 20.0), # 0.1..20
                "model__subsample": uniform(0.6, 0.4),             # 0.6..1.0
                "model__colsample_bytree": uniform(0.6, 0.4),      # 0.6..1.0
                "model__reg_lambda": loguniform(1e-2, 50.0),       # 0.01..50
                "model__reg_alpha": loguniform(1e-3, 10.0),        # 0.001..10
                "model__gamma": loguniform(1e-4, 5.0),             # 1e-4..5
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
