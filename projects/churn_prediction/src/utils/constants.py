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

MLFLOW_EXPERIMENT_NAME: str = "churn_baselines"

MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"
