import os
import mlflow
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from src.core.models.trainer import ChurnModelTrainer
from src.jobs.train import log_xgb_end_to_end_pyfunc
from src.api.app import app

def _setup_temp_mlflow(tmp_path) -> tuple[str, str]:
    """
    Cria um ambiente MLflow temporário usando SQLite e um diretório local para artefatos.

    Args:
        tmp_path: pytest fixture para criar um diretório temporário.

    Returns:
        tracking_uri e registry_uri. Ambos no formato "sqlite:////path/to/mlflow.db".
    """
    db_path = (tmp_path / "mlflow.db").resolve()
    artifacts_dir = (tmp_path / "mlartifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:////{db_path}"
    registry_uri = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    exp_name = "test-exp-api-hermetic"
    artifact_location = artifacts_dir.as_uri()
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    mlflow.set_experiment(exp_name)
    return tracking_uri, registry_uri

def _train_and_register_pyfunc_model(X, y, request, registered_name: str) -> str:
    """
    Treina um pipeline rápido, loga como pyfunc e registra no Model Registry temporário.

    Args:
        X: DataFrame de features.
        y: Series de target.
        request: pytest fixture para acessar o caminho do projeto.
        registered_name: nome para registrar o modelo no MLflow.

    Returns:
        O model_uri no formato models:/<name>/<version>.
    """
    trainer = ChurnModelTrainer(n_folds=2, seed=42)
    trainer.build(X=X, y=y, model=LogisticRegression(max_iter=200))
    summary = trainer.train()
    root = request.config.rootpath
    pyfunc_path = (root / "src" / "ml" / "churn_pyfunc_xgb.py").resolve()
    assert pyfunc_path.exists()
    with mlflow.start_run(run_name="api-hermetic-train") as run:
        input_example = X.head(50).copy()
        # evita enforcement chato do MLflow (ints -> float)
        int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns
        if len(int_cols) > 0:
            input_example[int_cols] = input_example[int_cols].astype("float64")
        model_uri = log_xgb_end_to_end_pyfunc(
            fitted_pipeline=summary.fitted_pipeline,
            name="model",
            pyfunc_code_path=str(pyfunc_path),
            pip_requirements=None,
            input_example=input_example,
        )
        mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
        version = str(mv.version)
    return f"models:/{registered_name}/{version}"

@pytest.fixture(scope="module")
def api_client(tmp_path_factory, X_y, request) -> TestClient:
    """
    Fixture hermética:
        - cria MLflow temporário
        - treina + registra um modelo
        - sobe a API via TestClient apontando para esse modelo
    """
    tmp_path = tmp_path_factory.mktemp("api_hermetic")
    tracking_uri, registry_uri = _setup_temp_mlflow(tmp_path)
    X, y = X_y
    registered_name = "churn_api_hermetic_model"
    model_uri = _train_and_register_pyfunc_model(X, y, request, registered_name=registered_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    os.environ["CHURN_MODEL_URI"] = model_uri
    os.environ["CHURN_THRESHOLD"] = "0.5"
    with TestClient(app) as client:
        yield client

def _valid_payload(**overrides) -> dict:
    """
    Dicionário com payload de exemplo válido para o endpoint de predição, com possibilidade de overrides.

    Args:
        **overrides: campos para sobrescrever o payload base.

    Returns:
        Dicionário com o payload.
    """
    base = {
        "customer_id": "12345",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 5,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.7,
        "TotalCharges": " ",
    }
    base.update(overrides)
    return base

def test_health_ok(api_client) -> None:
    """Teste simples para verificar se o endpoint de health check está funcionando."""
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ready_ok(api_client) -> None:
    """Teste simples para verificar se o endpoint de ready check está funcionando."""
    r = api_client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "model_uri" in body

def test_predict_valid_contract(api_client) -> None:
    """Teste de predição com payload válido, verificando estrutura da resposta."""
    r = api_client.post("/predict", json=_valid_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["y_pred"] in (0, 1)
    assert 0.0 <= body["y_pred_proba"] <= 1.0
    assert body["latency_ms"] > 0
    assert "model_uri" in body
    assert body["threshold"] == 0.5

def test_predict_missing_required_field_returns_422(api_client) -> None:
    """Teste para verificar que campos obrigatórios faltando retornam erro 422."""
    payload = _valid_payload()
    payload.pop("MonthlyCharges")
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 422

def test_predict_invalid_numeric_returns_422(api_client) -> None:
    """Teste para verificar que valores numéricos inválidos retornam erro 422."""
    payload = _valid_payload(tenure=-1)
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 422

def test_predict_blank_totalcharges_does_not_break(api_client) -> None:
    """Teste para verificar que campos em branco não quebram a API."""
    payload = _valid_payload(TotalCharges=" ")
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["y_pred"] in (0, 1)
    assert 0.0 <= body["y_pred_proba"] <= 1.0
