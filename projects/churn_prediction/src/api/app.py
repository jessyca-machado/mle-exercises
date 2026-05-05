"""
API de inferência — Churn (FastAPI + MLflow PyFunc + ChurnModelTrainer)

Recursos:
    - Carregamento do modelo no startup (lifespan)
    - /health e /ready
    - /predict (1 registro)
    - /predict_batch (lista de registros)
    - Validação de entrada com Pydantic
    - Preparado para testes com TestClient

Execução (dev):
    bash scripts/run_api.sh

Execução (manual):
    export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
    export MLFLOW_REGISTRY_URI="sqlite:///mlflow.db"
    export CHURN_MODEL_URI="models:/churn_xgb/15"
    export CHURN_THRESHOLD="0.5"
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Uso:
    Em outro terminal faz a chamada para enviar os dados: curl -s http://localhost:8000/ready
    insira o JSON de entrada e veja a resposta.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.metrics import (
    DB_INSERT_LATENCY_SECONDS,
    DB_INSERTS_TOTAL,
    HTTP_REQUEST_LATENCY_SECONDS,
    HTTP_REQUESTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    PREDICTIONS_TOTAL,
)
from src.infra.db.predictions_repo import PredictionRecord, PredictionsRepository
from src.jobs.predict import (
    PredictConfig,
    load_pyfunc_model,
    predict_proba_pyfunc,
    resolve_model_uri,
)

try:
    from pythonjsonlogger.json import JsonFormatter
except Exception:
    JsonFormatter = None

logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)

REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> str:
    return REQUEST_ID_CTX.get() or "-"


if not logger.handlers:
    handler = logging.StreamHandler()
    if JsonFormatter is not None:
        handler.setFormatter(
            JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s "
                "%(request_id)s %(method)s %(path)s %(status_code)s %(latency_ms)s %(model_uri)s"
            )
        )
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


MODEL_STATE: dict[str, Any] = {}


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        token = REQUEST_ID_CTX.set(request_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 3),
                    "model_uri": MODEL_STATE.get("model_uri", ""),
                },
            )
            raise
        finally:
            latency_s = time.perf_counter() - start
            HTTP_REQUEST_LATENCY_SECONDS.labels(
                method=request.method, path=request.url.path
            ).observe(latency_s)
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, path=request.url.path, status_code=str(status_code)
            ).inc()

            REQUEST_ID_CTX.reset(token)


NUMERIC_COLS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
]


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


REQUIRED_NUM = ["MonthlyCharges", "tenure"]


def validate_required_numeric(X: pd.DataFrame, required_num: list[str] = REQUIRED_NUM) -> None:
    missing = [c for c in required_num if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Campos obrigatórios ausentes: {missing}")
    if X[required_num].isna().any().any():
        raise HTTPException(status_code=422, detail="Campos numéricos inválidos.")


DROP_MODEL_COLS = {"customer_id"}


def normalize_customer_id(customer_id: str) -> str:
    return customer_id.strip().strip('"').strip("'")


def to_model_df(payload: dict) -> pd.DataFrame:
    data = {k: v for k, v in payload.items() if k not in DROP_MODEL_COLS}
    return pd.DataFrame([data])


def to_model_df_batch(payloads: list[dict]) -> pd.DataFrame:
    data = [{k: v for k, v in p.items() if k not in DROP_MODEL_COLS} for p in payloads]
    return pd.DataFrame(data)


def get_default_threshold() -> float:
    return float(os.getenv("CHURN_THRESHOLD", "0.5"))


def get_predict_config() -> PredictConfig:
    """
    Monta PredictConfig a partir de env vars.
    Prioridade:
        - CHURN_MODEL_URI
        - CHURN_MODEL_NAME + CHURN_MODEL_VERSION
    """
    model_uri = os.getenv("CHURN_MODEL_URI")
    model_name = os.getenv("CHURN_MODEL_NAME")
    model_version = os.getenv("CHURN_MODEL_VERSION")

    return PredictConfig(
        registry_uri=os.getenv(
            "MLFLOW_REGISTRY_URI", os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        ),
        model_uri=model_uri,
        model_name=model_name,
        model_version=model_version,
        threshold=get_default_threshold(),
        proba_col="y_pred_proba",
    )


def get_predictions_repo() -> PredictionsRepository | None:
    dsn = os.getenv("PREDICTIONS_DB_DSN")
    if not dsn:
        return None
    return PredictionsRepository(dsn)


def _safe_insert_many(repo, records) -> None:
    start = time.perf_counter()
    try:
        repo.insert_many(records)
        DB_INSERTS_TOTAL.labels(status="success").inc()
    except Exception:
        DB_INSERTS_TOTAL.labels(status="error").inc()
        logger.exception("db_insert_failed", extra={"request_id": get_request_id()})
        raise
    finally:
        DB_INSERT_LATENCY_SECONDS.observe(time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cfg = get_predict_config()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_registry_uri(cfg.registry_uri)

    model_uri = resolve_model_uri(cfg)
    logger.info("loading_model", extra={"model_uri": model_uri})

    model = load_pyfunc_model(model_uri, registry_uri=cfg.registry_uri)

    MODEL_STATE["model"] = model
    MODEL_STATE["model_uri"] = model_uri
    MODEL_STATE["default_threshold"] = cfg.threshold

    logger.info("startup_complete", extra={"model_uri": model_uri})
    yield

    MODEL_STATE.clear()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para previsão de churn (MLflow PyFunc)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(LatencyLoggingMiddleware)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ChurnPredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_id: str
    gender: str
    SeniorCitizen: float = Field(..., ge=0.0, le=1.0)
    Partner: float = Field(..., ge=0.0, le=1.0)
    Dependents: float = Field(..., ge=0.0, le=1.0)
    tenure: float = Field(..., ge=0.0, le=200.0)

    PhoneService: float = Field(..., ge=0.0, le=1.0)
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: float = Field(..., ge=0.0, le=1.0)
    PaymentMethod: str

    MonthlyCharges: float = Field(..., ge=0.0, le=1000.0)
    TotalCharges: float | str | None = None


class ChurnPredictResponse(BaseModel):
    y_pred: int
    y_pred_proba: float
    threshold: float
    latency_ms: float
    model_uri: str


class ChurnBatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: List[ChurnPredictRequest]
    threshold: float | None = None


class ChurnBatchPredictResponse(BaseModel):
    predictions: List[ChurnPredictResponse]
    latency_ms: float
    model_uri: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    if "model" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Modelo não disponível")
    return {"status": "ready", "model_uri": MODEL_STATE.get("model_uri", "")}


@app.post("/predict", response_model=ChurnPredictResponse)
def predict(
    request: ChurnPredictRequest,
    background_tasks: BackgroundTasks,
    threshold: Optional[float] = None,
) -> ChurnPredictResponse:
    if "model" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Modelo não disponível")

    model = MODEL_STATE["model"]
    model_uri: str = MODEL_STATE.get("model_uri", "")
    default_th = float(MODEL_STATE.get("default_threshold", 0.5))
    th = float(threshold) if threshold is not None else default_th

    payload = request.model_dump()
    payload["customer_id"] = normalize_customer_id(payload["customer_id"])

    X = to_model_df(payload)
    X = coerce_numeric(X)
    validate_required_numeric(X)

    start = time.perf_counter()

    inf_start = time.perf_counter()
    y_prob = predict_proba_pyfunc(model, X, proba_col="y_pred_proba")
    INFERENCE_LATENCY_SECONDS.labels(path="/predict").observe(time.perf_counter() - inf_start)
    y_pred = (y_prob >= th).astype(int)
    PREDICTIONS_TOTAL.labels(path="/predict", predicted_class=str(int(y_pred[0]))).inc()
    latency_ms = (time.perf_counter() - start) * 1000

    repo = get_predictions_repo()
    if repo is not None:
        rec = PredictionRecord(
            request_id=get_request_id(),
            batch_id=None,
            item_index=0,
            model_uri=model_uri,
            threshold=th,
            y_pred=int(y_pred[0]),
            y_pred_proba=float(y_prob[0]),
            features=payload,
        )
        background_tasks.add_task(_safe_insert_many, repo, [rec])

    logger.info(
        "inference_completed",
        extra={
            "request_id": get_request_id(),
            "method": "POST",
            "path": "/predict",
            "status_code": 200,
            "latency_ms": round(latency_ms, 3),
            "model_uri": model_uri,
        },
    )

    return ChurnPredictResponse(
        y_pred=int(y_pred[0]),
        y_pred_proba=float(y_prob[0]),
        threshold=th,
        latency_ms=float(latency_ms),
        model_uri=model_uri,
    )


@app.post("/predict_batch", response_model=ChurnBatchPredictResponse)
def predict_batch(
    payload: ChurnBatchPredictRequest,
    background_tasks: BackgroundTasks,
) -> ChurnBatchPredictResponse:
    if "model" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Modelo não disponível")

    model = MODEL_STATE["model"]
    model_uri: str = MODEL_STATE.get("model_uri", "")
    default_th = float(MODEL_STATE.get("default_threshold", 0.5))
    th = float(payload.threshold) if payload.threshold is not None else default_th

    items: list[dict[str, Any]] = []
    for item in payload.items:
        d = item.model_dump()
        d["customer_id"] = normalize_customer_id(d["customer_id"])
        items.append(d)

    X = to_model_df_batch(items)
    X = coerce_numeric(X)
    validate_required_numeric(X)

    start = time.perf_counter()

    inf_start = time.perf_counter()
    y_prob = predict_proba_pyfunc(model, X, proba_col="y_pred_proba")
    INFERENCE_LATENCY_SECONDS.labels(path="/predict_batch").observe(time.perf_counter() - inf_start)
    y_pred = (y_prob >= th).astype(int)

    for yp in y_pred:
        PREDICTIONS_TOTAL.labels(path="/predict_batch", predicted_class=str(int(yp))).inc()

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "inference_completed",
        extra={
            "request_id": get_request_id(),
            "method": "POST",
            "path": "/predict_batch",
            "status_code": 200,
            "latency_ms": round(latency_ms, 3),
            "model_uri": model_uri,
        },
    )

    repo = get_predictions_repo()
    if repo is not None:
        req_id = get_request_id()
        batch_id = str(uuid.uuid4())
        records = [
            PredictionRecord(
                request_id=req_id,
                batch_id=batch_id,
                item_index=i,
                model_uri=model_uri,
                threshold=th,
                y_pred=int(yp),
                y_pred_proba=float(pp),
                features=item,
            )
            for i, (item, yp, pp) in enumerate(zip(items, y_pred, y_prob, strict=False))
        ]
        background_tasks.add_task(_safe_insert_many, repo, records)

    preds = [
        ChurnPredictResponse(
            y_pred=int(yp),
            y_pred_proba=float(pp),
            threshold=th,
            latency_ms=0.0,
            model_uri=model_uri,
        )
        for yp, pp in zip(y_pred, y_prob, strict=False)
    ]

    return ChurnBatchPredictResponse(
        predictions=preds,
        latency_ms=float(latency_ms),
        model_uri=model_uri,
    )
