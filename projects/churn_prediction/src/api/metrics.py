from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
    ["path"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions returned",
    ["path", "predicted_class"],
)

DB_INSERTS_TOTAL = Counter(
    "predictions_db_inserts_total",
    "Total prediction DB inserts",
    ["status"],  # success|error
)

DB_INSERT_LATENCY_SECONDS = Histogram(
    "predictions_db_insert_latency_seconds",
    "DB insert latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
