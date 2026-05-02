set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
export MLFLOW_REGISTRY_URI="${MLFLOW_REGISTRY_URI:-$MLFLOW_TRACKING_URI}"
export CHURN_MODEL_URI="${CHURN_MODEL_URI:-models:/churn_xgb/13}"
export CHURN_THRESHOLD="${CHURN_THRESHOLD:-0.5}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "Starting API..."
echo "  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "  MLFLOW_REGISTRY_URI=$MLFLOW_REGISTRY_URI"
echo "  CHURN_MODEL_URI=$CHURN_MODEL_URI"
echo "  CHURN_THRESHOLD=$CHURN_THRESHOLD"
echo "  HOST=$HOST"
echo "  PORT=$PORT"
echo

# Opcional: matar processo na porta (evita 'Address already in use')
if command -v lsof >/dev/null 2>&1; then
    PIDS="$(lsof -ti tcp:"$PORT" || true)"
    if [[ -n "${PIDS}" ]]; then
    echo "Killing process(es) on port $PORT: $PIDS"
    kill $PIDS || true
    sleep 0.5
    fi
fi

exec uvicorn src.api.app:app --host "$HOST" --port "$PORT" --reload
