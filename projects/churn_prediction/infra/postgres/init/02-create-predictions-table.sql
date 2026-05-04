\connect churn;

CREATE TABLE IF NOT EXISTS churn_predictions (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id TEXT NOT NULL,
  model_uri TEXT NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,
  y_pred INTEGER NOT NULL,
  y_pred_proba DOUBLE PRECISION NOT NULL,
  features JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_created_at
  ON churn_predictions (created_at);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_request_id
  ON churn_predictions (request_id);

CREATE INDEX IF NOT EXISTS idx_churn_predictions_model_uri
  ON churn_predictions (model_uri);
