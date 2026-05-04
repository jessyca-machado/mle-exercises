from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import psycopg


@dataclass(frozen=True)
class PredictionRecord:
    request_id: str
    batch_id: str | None
    item_index: int | None
    model_uri: str
    threshold: float
    y_pred: int
    y_pred_proba: float
    features: Mapping[str, Any]


def insert(self, rec: PredictionRecord) -> None:
    self.insert_many([rec])


class PredictionsRepository:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def insert(self, rec: PredictionRecord) -> None:
        self.insert_many([rec])

    def insert_many(self, records: Iterable[PredictionRecord]) -> None:
        records = list(records)
        if not records:
            return

        sql = """
        INSERT INTO churn_predictions
            (request_id, batch_id, item_index, model_uri, threshold, y_pred, y_pred_proba, features)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        values = [
            (
                r.request_id,
                r.batch_id,
                r.item_index,
                r.model_uri,
                float(r.threshold),
                int(r.y_pred),
                float(r.y_pred_proba),
                json.dumps(r.features),
            )
            for r in records
        ]

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, values)
            conn.commit()
