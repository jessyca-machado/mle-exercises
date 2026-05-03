"""Classe de treino — MOdelo a ser utilizado na pipeline de treino e teste da previsão de churn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.feature_engineering import TelcoFeatureEngineeringBins
from src.ml.data_utils import build_preprocessor
from src.utils.constants import N_FOLDS, RANDOM_SEED


@dataclass
class CVSummary:
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    params: Dict[str, Any]
    fitted_pipeline: Pipeline


class ChurnModelTrainer:
    """
    Trainer para a previsão de churn, com:
        - Feature engineering: TelcoFeatureEngineeringBins
        - Preprocess: ColumnTransformer baseado em listas (CAT/NUM/BOL/BIN)
        - (Opcional) seleção: SelectKBest(mutual_info_classif)
        - CV: StratifiedKFold
    Sem dependência de infraestrutura.
    """

    def __init__(
        self,
        n_folds: int = 5,
        seed: int = RANDOM_SEED,
        k_best: Union[int, str] = "all",
        fe_kwargs: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[ColumnTransformer] = None,
    ) -> None:
        self.n_folds = n_folds
        self.seed = seed
        self.k_best = k_best
        self.fe_kwargs = fe_kwargs
        self.preprocessor_base = preprocessor or build_preprocessor()
        self.metrics_mean: Dict[str, float] = {}
        self.metrics_std: Dict[str, float] = {}
        self.params: Dict[str, Any] = {}
        self.pipeline: Optional[Pipeline] = None
        self.final_model: Optional[Pipeline] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.cv_results: Optional[Dict[str, Any]] = None

    def build(self, X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> None:
        """
        Constroi o pipeline de machine learning end-to-end vinculando o pré-processador ao
            +algoritmo.

        Args:
            X (pd.DataFrame): Conjunto de features de treinamento.
            y (pd.Series): Variável alvo.
            model (Any): Algoritmo de classificação (ex: XGBoostClassifier).
        """
        self.X = X
        self.y = y.astype(int)

        fe_kwargs = self.fe_kwargs or dict(monthlycharges_q=5, totalcharges_q=10)

        self.pipeline = Pipeline(
            steps=[
                ("feature_engineering", TelcoFeatureEngineeringBins(**fe_kwargs)),
                ("preprocess", clone(self.preprocessor_base)),
                ("drop_constant", VarianceThreshold(threshold=0.0)),
                ("select_kbest", SelectKBest(score_func=mutual_info_classif, k=self.k_best)),
                ("model", model),
            ]
        )

    def train(
        self,
        n_splits: int = N_FOLDS,
        random_state: int = RANDOM_SEED,
    ) -> CVSummary:
        """
        Executa o treinamento do modelo utilizando validação cruzada estratificada.

        Args:
            n_splits (int): Número de folds para a validação cruzada.
            random_state (int): Estado aleatório para replicabilidade.

        Returns:
            CVSummary: Um dataclass contendo as métricas médias, desvios padrão, parâmetros do
            modelo e o pipeline final ajustado.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline não construído. Chame build(X, y, model) antes de train()."
            )

        self.n_splits = n_splits
        self.seed = random_state
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }

        self.cv_results = cross_validate(
            self.pipeline,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            return_estimator=False,
        )

        keys = list(scoring.keys())
        self.metrics_mean = {k: float(np.mean(self.cv_results[f"test_{k}"])) for k in keys}
        self.metrics_std = {
            k: float(np.std(self.cv_results[f"test_{k}"], ddof=1)) if self.n_folds > 1 else 0.0
            for k in keys
        }

        self.params = self.pipeline.get_params(deep=True)

        self.final_model = clone(self.pipeline).fit(self.X, self.y)

        return CVSummary(
            metrics_mean=self.metrics_mean,
            metrics_std=self.metrics_std,
            params=self.params,
            fitted_pipeline=self.final_model,
        )

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
        proba_col: str = "y_pred_proba",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz classe (0/1) e probabilidade (classe positiva) a partir do `final_model`.

        Compatível com:
        1) Pipeline/estimadores scikit-learn (com `predict_proba` ou `decision_function`)
        2) MLflow PyFunc (`mlflow.pyfunc.PyFuncModel`), que expõe `.predict(X)`

        Para uso em API/FastAPI, o contrato recomendado do PyFunc é retornar sempre
        um DataFrame com a coluna `y_pred_proba`, ou então um array 1D de probabilidades.

        Args:
            X: Features de entrada.
            threshold: Threshold para converter probabilidade em classe quando necessário.
            proba_col: Nome da coluna esperada caso o PyFunc retorne um DataFrame.

        Returns:
            (y_pred, y_proba)
                - y_pred: np.ndarray int (0/1)
                - y_proba: np.ndarray float, probabilidade da classe positiva

        Raises:
            RuntimeError: se `final_model` não estiver definido.
            TypeError: se não for possível obter probabilidade do modelo.
            ValueError: se as dimensões retornadas pelo modelo forem inválidas.
        """
        if self.final_model is None:
            raise RuntimeError("Modelo final não definido. Injete `final_model` ou chame train().")

        if hasattr(self.final_model, "steps"):
            y_pred = self.final_model.predict(X)

            model = self.final_model.steps[-1][1]

            if hasattr(self.final_model, "predict_proba"):
                y_proba = self.final_model.predict_proba(X)[:, 1]
            elif hasattr(self.final_model, "decision_function"):
                scores = self.final_model.decision_function(X)
                y_proba = 1.0 / (1.0 + np.exp(-scores))
            elif hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            else:
                raise TypeError("Modelo sklearn não suporta predict_proba nem decision_function.")

            return np.asarray(y_pred).astype(int), np.asarray(y_proba).astype(float)

        if not hasattr(self.final_model, "predict"):
            raise TypeError("final_model não possui método predict(X).")

        raw = self.final_model.predict(X)

        if isinstance(raw, pd.DataFrame):
            if proba_col in raw.columns:
                y_proba = raw[proba_col].to_numpy(dtype=float)
            elif raw.shape[1] == 1:
                y_proba = raw.iloc[:, 0].to_numpy(dtype=float)
            else:
                raise TypeError(
                    f"PyFunc retornou DataFrame com colunas {list(raw.columns)}. "
                    f"Esperado coluna '{proba_col}' ou DataFrame de 1 coluna."
                )
        elif isinstance(raw, pd.Series):
            y_proba = raw.to_numpy(dtype=float)
        else:
            y_proba = np.asarray(raw, dtype=float)

        y_proba = np.asarray(y_proba).reshape(-1)

        if y_proba.ndim != 1:
            raise ValueError(f"Saída do modelo não é 1D. shape={getattr(y_proba, 'shape', None)}")

        y_pred = (y_proba >= float(threshold)).astype(int)

        return y_pred, y_proba.astype(float)
