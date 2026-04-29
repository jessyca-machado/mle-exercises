# """
# Pipelines - Preparação do dataset para treino e teste com:
# A classe SklearnPipelineRunner é um orquestrador de pipeline, que roda para diferentes modelos, que centraliza o fluxo de:
#     - Montar um pipeline (pré-processamento por tipo de coluna, opcional feature engineering, opcional seleção de features, conversão para float32 e o modelo);
#     - Treinar esse pipeline (fit), com opção de tuning via GridSearchCV;
#     - Utilizar o modelo treinado para predict (previsão binária) e predict_proba (probabilidade da previsão binária);
#     - Avaliar no teste com métricas prontas (evaluate, incluindo AUC quando possível);
#     - Rodar validação cruzada (cross_validate) nos dados de treino;
#     - Persistir e recarregar o melhor estimador (save/load).
# Uso:
#     python src/data/pipelines.py
# """
# from __future__ import annotations

# from typing import Any, Dict, Optional

# import joblib
# import logging
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     make_scorer,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
# from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from src.data.transformers import ToFloat32
# from src.ml.logging_utils import get_logger

# logger = get_logger(__name__)


# class _ModelLoggerAdapter(logging.LoggerAdapter):
#     """Prefixa logs com o nome do modelo."""
#     def process(self, msg, kwargs):
#         model = self.extra.get("model_name", "NA")
#         return f"[model={model}] {msg}", kwargs


# class SklearnPipelineRunner:
#     def __init__(
#         self,
#         model,
#         model_name: str | None = None,
#         categorical_cols=None,
#         numerical_cols=None,
#         boolean_cols=None,
#         binned_cols=None,
#         use_feature_engineering: bool = False,
#         feature_engineering_transformer=None,
#         use_feature_selection: bool = False,
#         k_best: Optional[int] = 15,
#         # tuning
#         use_grid_search: bool = False,
#         grid_param_grid: Optional[dict[str, Any]] = None,
#         grid_n_jobs: int = 1,
#         grid_verbose: int = 0,
#         # cv/scoring
#         cv: int = 5,
#         scoring: str = "f1",
#         pos_label: int = 1,
#     ) -> None:
#         self.model = model
#         self.model_name = model_name or type(model).__name__
#         self._log = _ModelLoggerAdapter(logger, {"model_name": self.model_name})

#         self.categorical_cols = categorical_cols or []
#         self.numerical_cols = numerical_cols or []
#         self.boolean_cols = boolean_cols or []
#         self.binned_cols = binned_cols or []

#         self.use_feature_engineering = use_feature_engineering
#         self.feature_engineering_transformer = feature_engineering_transformer
#         self.use_feature_selection = use_feature_selection
#         self.k_best = k_best

#         self.use_grid_search = use_grid_search
#         self.grid_param_grid = grid_param_grid or {}
#         self.grid_n_jobs = grid_n_jobs
#         self.grid_verbose = grid_verbose

#         self.cv = cv
#         self.scoring = scoring
#         self.pos_label = pos_label

#         self.pipeline = None
#         self.best_model = None

#     def _filter_existing_columns(self, X: pd.DataFrame) -> None:
#         existing = set(X.columns)
#         self.categorical_cols = [c for c in self.categorical_cols if c in existing]
#         self.numerical_cols = [c for c in self.numerical_cols if c in existing]
#         self.boolean_cols = [c for c in self.boolean_cols if c in existing]
#         self.binned_cols = [c for c in self.binned_cols if c in existing]

#     def _get_feature_selector(self) -> SelectKBest:
#         k = self.k_best if self.k_best is not None else "all"
#         return SelectKBest(score_func=f_classif, k=k)

#     def _build_preprocessor(self) -> ColumnTransformer:
#         num_cols = self.numerical_cols + self.binned_cols
#         return ColumnTransformer(
#             transformers=[
#                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
#                 ("num", StandardScaler(), num_cols),
#                 ("bol", "passthrough", self.boolean_cols),
#             ],
#             remainder="drop",
#             verbose_feature_names_out=False,
#         )

#     def _build_pipeline(self) -> Pipeline:
#         steps: list[tuple[str, Any]] = []
#         if self.use_feature_engineering:
#             if self.feature_engineering_transformer is None:
#                 raise ValueError("use_feature_engineering=True, mas feature_engineering_transformer=None.")
#             steps.append(("feature_engineering", self.feature_engineering_transformer))

#         steps.append(("preprocessing", self._build_preprocessor()))

#         if self.use_feature_selection:
#             steps.append(("feature_selection", self._get_feature_selector()))

#         steps.append(("to_float32", ToFloat32()))
#         steps.append(("model", self.model))
#         return Pipeline(steps=steps)

#     def build_estimator(self, X: pd.DataFrame | None = None):
#         if X is not None:
#             self._filter_existing_columns(X)
#         return self._build_pipeline()

#     def _should_force_float_y(self) -> bool:
#         return hasattr(self.model, "module") or hasattr(self.model, "device")

#     def _coerce_y_for_fit(self, y):
#         if not self._should_force_float_y():
#             return y

#         y_arr = np.asarray(y)
#         if y_arr.dtype == bool:
#             return y_arr.astype(np.float32)
#         if np.issubdtype(y_arr.dtype, np.number):
#             return y_arr.astype(np.float32)

#         uniq = pd.unique(y_arr)
#         if len(uniq) != 2:
#             raise ValueError(f"y precisa ser binário; encontrei {len(uniq)} classes: {uniq}")
#         uniq_sorted = sorted(list(uniq))
#         mapping = {uniq_sorted[0]: 0.0, uniq_sorted[1]: 1.0}
#         return np.vectorize(mapping.get)(y_arr).astype(np.float32)

#     def fit(self, X, y) -> "SklearnPipelineRunner":
#         self._filter_existing_columns(X)
#         y = self._coerce_y_for_fit(y)

#         self._log.info(
#             "Fit start | grid=%s | scoring=%s | cv=%s | param_grid_type=%s",
#             self.use_grid_search,
#             self.scoring,
#             type(self.cv).__name__,
#             type(self.grid_param_grid).__name__,
#         )

#         base_pipeline = self.build_estimator(X)

#         if self.use_grid_search:
#             self.pipeline = GridSearchCV(
#                 estimator=base_pipeline,
#                 param_grid=self.grid_param_grid,
#                 cv=self.cv,
#                 scoring=self.scoring,
#                 refit=True,
#                 n_jobs=self.grid_n_jobs,
#                 verbose=self.grid_verbose,
#                 error_score=0.0,
#             )
#         else:
#             self.pipeline = base_pipeline

#         self.pipeline.fit(X, y)

#         if self.use_grid_search:
#             self.best_model = self.pipeline.best_estimator_
#             self._log.info("Grid best_params=%s", self.pipeline.best_params_)
#             self._log.info("Grid best_score=%.6f | scoring=%s", float(self.pipeline.best_score_), self.scoring)
#             self.cv_best_score_ = float(self.pipeline.best_score_)
#             self.best_params_ = dict(self.pipeline.best_params_)
#         else:
#             self.best_model = self.pipeline

#         self._log.info("Fit end")
#         return self

#     def oof_predict_proba_after_fit(self, X, y) -> np.ndarray:
#         est = self._get_fitted_estimator()
#         if hasattr(est, "best_estimator_"):  # GridSearchCV
#             base_est = est.best_estimator_
#             cv_obj = est.cv
#         else:
#             base_est = est
#             cv_obj = self.cv

#         proba_oof = cross_val_predict(
#             base_est,
#             X,
#             np.asarray(y).reshape(-1),
#             cv=cv_obj,
#             method="predict_proba",
#             n_jobs=-1,
#         )
#         return proba_oof[:, 1]

#     def _get_fitted_estimator(self):
#         est = getattr(self, "best_model", None)
#         if est is not None:
#             return est
#         est = getattr(self, "pipeline", None)
#         if est is not None:
#             return est
#         raise RuntimeError("Nenhum estimador treinado encontrado. Rode .fit() antes.")

#     def _get_estimator_for_cv_after_fit(self):
#         est = self._get_fitted_estimator()
#         if hasattr(est, "best_estimator_"):
#             return est.best_estimator_
#         return est

#     def predict(self, X) -> pd.Series:
#         est = self._get_fitted_estimator()
#         self._log.debug("predict")
#         return est.predict(X)

#     def predict_proba(self, X) -> np.ndarray:
#         est = self._get_fitted_estimator()
#         self._log.debug("predict_proba")
#         if not hasattr(est, "predict_proba"):
#             raise AttributeError("Modelo não suporta predict_proba.")
#         proba = est.predict_proba(X)
#         return proba[:, 1]

#     # ---- API pública (para evitar confusão) ----
#     def cross_validate(self, X, y, include_auc: bool = True, estimator_override=None) -> Dict[str, float]:
#         return self._cross_validate(X, y, include_auc=include_auc, estimator_override=estimator_override)

#     # ---- Implementação interna ----
#     def _cross_validate(
#         self,
#         X,
#         y,
#         include_auc: bool = True,
#         estimator_override=None,
#     ) -> Dict[str, float]:
#         self._filter_existing_columns(X)
#         y = self._coerce_y_for_fit(y)

#         self._log.info("CV start | scoring=%s | cv=%s", self.scoring, type(self.cv).__name__)

#         estimator = estimator_override if estimator_override is not None else self._build_pipeline()
#         if estimator_override is None and self.use_grid_search:
#             self._log.info("CV will run with GridSearchCV inside (expensive).")
#             estimator = GridSearchCV(
#                 estimator=estimator,
#                 param_grid=self.grid_param_grid,
#                 cv=self.cv,
#                 scoring=self.scoring,
#                 refit=True,
#                 n_jobs=self.grid_n_jobs,
#                 verbose=0,
#                 error_score=0.0,
#             )

#         scoring = {
#             "accuracy": "accuracy",
#             "precision": make_scorer(precision_score, zero_division=0, pos_label=self.pos_label),
#             "recall": make_scorer(recall_score, zero_division=0, pos_label=self.pos_label),
#             "f1": make_scorer(f1_score, zero_division=0, pos_label=self.pos_label),
#         }
#         if include_auc:
#             scoring["auc_roc"] = "roc_auc"

#         res = cross_validate(
#             estimator=estimator,
#             X=X,
#             y=y,
#             cv=self.cv,
#             scoring=scoring,
#             n_jobs=-1,
#             return_train_score=False,
#         )

#         some_key = next(iter(scoring.keys()))
#         n_folds = int(len(res[f"test_{some_key}"]))

#         out: Dict[str, float] = {"cv_n_folds": n_folds}

#         for m in scoring.keys():
#             vals = np.asarray(res[f"test_{m}"], dtype=float)
#             out[f"cv_mean_{m}"] = float(vals.mean())

#         main = self.scoring
#         inverse = {v: k for k, v in scoring.items()}
#         main_key = inverse.get(main, main)

#         if f"test_{main_key}" in res:
#             main_vals = np.asarray(res[f"test_{main_key}"], dtype=float)
#             std_main = float(main_vals.std(ddof=1)) if len(main_vals) > 1 else 0.0
#             out[f"cv_std_{main_key}"] = std_main
#         else:
#             out[f"cv_std_{main_key}"] = float("nan")

#         self._log.info(
#             "CV end | scoring=%s mean=%.4f std=%.4f folds=%d",
#             main_key,
#             out.get(f"cv_mean_{main_key}", float("nan")),
#             out.get(f"cv_std_{main_key}", float("nan")),
#             n_folds,
#         )
#         return out

#     def evaluate(self, X, y, include_auc: bool = True) -> dict[str, float]:
#         self._log.info("Evaluate start")
#         est = self._get_fitted_estimator()
#         y_pred = self.predict(X)

#         metrics: dict[str, float] = {
#             "accuracy": float(accuracy_score(y, y_pred)),
#             "precision": float(precision_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)),
#             "recall": float(recall_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)),
#             "f1": float(f1_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)),
#         }

#         if include_auc:
#             auc = None
#             if hasattr(est, "predict_proba"):
#                 try:
#                     proba = est.predict_proba(X)
#                     y_score = proba[:, 1] if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2 else proba
#                     auc = float(roc_auc_score(y, y_score))
#                 except Exception:
#                     auc = None

#             if auc is None and hasattr(est, "decision_function"):
#                 try:
#                     y_score = est.decision_function(X)
#                     auc = float(roc_auc_score(y, y_score))
#                 except Exception:
#                     auc = None

#             if auc is not None:
#                 metrics["auc_roc"] = auc

#         self._log.info("Evaluate end | metrics=%s", metrics)
#         return metrics

#     def save(self, path) -> None:
#         self._log.info("Saving model to %s", path)
#         joblib.dump(self.best_model, path)

#     def load(self, path) -> "SklearnPipelineRunner":
#         self._log.info("Loading model from %s", path)
#         self.best_model = joblib.load(path)
#         return self


# def main():
#     df = load_data_churn()
#     df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")
#     X_raw = df_clean[FEATURES_COLS].copy()
#     y = df_clean[TARGET_COL].copy()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_raw,
#         y,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y if y.nunique() == 2 else None,
#     )

#     model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

#     # Exemplo de grid (ajuste conforme o modelo)
#     param_grid = {
#         "model__C": [0.01, 0.1, 1.0, 10.0],
#         "model__penalty": ["l2"],
#         "model__solver": ["lbfgs", "liblinear"],
#     }

#     runner = SklearnPipelineRunner(
#         model=model,
#         categorical_cols=CAT_COLS,
#         numerical_cols=NUM_COLS,
#         boolean_cols=BOL_COLS,
#         binned_cols=BIN_COLS,
#         use_feature_engineering=False,
#         feature_engineering_transformer=None,
#         use_feature_selection=False,
#         # GridSearchCV
#         use_grid_search=True,
#         grid_param_grid=param_grid,
#         grid_n_jobs=-1,
#         cv=5,
#         scoring="f1",
#         pos_label=1,
#     )

#     logger.info("Dataset: %d features | Treino: %d | Teste: %d", X_train.shape[1], len(X_train), len(X_test))
#     runner.fit(X_train, y_train)

#     metrics = runner.evaluate(X_test, y_test, include_auc=True)
#     logger.info("Métricas: %s", metrics)

#     runner.save("churn_model_pipeline.joblib")


# if __name__ == "__main__":
#     main()





from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# modelo campeão (sklearn API)
from xgboost import XGBClassifier

# suas constantes
from src.utils.constants import (
    CAT_COLS,
    NUM_COLS,
    BOL_COLS,
    BIN_COLS,
    RANDOM_STATE,
)

# -----------------------------
# Config de produção
# -----------------------------
@dataclass(frozen=True)
class ChurnModelConfig:
    """
    Config do modelo para produção.

    threshold:
        - usado apenas para decisão binária (predict_label)
        - predict_proba sempre retorna probabilidade
    """
    threshold: float = 0.05  # ajuste conforme seu cost_toolkit (best_thr do sweep)
    random_state: int = RANDOM_STATE


def build_preprocessor_from_df(df: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o preprocessor usando apenas colunas existentes no DataFrame (robusto para APIs).
    Mantém a mesma lógica dos seus treinos.
    """
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in NUM_COLS if c in df.columns]
    bol = [c for c in BOL_COLS if c in df.columns]
    bin_cols = [c for c in BIN_COLS if c in df.columns]

    # escalar numéricas + binárias (se binárias forem 0/1, scaling é opcional, mas mantém consistência)
    scaled_cols = num + bin_cols
    num = [c for c in scaled_cols if c not in cat]
    bol = [c for c in bol if c not in cat and c not in num]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
            ("bol", "passthrough", bol),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_xgboost_best_estimator(config: ChurnModelConfig) -> XGBClassifier:
    """
    Hiperparâmetros do seu melhor run (ajuste conforme o best_params que você logou).
    Você mostrou algo como:
        colsample_bytree=0.8, learning_rate=0.03, max_depth=3, n_estimators=300,
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        colsample_bytree=0.8,
        subsample=0.8             # se você tiver no best_params, use o valor correto
        reg_lambda=5.0,             # idem
        objective="binary:logistic",
        eval_metric="aucpr",        # otimiza PR-AUC durante treino
        n_jobs=-1,
        random_state=config.random_state,
    )


def build_churn_pipeline(example_df: pd.DataFrame, config: Optional[ChurnModelConfig] = None) -> Pipeline:
    """
    Monta pipeline preprocess + modelo.
    example_df é usado apenas para inferir colunas presentes (para o preprocessor).
    """
    config = config or ChurnModelConfig()
    preprocessor = build_preprocessor_from_df(example_df)
    model = build_xgboost_best_estimator(config)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# -----------------------------
# Helpers de inferência (API)
# -----------------------------
def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Retorna probabilidade de churn (classe positiva=1).
    """
    proba = pipe.predict_proba(X)[:, 1]
    return proba.astype(float)


def predict_label(pipe: Pipeline, X: pd.DataFrame, threshold: float) -> np.ndarray:
    """
    Retorna rótulo binário usando um threshold (decisão operacional).
    """
    proba = predict_proba(pipe, X)
    return (proba >= threshold).astype(int)


def validate_payload_columns(X: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Payload inválido: faltando colunas: {missing}")
