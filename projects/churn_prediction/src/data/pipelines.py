# """Pipelines - Preparação do dataset para churn (treino/teste) com:

# - load -> preprocess -> feature engineering -> split -> one-hot (fit no treino, transform no teste)

# Uso:
#     python src/data/pipelines.py
# """
# from __future__ import annotations

# import logging

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression

# from optuna.integration import OptunaSearchCV

# import joblib

# from sklearn.metrics import (
#     roc_auc_score,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score
# )

# from src.data.load_data import load_data_churn
# from src.data.preprocess import pre_processing
# from src.utils.constants import FEATURES_COLS, YES_NO_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE, CAT_COLS, NUM_COLS, BOL_COLS, BIN_COLS

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# logger = logging.getLogger(__name__)

# class SklearnPipelineRunner:
#     """
#     Pipeline Runner - Gerencia o processo de treino, avaliação e predição com pipelines do scikit-learn.
#     """
#     def __init__(
#         self,
#         model,
#         categorical_cols=None,
#         numerical_cols=None,
#         boolean_cols=None,
#         binned_cols=None,
#         use_feature_engineering=False,
#         feature_engineering_transformer=None,
#         use_feature_selection=False,
#         k_best=15,
#         use_grid_search=False,
#         param_grid=None,
#         cv=5,
#         scoring='accuracy',
#         pos_label=1,
#         use_optuna_search=False,
#         optuna_n_trials=30,
#         optuna_timeout=None,
#         optuna_param_distributions=None,
#     ) -> None:
#         """
#         Inicializa o pipeline runner.
        
#         Args:
#             model: Estimador do scikit-learn (ex.: LogisticRegression()).
#             categorical_cols: Lista de colunas categóricas para one-hot encoding.
#             numerical_cols: Lista de colunas numéricas para padronização.
#             use_feature_engineering: Se True, aplica transformação de features (bins aprendidos no treino).
#             feature_engineering_transformer: Instância de um transformer para feature engineering (ex.: TelcoFeatureEngineeringBins()).
#             use_feature_selection: Se True, aplica seleção de features (SelectKBest).
#             k_best: Número de features a selecionar (se use_feature_selection=True).
#             use_grid_search: Se True, realiza busca em grade para otimização de hiperparâmetros.
#             param_grid: Dicionário de parâmetros para grid search (ex.: {"model__C": [0.1, 1, 10]}).
#             cv: Número de folds para validação cruzada (usado em grid search e cross_validate).
#             scoring: Função de avaliação (ex.: 'roc_auc').
#             pos_label: Rótulo da classe positiva (default: 1).
#         """
#         self.model = model
#         self.categorical_cols = categorical_cols or []
#         self.numerical_cols = numerical_cols or []
#         self.boolean_cols = boolean_cols or []
#         self.binned_cols = binned_cols or []

#         self.use_feature_engineering = use_feature_engineering
#         self.feature_engineering_transformer = feature_engineering_transformer

#         self.use_feature_selection = use_feature_selection
#         self.k_best = k_best

#         self.use_grid_search = use_grid_search
#         self.param_grid = param_grid or {}
#         self.cv = cv
#         self.scoring = scoring
#         self.pos_label = pos_label

#         self.pipeline = None
#         self.best_model = None


#     def _filter_existing_columns(self, X: pd.DataFrame) -> None:
#         """Remove colunas inexistentes das listas do runner para evitar erro no ColumnTransformer."""
#         existing = set(X.columns)
#         self.categorical_cols = [c for c in self.categorical_cols if c in existing]
#         self.numerical_cols   = [c for c in self.numerical_cols   if c in existing]
#         self.boolean_cols     = [c for c in self.boolean_cols     if c in existing]
#         self.binned_cols      = [c for c in self.binned_cols      if c in existing]


#     def _get_feature_selector(self) -> SelectKBest:
#         """
#         Retorna o seletor de features (SelectKBest) configurado com o número de features a selecionar.
#         Se k_best for None, seleciona todas as features.
#         """
#         k = self.k_best if self.k_best is not None else "all"
#         feature_selector = SelectKBest(score_func=f_classif, k=k)

#         return feature_selector


#     def _build_pipeline(self) -> Pipeline:
#         """
#         Constrói o pipeline do scikit-learn com as etapas configuradas (feature engineering, pré-processamento, seleção de features, modelo).
#             - A etapa de feature engineering é opcional e pode ser um transformer customizado (ex.: TelcoFeatureEngineeringBins) que aprende os bins no fit e aplica no transform.
#             - O pré-processamento inclui one-hot encoding para colunas categóricas e padronização para colunas numéricas.
#             - A seleção de features é opcional e usa SelectKBest com f_classif.
#             - O modelo é o último step do pipeline.
#         """
#         steps = []

#         if self.use_feature_engineering:
#             if self.feature_engineering_transformer is None:
#                 raise ValueError(
#                     "use_feature_engineering=True, mas feature_engineering_transformer=None. "
#                     "Passe uma instância, ex.: TelcoFeatureEngineeringBins(...)."
#                 )
#             steps.append(("feature_engineering", self.feature_engineering_transformer))

#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
#                 ("num", StandardScaler(), self.numerical_cols),
#                 ("bol", "passthrough", self.boolean_cols),
#                 ("bin", "passthrough", self.binned_cols)
#             ],
#             remainder="drop",
#             verbose_feature_names_out=False
#         )

#         preprocessor.set_output(transform="pandas")

#         steps.append(("preprocessing", preprocessor))

#         if self.use_feature_selection:
#             fs = self._get_feature_selector()
#             try:
#                 fs.set_output(transform="pandas")
#             except Exception:
#                 pass
#             steps.append(("feature_selection", fs))

#         steps.append(("model", self.model))

#         return Pipeline(steps=steps)
    

#     def fit(self, X, y) -> SklearnPipelineRunner:
#         """
#         Ajusta o pipeline aos dados de treino (X, y). Se use_grid_search=True, realiza busca em grade para otimização de hiperparâmetros.
#             - O pipeline é construído com as etapas configuradas (feature engineering, pré-processamento, seleção de features, modelo).
#             - Se use_grid_search=True, o pipeline é envolvido por GridSearchCV para encontrar os melhores hiperparâmetros com base no param_grid, cv e scoring configurados.
#             - O modelo final ajustado (com ou sem grid search) é armazenado em self.best_model para uso posterior em predição e avaliação.
        
#         Args:
#             X: DataFrame de features de treino.
#             y: Série de rótulos de treino.

#         Returns:
#             self: Retorna a própria instância do pipeline runner após o ajuste.
#         """
#         self._filter_existing_columns(X)
        
#         base_pipeline = self._build_pipeline()

#         if self.use_optuna_search:
#             self.pipeline = OptunaSearchCV(
#                 base_pipeline,
#                 param_distributions=self.optuna_param_distributions,
#                 cv=self.cv,
#                 scoring=self.scoring,
#                 n_trials=self.optuna_n_trials,
#                 timeout=self.optuna_timeout,
#                 refit=True,
#                 random_state=RANDOM_STATE,
#                 n_jobs=1,
#             )
#         elif self.use_grid_search:
#             self.pipeline = GridSearchCV(
#                 base_pipeline,
#                 param_grid=self.param_grid,
#                 cv=self.cv,
#                 scoring=self.scoring,
#                 n_jobs=-1
#             )
#         else:
#             self.pipeline = base_pipeline

#         self.pipeline.fit(X, y)

#         if self.use_grid_search:
#             self.best_model = self.pipeline.best_estimator_

#             logger.info("Melhores parâmetros (GridSearchCV): %s", self.pipeline.best_params_)
#             logger.info(
#                 "cv_best_score=%.6f | scoring=%s",
#                 float(self.pipeline.best_score_),
#                 getattr(self, "scoring", None),
#             )

#             self.cv_best_score_ = float(self.pipeline.best_score_)
#             self.best_params_ = dict(self.pipeline.best_params_)
#         else:
#             self.best_model = self.pipeline

#         return self


#     def _get_fitted_estimator(self):
#         """
#         Retorna o estimador já treinado que deve ser usado para predict/evaluate.

#         Regra:
#         - Se best_model existe (caso comum: após GridSearchCV => best_estimator_), use ele.
#         - Caso contrário, caia para pipeline (pipeline simples já fitado).
#         """
#         est = getattr(self, "best_model", None)
#         if est is not None:
#             return est

#         est = getattr(self, "pipeline", None)
#         if est is not None:
#             return est

#         raise RuntimeError(
#             "Nenhum estimador treinado encontrado. Rode .fit() antes de .evaluate(). "
#             "(best_model e pipeline estão None)."
#         )
    

#     def predict(self, X) -> pd.Series:
#         """
#         Realiza predição usando o modelo ajustado (self.best_model) no conjunto de features X.
#             - O método predict do pipeline é chamado, o que garante que todas as etapas de pré
#             - Processamento e engenharia de features sejam aplicadas corretamente antes da predição.

#         Args:
#             X: DataFrame de features para predição.

#         Returns:
#             numpy.ndarray: Array com as predições.
#         """
#         est = self._get_fitted_estimator()
#         y_pred = est.predict(X)

#         return y_pred


#     def predict_proba(self, X) -> pd.Series:
#         """
#         Realiza predição de probabilidades usando o modelo ajustado (self.best_model) no conjunto de features X.
#             - O método predict_proba do modelo é chamado, o que garante que todas as etapas de pré-processamento e engenharia de features sejam aplicadas corretamente antes da predição.
#             - Retorna a probabilidade da classe positiva (pos_label) para cada amostra.

#         Args:
#             X: DataFrame de features para predição.

#         Returns:
#             numpy.ndarray: Array com as probabilidades da classe positiva.
#         """
#         est = self._get_fitted_estimator()
#         if not hasattr(est, "predict_proba"):
#             raise AttributeError("Modelo não suporta predict_proba (necessário p/ AUC).")
#         y_pred_proba = est.predict_proba(X)

#         return y_pred_proba[:, 1]


#     def cross_validate(self, X, y) -> np.ndarray:
#         """
#         Realiza validação cruzada usando o pipeline construído.
#             - O pipeline é construído com as etapas configuradas (feature engineering, pré-processamento, seleção de features, modelo).
#             - O método cross_val_score é chamado com o pipeline, X, y, cv e
#             - scoring configurados para obter as métricas de validação cruzada.
#             - Os scores de cada fold, bem como a média e o desvio padrão, são registrados no logger.

#         Args:
#             X: DataFrame de features para validação cruzada.
#             y: Série de rótulos para validação cruzada.

#         Returns:
#             numpy.ndarray: Array com os scores de cada fold.
#         """
#         self._filter_existing_columns(X)
        
#         estimator = self._build_pipeline()

#         if self.use_grid_search:
#             estimator = GridSearchCV(
#                 estimator,
#                 param_grid=self.param_grid,
#                 cv=self.cv,
#                 scoring=self.scoring,
#                 n_jobs=-1
#             )

#         scores = cross_val_score(
#             estimator,
#             X,
#             y,
#             cv=self.cv,
#             scoring=self.scoring,
#             n_jobs=-1
#         )
#         logger.info("CV Scores: %s", scores)
#         logger.info("Mean: %.4f | Std: %.4f", scores.mean(), scores.std())

#         return scores


#     def evaluate(self, X, y, include_auc=True) -> dict[str, float]:
#         """
#         Avalia o desempenho do modelo ajustado no conjunto de dados fornecido.
#             - Realiza predição usando o método predict.
#             - Calcula as métricas de avaliação (accuracy, precision, recall, f1)
#             - Se include_auc=True e o modelo suporta predict_proba, calcula também a métrica ROC AUC.
#             - As métricas calculadas são registradas no logger.

#         Args:
#             X: DataFrame de features para avaliação.
#             y: Série de rótulos verdadeiros para avaliação.
#             include_auc: Se True, inclui a métrica ROC AUC na avaliação (se suportada pelo modelo).

#         Returns:
#             dict: Dicionário com as métricas de avaliação calculadas.
#         """
#         est = self._get_fitted_estimator()
#         y_pred = self.predict(X)

#         pos_label = getattr(self, "pos_label", 1)

#         metrics: dict[str, float] = {
#             "accuracy": accuracy_score(y, y_pred),
#             "precision": precision_score(
#                 y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0
#             ),
#             "recall": recall_score(
#                 y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0
#             ),
#             "f1": f1_score(
#                 y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0
#             ),
#         }

#         if include_auc:
#             auc = None

#             if hasattr(est, "predict_proba"):
#                 try:
#                     proba = est.predict_proba(X)
#                     if proba is not None and getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
#                         y_score = proba[:, 1]
#                     else:
#                         y_score = proba
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

#         return metrics


#     def save(self, path) -> None:
#         """
#         Salva o modelo ajustado em um arquivo.
#             - O modelo salvo inclui todas as etapas do pipeline para garantir que a predição futura seja consistente com o processo de treino.
#             - O método joblib.dump é usado para salvar o modelo em um arquivo especificado pelo caminho fornecido.
#         Args:
#             path: Caminho para o arquivo onde o modelo será salvo.
        
#         Returns:
#             None
#         """
#         joblib.dump(self.best_model, path)


#     def load(self, path) -> SklearnPipelineRunner:
#         """
#         Carrega um modelo salvo de um arquivo.
#             - O método joblib.load é usado para carregar o modelo de um arquivo especificado pelo caminho fornecido.
#             - O modelo carregado é armazenado em self.best_model para uso posterior em predição e avaliação.
#         Args:
#             path: Caminho para o arquivo onde o modelo está salvo.
#         Returns:
#             self: Retorna a própria instância do pipeline runner após carregar o modelo.
#         """
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

#     model = LogisticRegression(random_state=RANDOM_STATE)

#     runner = SklearnPipelineRunner(
#         model=model,
#         categorical_cols=CAT_COLS,
#         numerical_cols=NUM_COLS,
#         boolean_cols=BOL_COLS,
#         binned_cols=BIN_COLS,
#         use_feature_engineering=False,
#         feature_engineering_transformer=None,
#         use_feature_selection=False,
#         use_grid_search=False,
#         param_grid=None,
#         cv=5,
#         scoring="accuracy",
#         pos_label=1
#     )

#     logger.info(
#         "Dataset: %d features | Treino: %d | Teste: %d",
#         X_train.shape[1],
#         len(X_train),
#         len(X_test),
#     )

#     runner.fit(X_train, y_train)

#     runner.evaluate(X_test, y_test, include_auc=True)
#     runner.save("churn_model_pipeline.joblib")


# if __name__ == "__main__":
#     main()


"""
Pipelines - Preparação do dataset para churn (treino/teste) com:
- load -> preprocess -> feature engineering -> split -> one-hot/scale -> (opcional) feature selection -> modelo
Agora com suporte adicional a:
- OptunaSearchCV (além de GridSearchCV)

Uso:
    python src/data/pipelines.py
"""

from __future__ import annotations

from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
# from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.model_selection import cross_val_score, train_test_split  # MODIFICADO: remove GridSearchCV import
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from optuna_integration import OptunaSearchCV
except Exception:
    OptunaSearchCV = None

try:
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
except Exception:
    optuna = None
    CategoricalDistribution = FloatDistribution = IntDistribution = None


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from optuna_integration import OptunaSearchCV
except Exception:  # pragma: no cover
    OptunaSearchCV = None  # type: ignore

from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.utils.constants import (
    BIN_COLS,
    BOL_COLS,
    CAT_COLS,
    FEATURES_COLS,
    NUM_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    YES_NO_COLS,
)

from src.data.transformers import ToFloat32
from src.ml.logging_utils import get_logger
from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)

class SklearnPipelineRunner:
    """
    Pipeline Runner - Gerencia o processo de treino, avaliação e predição com pipelines do scikit-learn.
    Atualização:
    - suporte opcional a OptunaSearchCV.
    """

    def __init__(
        self,
        model,
        categorical_cols=None,
        numerical_cols=None,
        boolean_cols=None,
        binned_cols=None,
        use_feature_engineering: bool = False,
        feature_engineering_transformer=None,
        use_feature_selection: bool = False,
        k_best: Optional[int] = 15,
        # tuning
        use_optuna_search: bool = False,
        optuna_param_distributions: Optional[dict[str, Any]] = None,
        optuna_n_trials: int = 30,
        optuna_timeout: Optional[int] = None,
        optuna_n_jobs: int = 1,
        # cv/scoring
        cv: int = 5,
        scoring: str = "accuracy",
        pos_label: int = 1,
    ) -> None:
        self.model = model
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.boolean_cols = boolean_cols or []
        self.binned_cols = binned_cols or []
        self.use_feature_engineering = use_feature_engineering
        self.feature_engineering_transformer = feature_engineering_transformer
        self.use_feature_selection = use_feature_selection
        self.k_best = k_best

        # MODIFICADO: apenas Optuna
        self.use_optuna_search = use_optuna_search
        self.optuna_param_distributions = optuna_param_distributions or {}
        self.optuna_n_trials = optuna_n_trials
        self.optuna_timeout = optuna_timeout
        self.optuna_n_jobs = optuna_n_jobs

        self.cv = cv
        self.scoring = scoring
        self.pos_label = pos_label
        self.pipeline = None
        self.best_model = None

        # validações básicas
        if self.use_optuna_search and OptunaSearchCV is None:
            raise ImportError(
                "OptunaSearchCV não está disponível. Instale: `pip install optuna optuna-integration`."
            )
        if self.use_optuna_search and optuna is None:
            raise ImportError("Optuna não está disponível. Instale: `pip install optuna`.")


    def _filter_existing_columns(self, X: pd.DataFrame) -> None:
        """Remove colunas inexistentes das listas do runner para evitar erro no ColumnTransformer."""
        existing = set(X.columns)
        self.categorical_cols = [c for c in self.categorical_cols if c in existing]
        self.numerical_cols = [c for c in self.numerical_cols if c in existing]
        self.boolean_cols = [c for c in self.boolean_cols if c in existing]
        self.binned_cols = [c for c in self.binned_cols if c in existing]

    def _get_feature_selector(self) -> SelectKBest:
        """Retorna o seletor de features (SelectKBest) configurado."""
        k = self.k_best if self.k_best is not None else "all"
        return SelectKBest(score_func=f_classif, k=k)

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Constrói e retorna o ColumnTransformer de preprocessing.

        Nota:
        - Para máxima compatibilidade com skorch/PyTorch, é geralmente melhor o output ser numpy.
            Por isso, aqui NÃO forçamos set_output(transform="pandas").
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
                ("num", StandardScaler(), self.numerical_cols),
                ("bol", "passthrough", self.boolean_cols),
                ("bin", "passthrough", self.binned_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return preprocessor

    def _build_pipeline(self) -> Pipeline:
        """
        Constrói o pipeline do scikit-learn com as etapas configuradas:
        feature_engineering (opcional) -> preprocessing -> feature_selection (opcional) -> model
        """
        steps: list[tuple[str, Any]] = []

        if self.use_feature_engineering:
            if self.feature_engineering_transformer is None:
                raise ValueError(
                    "use_feature_engineering=True, mas feature_engineering_transformer=None. "
                    "Passe uma instância, ex.: TelcoFeatureEngineeringBins(...)."
                )
            steps.append(("feature_engineering", self.feature_engineering_transformer))

        preprocessor = self._build_preprocessor()
        steps.append(("preprocessing", preprocessor))

        if self.use_feature_selection:
            fs = self._get_feature_selector()
            steps.append(("feature_selection", fs))

        steps.append(("to_float32", ToFloat32()))

        steps.append(("model", self.model))
        return Pipeline(steps=steps)

    def fit(self, X, y) -> "SklearnPipelineRunner":
        """
        Ajusta o pipeline aos dados de treino (X, y).

        Modos:
        - use_optuna_search=True  => OptunaSearchCV
        - caso contrário          => fit direto
        """
        self._filter_existing_columns(X)

        base_pipeline = self._build_pipeline()

        if self.use_optuna_search:
            if OptunaSearchCV is None:
                raise RuntimeError("OptunaSearchCV indisponível (instale optuna).")

            self.pipeline = OptunaSearchCV(
                estimator=base_pipeline,
                param_distributions=self.optuna_param_distributions,
                cv=self.cv,
                scoring=self.scoring,
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout,
                random_state=RANDOM_STATE,
                refit=True,
                n_jobs=self.optuna_n_jobs,  # recomenda-se 1 com torch
                verbose=1,
            )
        else:
            self.pipeline = base_pipeline

        self.pipeline.fit(X, y)

        # set best_model
        if self.use_optuna_search:
            self.best_model = self.pipeline.best_estimator_
            logger.info("Melhores parâmetros: %s", self.pipeline.best_params_)
            logger.info(
                "cv_best_score=%.6f | scoring=%s",
                float(self.pipeline.best_score_),
                getattr(self, "scoring", None),
            )
            self.cv_best_score_ = float(self.pipeline.best_score_)
            self.best_params_ = dict(self.pipeline.best_params_)
        else:
            self.best_model = self.pipeline

        return self

    def _get_fitted_estimator(self):
        """Retorna o estimador já treinado que deve ser usado para predict/evaluate."""
        est = getattr(self, "best_model", None)
        if est is not None:
            return est
        est = getattr(self, "pipeline", None)
        if est is not None:
            return est
        raise RuntimeError(
            "Nenhum estimador treinado encontrado. Rode .fit() antes de .evaluate(). "
            "(best_model e pipeline estão None)."
        )

    def predict(self, X) -> pd.Series:
        est = self._get_fitted_estimator()
        return est.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        est = self._get_fitted_estimator()
        if not hasattr(est, "predict_proba"):
            raise AttributeError("Modelo não suporta predict_proba (necessário p/ AUC).")
        proba = est.predict_proba(X)
        return proba[:, 1]

    def cross_validate(self, X, y) -> np.ndarray:
        """
        Realiza validação cruzada usando o pipeline construído.

        Observação:
        - Se use_optuna_search=True, aqui o comportamento padrão será rodar o estimator com OptunaSearchCV
            dentro de cada fold do cross_val_score, o que é caro (tuning dentro de CV).
            Em geral, escolha OU CV simples OU tuning, não ambos ao mesmo tempo.
        """
        self._filter_existing_columns(X)

        estimator = self._build_pipeline()

        if self.use_optuna_search:
            estimator = OptunaSearchCV(
                estimator=estimator,
                param_distributions=self.optuna_param_distributions,
                cv=self.cv,
                scoring=self.scoring,
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout,
                random_state=RANDOM_STATE,
                refit=True,
                n_jobs=self.optuna_n_jobs,
                verbose=0,
            )

        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        logger.info("CV Scores: %s", scores)
        logger.info("Mean: %.4f | Std: %.4f", scores.mean(), scores.std())
        return scores

    def evaluate(self, X, y, include_auc: bool = True) -> dict[str, float]:
        est = self._get_fitted_estimator()
        y_pred = self.predict(X)

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(
                precision_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)
            ),
            "recall": float(recall_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="binary", pos_label=self.pos_label, zero_division=0)),
        }

        if include_auc:
            auc = None

            # tenta predict_proba
            if hasattr(est, "predict_proba"):
                try:
                    proba = est.predict_proba(X)
                    if proba is not None and getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
                        y_score = proba[:, 1]
                    else:
                        y_score = proba
                    auc = float(roc_auc_score(y, y_score))
                except Exception:
                    auc = None

            # tenta decision_function
            if auc is None and hasattr(est, "decision_function"):
                try:
                    y_score = est.decision_function(X)
                    auc = float(roc_auc_score(y, y_score))
                except Exception:
                    auc = None

            if auc is not None:
                metrics["auc_roc"] = auc

        return metrics

    def save(self, path) -> None:
        joblib.dump(self.best_model, path)

    def load(self, path) -> "SklearnPipelineRunner":
        self.best_model = joblib.load(path)
        return self


def main():
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset and features")

    X_raw = df_clean[FEATURES_COLS].copy()
    y = df_clean[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() == 2 else None,
    )

    model = LogisticRegression(random_state=RANDOM_STATE)

    runner = SklearnPipelineRunner(
        model=model,
        categorical_cols=CAT_COLS,
        numerical_cols=NUM_COLS,
        boolean_cols=BOL_COLS,
        binned_cols=BIN_COLS,
        use_feature_engineering=False,
        feature_engineering_transformer=None,
        use_feature_selection=False,
        use_optuna_search=False,
        optuna_param_distributions=None,
        optuna_n_trials=30,
        optuna_timeout=None,
        optuna_n_jobs=1,
        cv=5,
        scoring="f1",
        pos_label=1,
    )

    logger.info("Dataset: %d features | Treino: %d | Teste: %d", X_train.shape[1], len(X_train), len(X_test))
    runner.fit(X_train, y_train)

    metrics = runner.evaluate(X_test, y_test, include_auc=True)
    logger.info("Métricas: %s", metrics)

    runner.save("churn_model_pipeline.joblib")


if __name__ == "__main__":
    main()
