"""feature_selection — Seleção simples de features via feature_importances_.

Fornecer uma forma reprodutível de selecionar um subconjunto de features após one-hot encoding, usando importâncias de um modelo
baseado em árvores.

- Fit de um RandomForestClassifier no conjunto de treino.
- Extração de feature_importances_.
- Seleção das Top-K features mais importantes.
- Aplicação do subconjunto tanto em treino quanto em teste (sem leakage, pois o selector é fitado apenas no treino).
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier

from dataclasses import dataclass
from typing import Literal, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif


@dataclass(frozen=True)
class ImportanceSelector:
    """Resultado da seleção por importância (RandomForest)."""
    selected_features_: list[str]
    importances_: pd.Series


@dataclass(frozen=True)
class AnovaSelector:
    """Resultado da seleção por ANOVA (SelectKBest)."""
    selected_features_: list[str]
    scores_: pd.Series
    pvalues_: pd.Series
    selector_: SelectKBest


def fit_feature_selector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
    random_state: int,
    feature_selection: Literal["feature_importance", "anova"],
) -> Union[ImportanceSelector, AnovaSelector]:
    """
    O usuário escolhe o tipo de seleção via `feature_selection`:
    - "feature_importance": retorna ImportanceSelector
    - "anova": retorna AnovaSelector
    """
    if top_k <= 0:
        raise ValueError("top_k deve ser > 0")

    if feature_selection == "feature_importance":
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)

        importances = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
            name="importance",
        ).sort_values(ascending=False)

        selected = importances.head(min(top_k, X_train.shape[1])).index.tolist()
        return ImportanceSelector(selected_features_=selected, importances_=importances)

    if feature_selection == "anova":
        k = min(top_k, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)

        scores = pd.Series(selector.scores_, index=X_train.columns, name="anova_score")
        pvalues = pd.Series(selector.pvalues_, index=X_train.columns, name="pvalue")

        selected_mask = selector.get_support()
        selected = X_train.columns[selected_mask].tolist()

        return AnovaSelector(
            selected_features_=selected,
            scores_= scores.sort_values(ascending=False),
            pvalues_= pvalues.loc[scores.sort_values(ascending=False).index],
            selector_= selector,
        )

    raise ValueError("feature_selection deve ser 'feature_importance' ou 'anova'")


def apply_selector(
    X: pd.DataFrame,
    selector: Union[ImportanceSelector, AnovaSelector],
) -> pd.DataFrame:
    """
    Aplica o selector em um DataFrame, mantendo apenas as features selecionadas.

    Args:
        X: DataFrame de entrada (treino/teste) com colunas compatíveis.
        selector: ImportanceSelector gerado no treino.

    Returns:
        pd.DataFrame: DataFrame contendo somente as colunas em selector.selected_features_.
    """
    return X.loc[:, selector.selected_features_].copy()
