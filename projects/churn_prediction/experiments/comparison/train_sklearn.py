"""train_sklearn.py — Treino de múltiplos modelos sklearn com Pipeline utilizando RandomizedSearchCV
e logging no MLflow.

Inclui run_randomized_search com barra de progresso por modelo.

Uso:
    python experiments/comparison/train_sklearn.py

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    PRIMARY_METRIC,
    TARGET_COL,
    YES_NO_COLS,
    RANDOM_SEED,
    FEATURES_COLS,
    PARAM_DISTS,
    N_ITER_BY_MODEL,
)
from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.feature_engineering import TelcoFeatureEngineeringBins
from src.ml.data_utils import compute_metrics, build_preprocessor


logger = logging.getLogger(__name__)
console = Console()

def scoring_from_primary_metric(primary: str) -> str:
    """
    Mapeia a métrica principal do projeto para a string de `scoring`
    esperada pelo scikit-learn.

    Args:
        primary: Nome da métrica principal.

    Returns:
        String de scoring compatível com scikit-learn.
    """
    mapping = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    if primary not in mapping:
        raise ValueError(f"PRIMARY_METRIC='{primary}' não mapeada para sklearn scoring.")
    return mapping[primary]


def get_models() -> Dict[str, Any]:
    """
    Retorna um dicionário de estimadores scikit-learn (e XGBoost) a serem avaliados.

    Returns:
        Dicionário {nome_modelo: estimador}, onde cada estimador está configurado com
        seeds/parâmetros básicos para reprodutibilidade.
    """
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "logreg": LogisticRegression(random_state=RANDOM_SEED, max_iter=5000, solver="saga"),
        "xgboost": XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, eval_metric="aucpr", verbosity=0),
    }


def log_skops_pipeline(pipeline: Pipeline, artifact_path: str = "artifacts") -> None:
    """
    Serializa um pipeline scikit-learn para arquivo `.skops` e faz upload como artifact no MLflow.

    Args:
        pipeline: Pipeline scikit-learn já treinado.
        artifact_path: Caminho de artifacts no MLflow onde o arquivo será salvo.
    """
    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)
    p = tmp_dir / "pipeline.skops"
    sio.dump(pipeline, p)
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def print_results_table(results: List[dict], title: str) -> None:
    """
    Imprime uma tabela com o resumo de resultados por modelo:
    - melhor score de CV
    - melhores hiperparâmetros

    Args:
        results: Lista de dicionários com chaves como:
            - "model_name"
            - "best_cv_score"
            - "best_params"
        title: Título da tabela.
    """
    if not results:
        console.print("[bold red]Nenhum resultado para exibir.[/bold red]")
        return

    table = Table(title=f"\n[bold]{title}[/bold]")
    table.add_column("Model", style="cyan")
    table.add_column("best_cv_score", justify="right")
    table.add_column("best_params", justify="left")

    best_idx = max(range(len(results)), key=lambda i: results[i]["best_cv_score"])
    for i, r in enumerate(results):
        style = "bold green" if i == best_idx else ""
        marker = " *" if i == best_idx else ""
        table.add_row(
            r["model_name"] + marker,
            f"{r['best_cv_score']:.5f}",
            str(r["best_params"])[:120],
            style=style,
        )

    console.print(table)
    best = results[best_idx]
    console.print(
        f"\n[bold green]Melhor:[/bold green] {best['model_name']} "
        f"(best_cv_score={best['best_cv_score']:.5f})"
    )


def _merge_kbest_into_param_dist(param_dist: Any) -> Any:
    """
    Garante que `select_kbest__k` esteja presente no espaço de busca (`param_dist`)
    para que o RandomizedSearchCV possa testar diferentes valores de K.

    Args:
        param_dist: Estrutura de parâmetros no formato aceito pelo scikit-learn:
            - dict[str, Any] ou
            - list[dict[str, Any]]

    Returns:
        `param_dist` com `select_kbest__k` garantido quando aplicável.
    """
    kbest_key = "select_kbest__k"

    def _has_kbest(d: dict) -> bool:
        return kbest_key in d

    if isinstance(param_dist, list):
        if any(_has_kbest(d) for d in param_dist):
            return param_dist
        return [{**d, kbest_key: [15, "all"]} for d in param_dist]

    if isinstance(param_dist, dict):
        if kbest_key in param_dist:
            return param_dist
        return {**param_dist, kbest_key: [15, "all"]}

    return param_dist


def log_best_estimator_fold_metrics(best_estimator: Pipeline, X_df: pd.DataFrame, y: pd.Series) -> None:
    """
    Gera predições out-of-fold (OOF) com o `best_estimator` e loga métricas por fold no MLflow.

    Args:
        best_estimator: Pipeline treinável (pipeline completo), que será clonado e treinado fold a fold.
        X_df: DataFrame com features.
        y: Série com target.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_df, y)):
        X_tr, X_te = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_tr = y.iloc[train_idx].to_numpy().astype(int)
        y_te = y.iloc[test_idx].to_numpy().astype(int)

        est = clone(best_estimator)
        est.fit(X_tr, y_tr)

        y_pred = est.predict(X_te)

        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X_te)[:, 1]
        else:
            y_score = est.decision_function(X_te)
            y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)

        metrics = compute_metrics(y_te, y_pred, y_prob)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value), step=fold_idx)

        y_true_path = tmp_dir / f"y_true_fold_{fold_idx}.npy"
        y_proba_path = tmp_dir / f"y_proba_fold_{fold_idx}.npy"
        np.save(y_true_path, y_te.astype(int))
        np.save(y_proba_path, y_prob.astype(float))
        mlflow.log_artifact(str(y_true_path), artifact_path="oof")
        mlflow.log_artifact(str(y_proba_path), artifact_path="oof")


def run_randomsearch_for_model(
    model_name: str,
    estimator: Any,
    X_df: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Executa RandomizedSearchCV para um modelo específico, com Pipeline completo.
    Faz logging de parâmetros e artefatos no MLflow.

    Args:
        model_name: Nome do modelo (chave no dicionário de modelos).
        estimator: Estimador scikit-learn ou XGBClassifier a ser inserido no pipeline.
        X_df: DataFrame com features.
        y: Série com o target.

    Returns:
        Dicionário com:
            - model_name
            - run_id
            - best_cv_score
            - best_params
    """
    pipe = Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)),
            ("preprocess", build_preprocessor()),
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            ("select_kbest", SelectKBest(score_func=mutual_info_classif, k="all")),
            ("model", estimator),
        ]
    )

    param_dist = PARAM_DISTS.get(model_name, {})
    param_dist = _merge_kbest_into_param_dist(param_dist)

    n_iter = N_ITER_BY_MODEL.get(model_name, 50)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = scoring_from_primary_metric(PRIMARY_METRIC)

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
        random_state=RANDOM_SEED,
        return_train_score=False,
        error_score="raise",
    )

    with mlflow.start_run(run_name=f"{model_name}_randomsearch") as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("search_type", "randomized")
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", N_FOLDS)
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("scoring", scoring)

        rs.fit(X_df, y)

        mlflow.log_metric("best_cv_score", float(rs.best_score_))
        mlflow.log_params(rs.best_params_)

        best_pipe: Pipeline = rs.best_estimator_

        n_before = int(best_pipe.named_steps["preprocess"].get_feature_names_out().shape[0])
        k_best = best_pipe.named_steps["select_kbest"].k
        n_after = n_before if k_best == "all" else int(k_best)

        mlflow.log_param("best_kbest", str(k_best))
        mlflow.log_param("n_features_before_kbest", n_before)
        mlflow.log_param("n_features_after_kbest", n_after)

        log_best_estimator_fold_metrics(best_pipe, X_df, y)

        cv_df = pd.DataFrame(rs.cv_results_)
        tmp_dir = Path("mlflow_artifacts_tmp")
        tmp_dir.mkdir(exist_ok=True)
        cv_path = tmp_dir / f"{model_name}_cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

        log_skops_pipeline(best_pipe, artifact_path="artifacts")

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "best_cv_score": float(rs.best_score_),
            "best_params": rs.best_params_,
        }


def run_random_search(models: Dict[str, Any], X_df: pd.DataFrame, y: pd.Series) -> List[dict]:
    """
    Executa RandomizedSearchCV para todos os modelos fornecidos por modelo.

    Args:
        models: Dicionário {model_name: estimator}.
        X_df: DataFrame com features.
        y: Série com o target.

    Returns:
        Lista de resultados por modelo.
    """
    results: List[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]RandomizedSearchCV por modelo", total=len(models))

        for model_name, estimator in models.items():
            progress.update(task, description=f"[cyan]{model_name}")
            console.rule(f"[bold cyan]{model_name}[/bold cyan]")
            r = run_randomsearch_for_model(model_name, estimator, X_df, y)
            results.append(r)
            progress.update(task, advance=1)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    console.print("\n[bold]Carregando dataset...[/bold]")
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")

    y = df_clean[TARGET_COL].astype(int)
    X_df = df_clean[FEATURES_COLS]

    console.print(
        f"[dim]Dataset: {X_df.shape[0]} amostras, {X_df.shape[1]} features, {N_FOLDS}-fold CV[/dim]\n"
    )

    models = get_models()
    results = run_random_search(models, X_df, y)

    print_results_table(results, title="Model Comparison — Best CV Score")

    console.print("\n[bold green]Treino sklearn concluído![/bold green]")
    console.print(f"[dim]Resultados logados no MLflow: {MLFLOW_TRACKING_URI}[/dim]\n")


if __name__ == "__main__":
    main()
