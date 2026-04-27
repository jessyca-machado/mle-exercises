"""
Grid search de múltiplos modelos sklearn com Pipeline (preprocess + model),
K-Fold CV via RandomizedSearchCV e logging no MLflow.

Inclui run_grid_search com barra de progresso (Rich) por modelo.

Uso:
    python experiments/comparison/comparison_model.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,  
)
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif

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
    RANDOM_SEED,
    PRIMARY_METRIC,
    TARGET_COL,
    YES_NO_COLS,
    RANDOM_STATE,
    FEATURES_COLS,
    CAT_COLS,
    NUM_COLS,
    BOL_COLS,
    BIN_COLS,
    # GRID_PARAM_GRIDS,
    PARAM_DISTS,
    N_ITER_BY_MODEL,
)
from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from src.data.feature_engineering import TelcoFeatureEngineeringBins


logger = logging.getLogger(__name__)
console = Console()

def build_preprocessor_from_df(df: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in NUM_COLS if c in df.columns]
    bol = [c for c in BOL_COLS if c in df.columns]
    bin_cols = [c for c in BIN_COLS if c in df.columns]

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }


def scoring_from_primary_metric(primary: str) -> str:
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


def get_models():
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logreg": LogisticRegression(random_state=RANDOM_STATE, max_iter=5000, solver="saga"),
        "xgboost": XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="aucpr", verbosity=0),
    }


def log_skops_pipeline(pipeline: Pipeline, artifact_path: str = "artifacts") -> None:
    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)
    p = tmp_dir / "pipeline.skops"
    sio.dump(pipeline, p)
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def print_results_table(results: List[dict], title: str):
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


def _merge_kbest_into_param_dist(param_dist):
    """
    Garante que select_kbest__k vindo do PARAM_DISTS seja aplicado.
    - Se já existir no param_dist, não sobrescreve.
    - Se param_dist for list[dict], adiciona em cada dict.
    """
    kbest_key = "select_kbest__k"

    # se o user já colocou no param_dist, não mexe
    def _has_kbest(d: dict) -> bool:
        return kbest_key in d

    if isinstance(param_dist, list):
        if any(_has_kbest(d) for d in param_dist):
            return param_dist
        # fallback: se você quiser sempre forçar, troque aqui
        return [{**d, kbest_key: [15, "all"]} for d in param_dist]

    if isinstance(param_dist, dict):
        if kbest_key in param_dist:
            return param_dist
        return {**param_dist, kbest_key: [15, "all"]}

    return param_dist
    

def run_randomsearch_for_model(model_name: str, estimator, X_df: pd.DataFrame, y: pd.Series) -> dict:
    fe_example = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    X_example_fe = fe_example.fit_transform(X_df.head(200).copy())  # só para inferir colunas
    preprocessor = build_preprocessor_from_df(X_example_fe)

    pipe = Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)),
            ("preprocess", preprocessor),
            ("select_kbest", SelectKBest(score_func=f_classif, k="all")),
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

        # log quantidades de features para o melhor pipeline
        best_pipe = rs.best_estimator_
        # número de features após preprocess (antes do SelectKBest)
        n_before = int(best_pipe.named_steps["preprocess"].get_feature_names_out().shape[0])
        k_best = best_pipe.named_steps["select_kbest"].k
        n_after = n_before if k_best == "all" else int(k_best)
        mlflow.log_param("best_kbest", str(k_best))
        mlflow.log_param("n_features_before_kbest", n_before)
        mlflow.log_param("n_features_after_kbest", n_after)

        # métricas por fold + OOF artifacts (para seleção estatística e business toolkit)
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


def run_grid_search(models: Dict[str, Any], X_df: pd.DataFrame, y: pd.Series) -> List[dict]:
    """Executa RandomizedSearchCV para todos os modelos com barra de progresso (por modelo)."""
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


def log_best_estimator_fold_metrics(best_estimator: Pipeline, X_df: pd.DataFrame, y: pd.Series) -> None:
    """
    Gera predições out-of-fold (OOF) com o best_estimator e loga métricas por fold no MLflow,
    usando o MESMO esquema de folds do treino (StratifiedKFold com RANDOM_SEED).
    
    Importante: isso não é "o mesmo" que os scores internos do RandomizedSearchCV,
    mas é consistente e comparável entre modelos para teste estatístico.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)

    # Para ter métricas por fold, fazemos loop manual nos splits
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
    results = run_grid_search(models, X_df, y)

    print_results_table(results, title="Model Comparison — Best CV Score")

    console.print("\n[bold green]Treino sklearn concluído![/bold green]")
    console.print(f"[dim]Resultados logados no MLflow: {MLFLOW_TRACKING_URI}[/dim]\n")


if __name__ == "__main__":
    main()
