"""train_mlp_torch.py — Arquitetura de MLP em PyTorch para treino estratificado com CV manual.

Uso:
    python experiments/deep_learning/train_mlp_torch.py
"""
from __future__ import annotations

from pathlib import Path
import requests
from mlflow.exceptions import MlflowException

from typing import Iterable, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import pandas as pd

import logging

import mlflow
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    METRICS,
    MLFLOW_TRACKING_URI,
    MLP_GRID,
    N_FOLDS,
    PRIMARY_METRIC,
    RANDOM_SEED,
    FEATURES_COLS,
    YES_NO_COLS,TARGET_COL,
)
from src.data.load_data import load_data_churn
from src.data.preprocess import pre_processing
from sklearn.feature_selection import SelectKBest, f_classif
from src.data.feature_engineering import TelcoFeatureEngineeringBins
from sklearn.base import BaseEstimator
from src.ml.data_utils import compute_metrics, build_preprocessor_from_df

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from mlflow.models.signature import ModelSignature

import mlflow.pyfunc
import skops.io as sio

from mlflow.models import infer_signature


HiddenDims = Union[str, Tuple[int, ...], Iterable[int]]

logger = logging.getLogger(__name__)
console = Console()


def log_end_to_end_model(
    feature_engineering: BaseEstimator,
    preprocessor: Any,
    selector: BaseEstimator,
    model: torch.nn.Module,
    input_dim: int,
    input_example: pd.DataFrame,
    signature: ModelSignature,
) -> None:
    """
    Registra no MLflow um modelo end-to-end para produção, incluindo:
    - artefatos do pipeline de transformação (feature engineering, preprocess e seleção)
    - modelo PyTorch exportado como TorchScript
    - wrapper PyFunc para inferência end-to-end

    A função salva os componentes em disco e faz upload como artifacts no MLflow via `mlflow.pyfunc.log_model`.

    Args:
        feature_engineering: Transformer/estimador scikit-learn responsável pelo feature engineering.
        preprocessor: Objeto responsável pelo preprocess (tipicamente `ColumnTransformer` ou pipeline).
        selector: Estimador scikit-learn responsável pela seleção de features.
        model: Modelo PyTorch (`torch.nn.Module`) treinado.
        input_dim: Dimensão de entrada esperada pelo modelo após preprocess + seleção.
        input_example: Exemplo de entrada crua (DataFrame) utilizado para registrar o modelo.
        signature: Assinatura do modelo (MLflow ModelSignature) inferida a partir do input/output.

    """
    artifacts_dir = Path("mlflow_artifacts_tmp")
    artifacts_dir.mkdir(exist_ok=True)

    fe_path = artifacts_dir / "feature_engineering.skops"
    sio.dump(feature_engineering, fe_path)

    preproc_path = artifacts_dir / "preprocessor.skops"
    sio.dump(preprocessor, preproc_path)

    selector_path = artifacts_dir / "selector.skops"
    sio.dump(selector, selector_path)

    model.eval()
    device = next(model.parameters()).device
    example = torch.randn(1, input_dim, device=device)
    ts = torch.jit.trace(model, example).cpu()
    ts_path = artifacts_dir / "model_ts.pt"
    ts.save(str(ts_path))

    code_path = Path("src/ml/churn_pyfunc.py").resolve()
    mlflow.pyfunc.log_model(
        name="model",
        python_model=str(code_path),
        artifacts={
            "feature_engineering": str(fe_path.resolve()),
            "preprocessor": str(preproc_path.resolve()),
            "selector": str(selector_path.resolve()),
            "torchscript_model": str(ts_path.resolve()),
        },
        pip_requirements="requirements-mlflow.txt",
        input_example=input_example,
        signature=signature,
    )


class MLP(nn.Module):
    """
    MLP simples para classificação binária:
        - camadas densas + ReLU + Dropout
        - saída: 1 logit (sem sigmoid no forward)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: HiddenDims = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, str):

            hidden_dims = tuple(int(x.strip()) for x in hidden_dims.split(",") if x.strip())

        hidden_dims = tuple(hidden_dims)

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass.

        Args:
            x: Tensor de entrada com shape (batch_size, input_dim).

        Returns:
            Tensor de logits com shape (batch_size,). A sigmoid não é aplicada aqui.
        """
        return self.network(x).squeeze(1)


def _is_http_uri(uri: str) -> bool:
    """
    Verifica se uma URI é HTTP/HTTPS.

    Args:
        uri: URI a ser verificada.

    Returns:
        True se `uri` começa com "http://" ou "https://", caso contrário False.
    """
    return uri.startswith("http://") or uri.startswith("https://")


def setup_mlflow(tracking_uri: str, experiment_name: str, fallback_dir: str = "mlruns") -> None:
    """
    Configura o MLflow para logging.

    Comportamento:
    - Se `tracking_uri` for HTTP/HTTPS, tenta conectar.
    - Se falhar, faz fallback para um backend local (`file:<fallback_dir>`).
    - Garante que o experimento exista e o seleciona.

    Args:
        tracking_uri: URI do tracking server do MLflow.
        experiment_name: Nome do experimento no MLflow.
        fallback_dir: Diretório local utilizado no fallback quando o server HTTP não estiver acessível.
    """
    if _is_http_uri(tracking_uri):
        try:
            requests.get(tracking_uri, timeout=1.5)
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as e:
            local_uri = f"file:{Path(fallback_dir).resolve()}"
            logging.warning(
                "Não foi possível conectar no MLflow em %s (%s). "
                "Fazendo fallback para %s",
                tracking_uri, repr(e), local_uri
            )
            mlflow.set_tracking_uri(local_uri)
    else:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        exp_id = mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logging.info("Experimento criado: %s (id=%s)", experiment_name, exp_id)


def set_seed(seed: int) -> None:
    """
    Define seeds para reprodutibilidade (NumPy e PyTorch).

    Args:
        seed: Valor da seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def get_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Gera splits de validação cruzada estratificada.

    Args:
        X: Matriz de features.
        y: Vetor de rótulos.
        n_folds: Número de folds.
        seed: Seed para embaralhamento e reprodutibilidade.

    Returns:
        Lista de tuplas (train_idx, test_idx), onde cada item contém arrays de índices.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def train_one_fold(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
) -> tuple[MLP, list[float]]:
    """
    Treina uma MLP em um fold com early stopping baseado na loss de validação.
    Também utiliza `pos_weight` em `BCEWithLogitsLoss` para lidar com desbalanceamento.

    Args:
        model: Instância do modelo MLP.
        X_train: Features de treino (numpy array float32).
        y_train: Rótulos de treino (numpy array float32).
        X_val: Features de validação.
        y_val: Rótulos de validação.
        params: Dicionário de hiperparâmetros, esperado conter:
            - "batch_size"
            - "learning_rate"
            - "max_epochs"
            - "patience"

    Returns:
        Uma tupla com:
            - modelo treinado, com melhor estado restaurado
            - lista de losses de treino por época
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    best_state = None

    for epoch in range(params["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                logger.debug("Early stopping na epoca %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses


def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    """
    Gera probabilidades de predição a partir de um modelo MLP treinado.

    Args:
        model: Modelo MLP treinado.
        X: Features (numpy array) no espaço já transformado (após preprocess + seleção).

    Returns:
        Array NumPy com probabilidades da classe positiva, shape (n_amostras,).
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_config_cv(
    params: dict[str, Any],
    run_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    preprocessor_base: ColumnTransformer,
) -> dict[str, Any]:
    """
    Treina e avalia uma configuração de MLP com K-Fold CV (loop manual).

    Para cada fold:
    - aplica feature engineering + preprocess + SelectKBest
    - treina a MLP com early stopping
    - calcula métricas e salva predições OOF (y_true e y_proba)

    Args:
        params: Hiperparâmetros da MLP e do treino (ex.: hidden_layers, dropout, batch_size, etc.).
        run_name: Nome identificador da configuração (utilizado para identificar o resultado).
        X: DataFrame com features cruas.
        y: Série com o target.
        splits: Lista de tuplas (train_idx, test_idx) geradas por `get_cv_splits`.
        preprocessor_base: Preprocessador base (ColumnTransformer) a ser clonado por fold.

    Returns:
        Dicionário de resumo contendo:
            - "params": params de entrada
            - "k_best": k do SelectKBest
            - "cv_mean": médias por métrica
            - "cv_std": desvios por métrica
            - "fold_oof": lista por fold com y_true, y_proba e métricas
    """
    fold_metrics = {m: [] for m in METRICS}
    fold_oof = []

    k_best = params.get("k_best", "all")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        set_seed(RANDOM_SEED + fold_idx)

        X_train_df = X.iloc[train_idx]
        X_test_df = X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy(dtype=np.float32)
        y_test = y.iloc[test_idx].to_numpy(dtype=np.float32)

        preprocessor_fold = clone(preprocessor_base)

        prep_pipe = make_prep_pipe(preprocessor_base, k_best)
        X_train_sel = prep_pipe.fit_transform(X_train_df, y_train.astype(int)).astype(np.float32)
        X_test_sel = prep_pipe.transform(X_test_df).astype(np.float32)

        model = MLP(X_train_sel.shape[1], params["hidden_layers"], params["dropout"])
        model, train_losses = train_one_fold(model, X_train_sel, y_train, X_test_sel, y_test, params)

        y_prob = predict_proba(model, X_test_sel)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test.astype(int), y_pred, y_prob)
        for m in METRICS:
            fold_metrics[m].append(metrics[m])

        fold_oof.append({
            "fold": fold_idx,
            "y_true": y_test.astype(int),
            "y_proba": y_prob.astype(float),
            "metrics": metrics,
            "n_features_after": int(X_train_sel.shape[1]),
        })

    summary = {
        "params": params,
        "k_best": k_best,
        "cv_mean": {m: float(np.mean(fold_metrics[m])) for m in METRICS},
        "cv_std": {m: float(np.std(fold_metrics[m], ddof=1)) if len(fold_metrics[m]) > 1 else 0.0 for m in METRICS},
        "fold_oof": fold_oof,
    }
    return summary


def print_results_table(results: list[dict[str, Any]], title: str) -> None:
    """
    Imprime uma tabela com o resumo de desempenho por configuração de MLP.
    A tabela destaca a melhor configuração de acordo com `PRIMARY_METRIC`.

    Args:
        results: Lista de dicionários no formato retornado por `train_config_cv`, enriquecidos
            com "run_name".
        title: Título da tabela.
    """
    table = Table(title=f"\n[bold]{title}[/bold]")
    table.add_column("Config", style="cyan")
    table.add_column(f"cv_mean_{PRIMARY_METRIC}", justify="right")
    table.add_column(f"cv_std_{PRIMARY_METRIC}", justify="right")
    table.add_column("cv_mean_accuracy", justify="right")
    table.add_column("cv_mean_precision", justify="right")
    table.add_column("cv_mean_recall", justify="right")
    table.add_column("cv_mean_roc_auc", justify="right")

    best_idx = max(range(len(results)), key=lambda i: results[i]["cv_mean"][PRIMARY_METRIC])

    for i, r in enumerate(results):
        style = "bold green" if i == best_idx else ""
        marker = " *" if i == best_idx else ""
        table.add_row(
            r["run_name"] + marker,
            f"{r['cv_mean'][PRIMARY_METRIC]:.4f}",
            f"{r['cv_std'][PRIMARY_METRIC]:.4f}",
            f"{r['cv_mean']['accuracy']:.4f}",
            f"{r['cv_mean']['precision']:.4f}",
            f"{r['cv_mean']['recall']:.4f}",
            f"{r['cv_mean']['roc_auc']:.4f}",
            style=style,
        )

    console.print(table)
    best = results[best_idx]
    console.print(
        f"\n[bold green]Melhor:[/bold green] {best['run_name']} "
        f"({PRIMARY_METRIC}={best['cv_mean'][PRIMARY_METRIC]:.4f})"
    )


def make_prep_pipe(
    preprocessor_base: ColumnTransformer,
    k_best: Union[int, str],
    fe_kwargs: Optional[dict[str, Any]] = None,
) -> Pipeline:
    """
    Cria um pipeline de preparação de dados (feature engineering + preprocess + seleção).

    Args:
        preprocessor_base: Preprocessador base a ser clonado.
        k_best: Valor de k para SelectKBest (int ou "all").
        fe_kwargs: Parâmetros para o feature engineering (default: quantis padrão).

    Returns:
        Um `Pipeline` scikit-learn configurado.
    """
    fe_kwargs = fe_kwargs or dict(monthlycharges_q=5, totalcharges_q=10)
    return Pipeline(steps=[
        ("feature_engineering", TelcoFeatureEngineeringBins(**fe_kwargs)),
        ("preprocess", clone(preprocessor_base)),
        ("select_kbest", SelectKBest(score_func=f_classif, k=k_best)),
    ])


def fit_transform_fold(
    prep_pipe: Pipeline,
    X_train_df: pd.DataFrame,
    y_train_int: np.ndarray,
    X_test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ajusta o pipeline de preparação no fold de treino e transforma treino e teste.

    Args:
        prep_pipe: Pipeline de preparação (feature engineering + preprocess + seleção).
        X_train_df: Features cruas de treino.
        y_train_int: Target de treino como int (0/1), utilizado pelo SelectKBest.
        X_test_df: Features cruas de teste (DataFrame).

    Returns:
        Tupla (X_train_sel, X_test_sel) com arrays float32 após transformações.
    """
    X_train_sel = prep_pipe.fit_transform(X_train_df, y_train_int).astype(np.float32)
    X_test_sel = prep_pipe.transform(X_test_df).astype(np.float32)
    return X_train_sel, X_test_sel


def fit_eval_mlp(
    best_params: dict[str, Any],
    X_train_sel: np.ndarray,
    y_train_f: np.ndarray,
    X_test_sel: np.ndarray,
    y_test_f: np.ndarray,
) -> tuple[MLP, list[float], np.ndarray, dict[str, float]]:
    """
    Treina uma MLP em um fold e calcula métricas no conjunto de teste do fold.

    Args:
        best_params: Hiperparâmetros do modelo/treino (contendo hidden_layers e dropout).
        X_train_sel: Features de treino já transformadas/selecionadas.
        y_train_f: Target de treino float32.
        X_test_sel: Features de teste já transformadas/selecionadas.
        y_test_f: Target de teste float32.

    Returns:
        Tupla contendo:
            - modelo treinado
            - lista de losses por época
            - probabilidades previstas no teste (classe positiva)
            - dicionário de métricas calculadas
    """
    model = MLP(X_train_sel.shape[1], best_params["hidden_layers"], best_params["dropout"])
    model, train_losses = train_one_fold(model, X_train_sel, y_train_f, X_test_sel, y_test_f, best_params)
    y_prob = predict_proba(model, X_test_sel)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test_f.astype(int), y_pred, y_prob)
    return model, train_losses, y_prob, metrics


def run_cv_mlp(
    best_params: dict[str, Any],
    X_df: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    preprocessor_base: ColumnTransformer,
) -> dict[str, Any]:
    """
    Executa K-Fold CV para um conjunto de hiperparâmetros do MLP e retorna um resumo.

    Args:
        best_params: Hiperparâmetros do MLP/treino.
        X_df: Features cruas (DataFrame).
        y: Target (Series).
        splits: Lista de splits (train_idx, test_idx).
        preprocessor_base: Preprocessador base a ser clonado.

    Returns:
        Dicionário contendo:
            - "k_best"
            - "cv_mean"
            - "cv_std"
            - "fold_oof" (inclui métricas e predições por fold)
    """
    k_best = best_params.get("k_best", "all")
    fold_scores = {m: [] for m in METRICS}
    fold_oof = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        set_seed(RANDOM_SEED + fold_idx)

        X_train_df, X_test_df = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train_f = y.iloc[train_idx].to_numpy(dtype=np.float32)
        y_test_f = y.iloc[test_idx].to_numpy(dtype=np.float32)

        prep_pipe = make_prep_pipe(preprocessor_base, k_best)
        X_train_sel, X_test_sel = fit_transform_fold(prep_pipe, X_train_df, y_train_f.astype(int), X_test_df)

        _, train_losses, y_prob, metrics = fit_eval_mlp(best_params, X_train_sel, y_train_f, X_test_sel, y_test_f)

        for m in METRICS:
            fold_scores[m].append(float(metrics[m]))

        fold_oof.append({
            "fold": fold_idx,
            "y_true": y_test_f.astype(int),
            "y_proba": y_prob.astype(float),
            "metrics": metrics,
            "n_epochs": len(train_losses),
        })

    cv_mean = {m: float(np.mean(fold_scores[m])) for m in METRICS}
    cv_std = {m: float(np.std(fold_scores[m], ddof=1)) if len(fold_scores[m]) > 1 else 0.0 for m in METRICS}

    return {"k_best": k_best, "cv_mean": cv_mean, "cv_std": cv_std, "fold_oof": fold_oof}


def log_cv_oof_to_mlflow(cv_summary: dict[str, Any], run_prefix: str = "") -> None:
    """
    Loga no MLflow:
    - métricas agregadas (cv_mean_*, cv_std_*)
    - métricas por fold (step=fold_idx)
    - artifacts OOF por fold em `oof/` (y_true_fold_k.npy e y_proba_fold_k.npy)

    Args:
        cv_summary: Resumo no formato retornado por `run_cv_mlp`.
        run_prefix: Prefixo opcional para namespacing de métricas agregadas.
    """
    tmp_dir = Path("mlflow_artifacts_tmp"); tmp_dir.mkdir(exist_ok=True)

    for m, v in cv_summary["cv_mean"].items():
        mlflow.log_metric(f"{run_prefix}cv_mean_{m}", float(v))
    for m, v in cv_summary["cv_std"].items():
        mlflow.log_metric(f"{run_prefix}cv_std_{m}", float(v))

    for item in cv_summary["fold_oof"]:
        fold_idx = int(item["fold"])
        for metric_name, value in item["metrics"].items():
            mlflow.log_metric(metric_name, float(value), step=fold_idx)
        mlflow.log_metric("n_epochs", float(item["n_epochs"]), step=fold_idx)

        y_true_path = tmp_dir / f"y_true_fold_{fold_idx}.npy"
        y_proba_path = tmp_dir / f"y_proba_fold_{fold_idx}.npy"
        np.save(y_true_path, item["y_true"])
        np.save(y_proba_path, item["y_proba"])
        mlflow.log_artifact(str(y_true_path), artifact_path="oof")
        mlflow.log_artifact(str(y_proba_path), artifact_path="oof")


def refit_final_mlp(
    best_params: dict[str, Any],
    X_df: pd.DataFrame,
    y: pd.Series,
    preprocessor_base: ColumnTransformer,
) -> dict[str, Any]:
    """
    Faz o refit final do MLP, ajustando:
    - feature engineering em todos os dados
    - preprocess em todos os dados
    - seleção de features em todos os dados
    - modelo final (treinado em 100%)

    O número de épocas é estimado via early stopping, e depois o modelo final
    é treinado em 100% dos dados por esse número de épocas.

    Args:
        best_params: Hiperparâmetros do MLP/treino.
        X_df: Features cruas completas.
        y: Target completo.
        preprocessor_base: Preprocessador base a ser clonado.

    Returns:
        Dicionário contendo:
            - "fe": feature engineering ajustado
            - "preprocessor": preprocessador ajustado
            - "selector": seletor ajustado
            - "model": modelo final treinado
            - "input_dim": dimensão final de entrada
            - "best_epochs": número de épocas estimado
    """
    k_best = best_params.get("k_best", "all")
    fe_kwargs = dict(monthlycharges_q=5, totalcharges_q=10)

    fe_final = TelcoFeatureEngineeringBins(**fe_kwargs)
    X_full_fe = fe_final.fit_transform(X_df)

    preprocessor_final = clone(preprocessor_base)
    X_full_pp = preprocessor_final.fit_transform(X_full_fe).astype(np.float32)

    selector_final = SelectKBest(score_func=f_classif, k=k_best)
    y_full = y.to_numpy(dtype=np.float32)
    X_full = selector_final.fit_transform(X_full_pp, y_full.astype(int)).astype(np.float32)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
    tr_idx, va_idx = next(sss.split(X_full, y_full.astype(int)))
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_va, y_va = X_full[va_idx], y_full[va_idx]

    set_seed(RANDOM_SEED)
    model_tmp = MLP(X_full.shape[1], best_params["hidden_layers"], best_params["dropout"])
    _, losses = train_one_fold(model_tmp, X_tr, y_tr, X_va, y_va, best_params)
    best_epochs = len(losses)

    params_full = {**best_params, "max_epochs": int(best_epochs), "patience": int(best_epochs) + 1}
    set_seed(RANDOM_SEED)
    model_final = MLP(X_full.shape[1], best_params["hidden_layers"], best_params["dropout"])
    model_final, _ = train_one_fold(model_final, X_full, y_full, X_full, y_full, params_full)

    return {
        "fe": fe_final,
        "preprocessor": preprocessor_final,
        "selector": selector_final,
        "model": model_final,
        "input_dim": int(X_full.shape[1]),
        "best_epochs": int(best_epochs),
    }


def log_best_mlp_run(
    best_params: dict[str, Any],
    X_df: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    preprocessor_base: ColumnTransformer,
) -> None:
    """
    Executa novamente o CV do melhor MLP e loga:
    - métricas por fold
    - artifacts OOF por fold
    - métricas agregadas (cv_mean/cv_std)
    - refit final em 100% dos dados
    - modelo end-to-end para produção (pyfunc + TorchScript + artifacts do pipeline)

    Args:
        best_params: Hiperparâmetros do melhor MLP.
        X_df: Features cruas.
        y: Target.
        splits: Splits de CV.
        preprocessor_base: Preprocessador base a ser clonado.
    """
    k_best = best_params.get("k_best", "all")
    mlflow.log_param("k_best", str(k_best))
    mlflow.log_param("fe_monthlycharges_q", 5)
    mlflow.log_param("fe_totalcharges_q", 10)

    cv_summary = run_cv_mlp(best_params, X_df, y, splits, preprocessor_base)
    log_cv_oof_to_mlflow(cv_summary)

    final = refit_final_mlp(best_params, X_df, y, preprocessor_base)
    mlflow.log_param("final_best_epochs", final["best_epochs"])
    mlflow.log_param("final_input_dim", final["input_dim"])

    input_example = X_df.head(50).copy()
    int_cols = input_example.select_dtypes(include=["int", "int32", "int64"]).columns
    if len(int_cols) > 0:
        input_example[int_cols] = input_example[int_cols].astype("float64")

    X_ex = final["selector"].transform(
        final["preprocessor"].transform(final["fe"].transform(input_example)).astype(np.float32)
    ).astype(np.float32)

    output_example = predict_proba(final["model"], X_ex)
    signature = infer_signature(input_example, output_example)

    log_end_to_end_model(
        feature_engineering=final["fe"],
        preprocessor=final["preprocessor"],
        selector=final["selector"],
        model=final["model"],
        input_dim=final["input_dim"],
        input_example=input_example,
        signature=signature,
    )


def log_config_run_to_mlflow(summary: dict[str, Any], run_name: str) -> None:
    """
    Loga no MLflow os resultados de uma configuração do MLP (um run por configuração).

    São logados:
    - parâmetros do modelo/treino
    - métricas agregadas (cv_mean_*, cv_std_*)
    - métricas por fold (step=fold_idx)
    - artifacts OOF por fold em `oof/`

    Args:
        summary: Resumo retornado por `train_config_cv`.
        run_name: Nome do run/configuração, utilizado para identificar o run no MLflow.
    """
    params = summary["params"]
    mlflow.log_param("model_name", "mlp")
    mlflow.log_param("config_name", run_name)
    mlflow.log_param("search_type", "manual_grid_run_per_config")
    mlflow.log_param("cv_folds", N_FOLDS)
    mlflow.log_param("primary_metric", PRIMARY_METRIC)

    for k, v in params.items():
        mlflow.log_param(f"mlp_{k}", v)
    mlflow.log_param("k_best", str(summary.get("k_best", "all")))

    for m, v in summary["cv_mean"].items():
        mlflow.log_metric(f"cv_mean_{m}", float(v))
    for m, v in summary["cv_std"].items():
        mlflow.log_metric(f"cv_std_{m}", float(v))

    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)
    for fold_item in summary["fold_oof"]:
        fold_idx = int(fold_item["fold"])
        metrics = fold_item["metrics"]
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value), step=fold_idx)

        y_true_path = tmp_dir / f"{run_name}_y_true_fold_{fold_idx}.npy"
        y_proba_path = tmp_dir / f"{run_name}_y_proba_fold_{fold_idx}.npy"
        np.save(y_true_path, fold_item["y_true"])
        np.save(y_proba_path, fold_item["y_proba"])
        mlflow.log_artifact(str(y_true_path), artifact_path="oof")
        mlflow.log_artifact(str(y_proba_path), artifact_path="oof")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
    setup_mlflow(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, fallback_dir="mlruns")

    console.print("\n[bold]Carregando dataset...[/bold]")
    df = load_data_churn()
    df_clean = pre_processing(df, YES_NO_COLS, "Cleaned dataset")

    y = df_clean[TARGET_COL].astype(int)
    X_df = df_clean[FEATURES_COLS].copy()

    fe_example = TelcoFeatureEngineeringBins(monthlycharges_q=5, totalcharges_q=10)
    X_example_fe = fe_example.fit_transform(X_df.head(200).copy())
    preprocessor_base = build_preprocessor_from_df(X_example_fe)

    splits = get_cv_splits(X_df, y, N_FOLDS, RANDOM_SEED)

    console.print(
        f"[dim]Dataset: {X_df.shape[0]} amostras, {X_df.shape[1]} features | "
        f"CV={N_FOLDS} folds | configs={len(MLP_GRID)}[/dim]\n"
    )

    console.rule("[bold cyan]MLP — 1 run por config (MLflow)[/bold cyan]")

    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Treinando configs do MLP", total=len(MLP_GRID))

        for i, params in enumerate(MLP_GRID):
            kb = params.get("k_best", "NA")
            run_name = f"mlp_config_{i}_kbest_{kb}"
            progress.update(task, description=f"[cyan]{run_name}")

            summary = train_config_cv(params, run_name, X_df, y, splits, preprocessor_base)
            results.append({"run_name": run_name, **summary})

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("model_name", "mlp")
                mlflow.log_param("config_name", run_name)
                mlflow.log_param("search_type", "manual_grid_run_per_config")
                mlflow.log_param("cv_folds", N_FOLDS)
                mlflow.log_param("primary_metric", PRIMARY_METRIC)

                mlflow.log_params({f"mlp_{k}": v for k, v in params.items()})
                mlflow.log_param("k_best", str(summary.get("k_best", "all")))
                mlflow.log_param("fe_monthlycharges_q", 5)
                mlflow.log_param("fe_totalcharges_q", 10)

                for m in METRICS:
                    mlflow.log_metric(f"cv_mean_{m}", float(summary["cv_mean"][m]))
                    mlflow.log_metric(f"cv_std_{m}", float(summary["cv_std"][m]))
                    mlflow.log_metric("best_cv_score", summary["cv_mean"]["recall"])

                for fold_item in summary["fold_oof"]:
                    fold_idx = int(fold_item["fold"])
                    for metric_name, value in fold_item["metrics"].items():
                        mlflow.log_metric(metric_name, float(value), step=fold_idx)

                    y_true_path = tmp_dir / f"{run_name}_y_true_fold_{fold_idx}.npy"
                    y_proba_path = tmp_dir / f"{run_name}_y_proba_fold_{fold_idx}.npy"
                    np.save(y_true_path, fold_item["y_true"])
                    np.save(y_proba_path, fold_item["y_proba"])
                    mlflow.log_artifact(str(y_true_path), artifact_path="oof")
                    mlflow.log_artifact(str(y_proba_path), artifact_path="oof")

            progress.update(task, advance=1)

    print_results_table(results, title="MLP — Best CV Score")

    best_idx = max(range(len(results)), key=lambda i: results[i]["cv_mean"][PRIMARY_METRIC])
    best = results[best_idx]
    console.print(f"\n[bold green]Melhor config:[/bold green] {best['run_name']}")

    rows = []
    for r in results:
        row = {"run_name": r["run_name"]}
        for k, v in r["params"].items():
            row[f"mlp_{k}"] = v
        for m in METRICS:
            row[f"cv_mean_{m}"] = r["cv_mean"][m]
            row[f"cv_std_{m}"] = r["cv_std"][m]
        rows.append(row)

    cv_df = pd.DataFrame(rows).sort_values(by=f"cv_mean_{PRIMARY_METRIC}", ascending=False)
    cv_path = tmp_dir / "mlp_manual_search_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)

    console.rule("[bold cyan]MLflow — refit final + log do modelo campeão[/bold cyan]")
    with mlflow.start_run(run_name="mlp_best_refit_and_model") as run:
        mlflow.log_param("model_name", "mlp")
        mlflow.log_param("search_type", "manual_grid_best_refit")
        mlflow.log_param("best_config_name", best["run_name"])
        mlflow.log_params({f"mlp_{k}": v for k, v in best["params"].items()})
        mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

        log_best_mlp_run(best["params"], X_df, y, splits, preprocessor_base)

    console.print("\n[bold green]Treino MLP concluído (1 run por config + modelo final)![/bold green]")
    console.print(f"[dim]Resultados logados no MLflow: {MLFLOW_TRACKING_URI}[/dim]\n")


if __name__ == "__main__":
    main()
