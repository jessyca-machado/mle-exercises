"""
Avaliação de custo/benefício multi-modelo (Telco churn) a partir de runs no MLflow.

Assume que cada run candidato possui artifacts por fold:
- oof/y_true_fold_{k}.npy
- oof/y_proba_fold_{k}.npy

E que N_FOLDS é o mesmo para todos.

Uso:
    # cálculo do trade off de custo normal, assumindo threshold=0.5:
    python experiments/selection/cost_toolkit_metrics.py --metric net_value

    # cálculo do trade off de custo somente para os modelos com recall de treino >= 0.8,
    # buscando o threshold que maximize o valor líquido:
    python experiments/selection/cost_toolkit_metrics.py \
        --sweep-thresholds 0.3 0.95 0.05 \
        --gate-best-cv-score 0.8 \
        --gate-metric-name best_cv_score \
        --log-mlflow

Para visualizar:
    mlflow ui --backend-store-uri sqlite:///mlflow.db # Inicia UI em http://localhost:5000
"""
import logging
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import mlflow
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
)
from src.ml.mlflow_selection_utils import (
    get_latest_runs_with_mlp_from_refit,
    run_display_name,
    has_any_oof,
)
from src.ml.cost_utils import CostSpec, sweep_thresholds_cost
from src.entrypoints.cli import parse_args

logger = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class BusinessScenario:
    """Especificação de custos/benefícios do cenário de negócio."""
    benefit_tp: float = 10.0
    cost_fp: float = 1.0
    cost_fn: float = 5.0


def net_value_from_confusion(
        tp: int,
        fp: int,
        fn: int,
        scenario: BusinessScenario
) -> float:
    """
    Calcula o valor líquido (net value) a partir de TP/FP/FN e de um cenário de negócio.

    Args:
        tp: Número de verdadeiros positivos.
        fp: Número de falsos positivos.
        fn: Número de falsos negativos.
        scenario: Instância de `BusinessScenario` contendo os pesos de custo/benefício.

    Returns:
        Valor líquido calculado como float.
    """
    return (tp * scenario.benefit_tp) - (fp * scenario.cost_fp) - (fn * scenario.cost_fn)


def roi_from_net_value(
        net_value: float,
        total_cost: float,
        on_zero_cost: str = "nan"
) -> float:
    """
    Calcula ROI (retorno sobre investimento) a partir do valor líquido e do custo total.
    Quando total_cost <= 0, o comportamento depende de `on_zero_cost`.

    Args:
        net_value: Valor líquido calculado.
        total_cost: Custo total.
        on_zero_cost: Estratégia quando total_cost <= 0:
            - "nan": retorna NaN.
            - "inf": retorna +inf se net_value>0 e -inf caso contrário.
            - "zero": retorna 0.0.

    Returns:
        ROI como float.
    """
    if total_cost <= 0:
        if on_zero_cost == "inf":
            return float("inf") if net_value > 0 else float("-inf")
        if on_zero_cost == "zero":
            return 0.0
        return float("nan")
    return float(net_value / total_cost)


def load_oof_for_run(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Baixa artifacts OOF (out-of-fold) de um run no MLflow e retorna listas por fold.

    Args:
        client: Instância de `mlflow.tracking.MlflowClient` para listar e baixar artifacts.
        run_id: ID do run no MLflow.

    Returns:
        Uma tupla (y_true_folds, y_proba_folds), onde cada item é uma lista de arrays NumPy,
        um por fold, com os rótulos verdadeiros e as probabilidades/scores.
    """
    y_true_folds: List[np.ndarray] = []
    y_proba_folds: List[np.ndarray] = []

    files = client.list_artifacts(run_id, path="oof")
    paths = [f.path for f in files]

    if "oof/y_true_fold_0.npy" in paths and "oof/y_proba_fold_0.npy" in paths:
        prefix = ""
    else:
        true0_candidates = [p for p in paths if p.endswith("y_true_fold_0.npy")]
        proba0_candidates = [p for p in paths if p.endswith("y_proba_fold_0.npy")]
        if not true0_candidates or not proba0_candidates:
            raise FileNotFoundError(
                "Artifacts OOF não encontrados em 'oof/'. "
                f"Arquivos encontrados em oof/: {paths[:200]}"
            )

        true0 = true0_candidates[0]
        prefix = true0[len("oof/") : -len("y_true_fold_0.npy")]
        expected_proba0 = f"oof/{prefix}y_proba_fold_0.npy"
        if expected_proba0 not in paths:
            raise FileNotFoundError(
                f"Detectei prefixo='{prefix}', mas não achei '{expected_proba0}'. "
                f"Arquivos encontrados em oof/: {paths[:200]}"
            )

    tmpdir = tempfile.mkdtemp(prefix=f"mlflow_oof_{run_id}_")
    for k in range(N_FOLDS):
        true_art = f"oof/{prefix}y_true_fold_{k}.npy"
        prob_art = f"oof/{prefix}y_proba_fold_{k}.npy"

        local_true = client.download_artifacts(run_id, true_art, dst_path=tmpdir)
        local_prob = client.download_artifacts(run_id, prob_art, dst_path=tmpdir)

        y_true = np.load(local_true).astype(int)
        y_proba = np.load(local_prob).astype(float)

        if y_true.shape[0] != y_proba.shape[0]:
            raise ValueError(f"Fold {k}: y_true e y_proba com tamanhos diferentes em run {run_id}")

        y_true_folds.append(y_true)
        y_proba_folds.append(y_proba)

    return y_true_folds, y_proba_folds


def confusion_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thr: float,
) -> Tuple[int, int, int, int]:
    """
    Calcula TN, FP, FN e TP a partir de y_true e y_proba, utilizando um threshold fixo.

    Args:
        y_true: Array de rótulos verdadeiros (0/1).
        y_proba: Array de probabilidades/scores (quanto maior, maior chance de classe positiva).
        thr: Threshold (limiar) para converter probabilidades em predição binária.

    Returns:
        Uma tupla (tn, fp, fn, tp) como inteiros.
    """
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def recall_from_confusion(tp: int, fn: int) -> float:
    """
    Calcula recall (sensibilidade) a partir de TP e FN.

    Args:
        tp: Número de verdadeiros positivos.
        fn: Número de falsos negativos.

    Returns:
        Recall como float. Se TP+FN == 0, retorna NaN.
    """
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def net_value_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thr: float,
    scenario: BusinessScenario,
    roi_on_zero_cost: str = "nan",
) -> Dict[str, float]:
    """
    Avalia valor líquido e ROI para um threshold fixo, com base no cenário de negócio.

    Args:
        y_true: Array de rótulos verdadeiros (0/1).
        y_proba: Array de probabilidades/scores.
        thr: Threshold para classificação.
        scenario: Instância de `BusinessScenario`.
        roi_on_zero_cost: Estratégia quando total_cost <= 0.

    Returns:
        Dicionário:
            - "tn", "fp", "fn", "tp": contagens como float
            - "net_value": valor líquido
            - "roi": retorno sobre custo
    """
    tn, fp, fn, tp = confusion_at_threshold(y_true, y_proba, thr)
    net_value = net_value_from_confusion(tp, fp, fn, scenario)
    total_cost = fp * scenario.cost_fp
    roi = roi_from_net_value(net_value, total_cost, on_zero_cost=roi_on_zero_cost)
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "net_value": float(net_value),
        "roi": float(roi),
    }


def net_value_at_topk(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    topk: float,
    scenario: BusinessScenario,
    roi_on_zero_cost: str = "nan",
) -> Dict[str, float]:
    """
    Avalia valor líquido e ROI no top 4 K modelos(top-k).
    Nesse modo, apenas os `k` modelos com maiores scores são tratados.

    Args:
        y_true: Array de rótulos verdadeiros (0/1).
        y_proba: Array de probabilidades/scores.
        topk: Fração da base a tratar (0 < topk <= 1).
        scenario: Instância de `BusinessScenario`.
        roi_on_zero_cost: Estratégia quando total_cost <= 0.

    Returns:
        Dicionário com chaves:
            - "k": número absoluto tratado
            - "topk": fração tratada
            - "tn", "fp", "fn", "tp": contagens como float
            - "net_value": valor líquido
            - "roi": retorno sobre custo
    """
    if not (0 < topk <= 1.0):
        raise ValueError("topk deve estar em (0, 1].")

    n = len(y_true)
    k = max(int(np.ceil(n * topk)), 1)

    idx = np.argsort(-y_proba)[:k]
    treated = np.zeros(n, dtype=int)
    treated[idx] = 1

    tp = int(((y_true == 1) & (treated == 1)).sum())
    fp = int(((y_true == 0) & (treated == 1)).sum())
    fn = int(((y_true == 1) & (treated == 0)).sum())
    tn = int(((y_true == 0) & (treated == 0)).sum())

    net_value = (tp * scenario.benefit_tp) - (fp * scenario.cost_fp)
    total_cost = fp * scenario.cost_fp
    roi = roi_from_net_value(net_value, total_cost, on_zero_cost=roi_on_zero_cost)

    return {
        "k": float(k),
        "topk": float(topk),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "net_value": float(net_value),
        "roi": float(roi),
    }


def summarize_folds(values: List[float]) -> Dict[str, float]:
    """
    Resume uma lista de valores por fold em estatísticas descritivas: média, desvio padrão, mínimo e máximo.
    Valores não finitos (NaN/inf) são removidos antes do cálculo.

    Args:
        values: Lista de valores (um por fold).

    Returns:
        Dicionário com chaves:
            - "mean": média
            - "std": desvio padrão amostral
            - "min": mínimo
            - "max": máximo
        Se não houver valores finitos, retorna tudo como NaN.
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def mode_label(meta: Dict[str, Any]) -> str:
    """
    Gera um identificador textual do modo de avaliação, utilizado para namespacing
    de métricas no MLflow.

    Args:
        meta: Dicionário com metadados do modo de avaliação. Espera a chave "mode" e,
            dependendo do modo, chaves adicionais como "threshold" ou "topk".

    Returns:
        String com o rótulo do modo, por exemplo:
            - "threshold_thr_0.50"
            - "topk_0.1000"
            - "sweep_foldwise"
    """
    if meta["mode"] == "threshold":
        return f"threshold_thr_{meta['threshold']:.2f}"
    if meta["mode"] == "topk":
        return f"topk_{meta['topk']:.4f}"
    return "sweep_foldwise"


def get_run_metric_value(run: Any, metric_name: str) -> Optional[float]:
    """
    Lê o valor de uma métrica agregada armazenada no próprio run do MLflow.

    Args:
        run: Objeto `mlflow.entities.Run`.
        metric_name: Nome da métrica a ser lida de `run.data.metrics`.

    Returns:
        O valor da métrica como float se existir; caso contrário, None.
    """
    v = run.data.metrics.get(metric_name)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    args = parse_args()

    scenario = BusinessScenario(benefit_tp=args.benefit_tp, cost_fp=args.cost_fp, cost_fn=args.cost_fn)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    runs = get_latest_runs_with_mlp_from_refit(client)

    if args.gate_best_cv_score and args.gate_best_cv_score > 0:
        kept = []
        table = Table(title=f"Gate: {args.gate_metric_name} >= {args.gate_best_cv_score}")
        table.add_column("model_name", style="cyan")
        table.add_column(args.gate_metric_name, justify="right")
        table.add_column("pass", justify="center")

        for r in runs:
            name = run_display_name(r)
            v = get_run_metric_value(r, args.gate_metric_name)
            ok = (v is not None) and (v >= args.gate_best_cv_score)
            table.add_row(name, f"{v:.5f}" if v is not None else "NA", "✓" if ok else "✗")
            if ok:
                kept.append(r)

        console.print(table)
        runs = kept

        if len(runs) < 1:
            console.print("[red]Nenhum run passou o gate.[/red]")
            return


    if len(runs) < 2:
        console.print("[red]Precisa de pelo menos 2 runs candidatos (com artifacts OOF).[/red]")
        return

    console.rule("[bold]Business Evaluation — Net Value / ROI (OOF por fold)[/bold]")
    console.print(
        f"[dim]Folds={N_FOLDS} | scenario: benefit_tp={scenario.benefit_tp}, cost_fp={scenario.cost_fp}, cost_fn={scenario.cost_fn}[/dim]"
    )

    results = []

    for r in runs:
        name = run_display_name(r)
        ok = has_any_oof(client, r.info.run_id)
        console.print(name, r.info.run_id, ok)

        if not has_any_oof(client, r.info.run_id):
            console.print(f"[yellow]Pulando {name} (run_id={r.info.run_id}): sem artifacts OOF em 'oof/'.[/yellow]")
            continue

        y_true_folds, y_proba_folds = load_oof_for_run(client, r.info.run_id)

        fold_net_values: List[float] = []
        fold_rois: List[float] = []
        fold_recalls: List[float] = []
        fold_thresholds: List[float] = []

        if args.sweep_thresholds is not None:
            start, end, step = args.sweep_thresholds
            thresholds = np.arange(start, end + 1e-12, step)

            spec = CostSpec(
                benefit_tp=scenario.benefit_tp,
                cost_fp=scenario.cost_fp,
                cost_fn=scenario.cost_fn,
            )

            fold_thresholds = []
            fold_net_values = []
            fold_rois = []
            fold_recalls = []

            for y_true, y_proba in zip(y_true_folds, y_proba_folds):
                df_thr, best = sweep_thresholds_cost(
                    y_true=y_true,
                    y_score=y_proba,
                    spec=spec,
                    thresholds=thresholds,
                    objective="net_value",
                )
                if not best:
                    fold_thresholds.append(float("nan"))
                    fold_net_values.append(float("nan"))
                    fold_rois.append(float("nan"))
                    fold_recalls.append(float("nan"))
                    continue

                thr = float(best["threshold"])
                fold_thresholds.append(thr)

                net_value = float(best["net_value"])

                fp = int(best["fp"])
                tp = int(best["tp"])
                fn = int(best["fn"])
                fold_recalls.append(recall_from_confusion(tp, fn))

                total_cost = fp * scenario.cost_fp
                roi = roi_from_net_value(net_value, total_cost, on_zero_cost=args.roi_on_zero_cost)

                fold_net_values.append(net_value)
                fold_rois.append(float(roi))

            thr_sum = summarize_folds([t for t in fold_thresholds if np.isfinite(t)])
            meta = {
                "mode": "sweep_thresholds",
                "best_threshold": float(thr_sum["mean"]) if np.isfinite(thr_sum["mean"]) else float("nan"),
                "best_threshold_std": float(thr_sum["std"]) if np.isfinite(thr_sum["std"]) else float("nan"),
            }

        elif args.topk and args.topk > 0:
            for y_true, y_proba in zip(y_true_folds, y_proba_folds):
                m = net_value_at_topk(y_true, y_proba, float(args.topk), scenario, args.roi_on_zero_cost)
                fold_net_values.append(m["net_value"])
                fold_rois.append(m["roi"])
                fold_recalls.append(recall_from_confusion(m["tp"], m["fn"]))
            meta = {"mode": "topk", "topk": float(args.topk)}

        else:
            thr = float(args.threshold)
            for y_true, y_proba in zip(y_true_folds, y_proba_folds):
                m = net_value_at_threshold(y_true, y_proba, thr, scenario, args.roi_on_zero_cost)
                fold_net_values.append(m["net_value"])
                fold_rois.append(m["roi"])
                fold_recalls.append(recall_from_confusion(m["tp"], m["fn"]))
            meta = {"mode": "threshold", "threshold": thr}

        nv_sum = summarize_folds(fold_net_values)
        roi_sum = summarize_folds(fold_rois)
        rec_sum = summarize_folds(fold_recalls)

        results.append({
            "run": name,
            "run_id": r.info.run_id,
            "net_value": nv_sum,
            "roi": roi_sum,
            "recall": rec_sum,
            **meta
        })

        if args.log_mlflow:
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            tag = f".{args.business_tag}" if args.business_tag else ""

            mode_id = mode_label(meta)
            if meta["mode"] == "sweep_thresholds":
                mode_id = "sweep_foldwise"

            parent_nv_name = f"business{tag}.{mode_id}.net_value"
            parent_roi_name = f"business{tag}.{mode_id}.roi"
            parent_recall_name = f"business{tag}.{mode_id}.recall"

            with mlflow.start_run(run_id=r.info.run_id):
                for fold_idx, (nv, roi, rec) in enumerate(zip(fold_net_values, fold_rois, fold_recalls)):
                    mlflow.log_metric(parent_nv_name, float(nv), step=fold_idx)
                    mlflow.log_metric(parent_roi_name, float(roi), step=fold_idx)
                    mlflow.log_metric(parent_recall_name, float(rec), step=fold_idx)

                mlflow.log_metric(f"{parent_nv_name}_mean", nv_sum["mean"])
                mlflow.log_metric(f"{parent_nv_name}_std", nv_sum["std"])
                mlflow.log_metric(f"{parent_roi_name}_mean", roi_sum["mean"])
                mlflow.log_metric(f"{parent_roi_name}_std", roi_sum["std"])
                mlflow.log_metric(f"{parent_recall_name}_mean", rec_sum["mean"])
                mlflow.log_metric(f"{parent_recall_name}_std", rec_sum["std"])

                if meta["mode"] == "sweep_thresholds":
                    mlflow.log_metric(
                        f"business{tag}.sweep_foldwise.best_threshold_mean",
                        float(meta["best_threshold"])
                    )
                    mlflow.log_metric(
                        f"business{tag}.sweep_foldwise.best_threshold_std",
                        float(meta.get("best_threshold_std", float("nan")))
                    )  

                with mlflow.start_run(run_name=f"business_eval_{mode_id}", nested=True):
                    mlflow.log_param("business_mode", meta["mode"])
                    mlflow.log_param("roi_on_zero_cost", args.roi_on_zero_cost)
                    mlflow.log_param("scenario_benefit_tp", scenario.benefit_tp)
                    mlflow.log_param("scenario_cost_fp", scenario.cost_fp)
                    mlflow.log_param("scenario_cost_fn", scenario.cost_fn)
                    if meta["mode"] == "threshold":
                        mlflow.log_param("threshold", meta["threshold"])
                    elif meta["mode"] == "topk":
                        mlflow.log_param("topk", meta["topk"])
                    else:
                        mlflow.log_param("best_threshold", meta["best_threshold"])

                    for fold_idx, (nv, roi) in enumerate(zip(fold_net_values, fold_rois)):
                        mlflow.log_metric("net_value", float(nv), step=fold_idx)
                        mlflow.log_metric("roi", float(roi), step=fold_idx)

                    mlflow.log_metric("net_value_mean", nv_sum["mean"])
                    mlflow.log_metric("net_value_std", nv_sum["std"])
                    mlflow.log_metric("roi_mean", roi_sum["mean"])
                    mlflow.log_metric("roi_std", roi_sum["std"])
                    mlflow.log_metric("recall_mean", rec_sum["mean"])
                    mlflow.log_metric("recall_std", rec_sum["std"])

    if not results:
        console.print("[red]Nenhum run com artifacts OOF encontrados.[/red]")
        return

    rank_key = args.metric
    results_sorted = sorted(results, key=lambda x: x[rank_key]["mean"], reverse=True)

    table = Table(title="Ranking (business)")
    table.add_column("Run", style="cyan")
    table.add_column("mode", justify="left")
    table.add_column("net_value_mean", justify="right")
    table.add_column("net_value_std", justify="right")
    table.add_column("roi_mean", justify="right")
    table.add_column("roi_std", justify="right")
    table.add_column("meta", justify="left")

    for r in results_sorted:
        if r["mode"] == "threshold":
            meta_str = f"thr={r['threshold']:.2f}"
        elif r["mode"] == "topk":
            meta_str = f"topk={r['topk']:.2%}"
        else:
            meta_str = f"best_thr={r['best_threshold']:.2f}"

        table.add_row(
            r["run"],
            r["mode"],
            f"{r['net_value']['mean']:.2f}",
            f"{r['net_value']['std']:.2f}",
            f"{r['roi']['mean']:.2f}",
            f"{r['roi']['std']:.2f}",
            f"{r['recall']['mean']:.2f}",
            f"{r['recall']['std']:.2f}",
            meta_str,
        )

    console.print(table)

    best = results_sorted[0]
    console.print(
        f"\n[bold green]Best (by {args.metric}):[/bold green] {best['run']} "
        f"{args.metric}={best[args.metric]['mean']:.2f} ± {best[args.metric]['std']:.2f}"
    )

    console.print(
        "\n[dim]Para testar estatisticamente no compare_models.py, a métrica no PARENT run agora tem namespace.\n"
        "Exemplo (topk=0.10): business.topk_0.1000.net_value\n"
        "Então rode:\n"
        "  python experiments/selection/compare_models.py --metric business.topk_0.1000.net_value\n"
        "Ou para sweep (best_thr=0.05):\n"
        "  python experiments/selection/compare_models.py --metric business.sweep_foldwise.net_value\n"
        "[/dim]"
    )


if __name__ == "__main__":
    main()
