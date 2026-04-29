"""
Uso:
    # comparação normal, utilizando como baleline a regressão logística:
    python experiments/selection/compare_models.py

    # comparação final, levando em consideração o valor líquido (net_value) dos modelos com recall de treino >= 0.8:
    python experiments/selection/compare_models.py \
    --metric business.sweep_foldwise.net_value \
    --gate-best-cv-score 0.8 \
    --gate-metric-name best_cv_score
"""
import argparse
import logging
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

import mlflow
import numpy as np
import scikit_posthocs as sp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats

from src.utils.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    PRIMARY_METRIC,
)

logger = logging.getLogger(__name__)
console = Console()
ALPHA = 0.05


# -----------------------------
# MLflow helpers
# -----------------------------
def is_cv_run(run) -> bool:
    p = run.data.params
    return (p.get("cv_folds") == str(N_FOLDS) or p.get("n_folds") == str(N_FOLDS))

def is_mlp_refit_run(run) -> bool:
    p = run.data.params
    return p.get("model_name") == "mlp" and ("best_config_name" in p)

def get_latest_mlp_refit_run(client) -> object | None:
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    # filtro por params no MLflow search_runs é meio chato (params.*),
    # então fazemos busca ampla e filtramos em Python.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    for r in runs:
        if is_mlp_refit_run(r):
            return r
    return None

def find_cv_run_by_config_name(client, config_name: str) -> object | None:
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    for r in runs:
        p = r.data.params
        # 1) melhor: param explícito
        if p.get("config_name") == config_name and is_cv_run(r):
            return r
        # 2) fallback: runName
        if r.data.tags.get("mlflow.runName") == config_name and is_cv_run(r):
            return r
    return None


def get_latest_runs_with_mlp_from_refit(client: mlflow.tracking.MlflowClient) -> list:
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    chosen = {}

    # 1) primeiro pega o MLP via refit -> best_config_name -> run de CV dessa config
    mlp_refit = get_latest_mlp_refit_run(client)
    if mlp_refit is not None:
        best_config_name = mlp_refit.data.params.get("best_config_name")
        if best_config_name:
            mlp_cv_run = find_cv_run_by_config_name(client, best_config_name)
            if mlp_cv_run is not None:
                chosen["mlp"] = mlp_cv_run
            else:
                console.print(
                    f"[yellow]Aviso:[/yellow] não achei run de CV para best_config_name='{best_config_name}'. "
                    "Vou cair no fallback do último CV do mlp."
                )

    # 2) para os demais modelos (e fallback do mlp se necessário): último run CV por model_name
    for r in all_runs:
        p = r.data.params
        model_name = p.get("model_name")
        if not model_name:
            continue
        if not is_cv_run(r):
            continue
        if model_name not in chosen:
            chosen[model_name] = r

    return list(chosen.values())



def run_group_key(run) -> str:
    p = run.data.params
    return (
        p.get("model_name")
        or p.get("config_name")
        or run.data.tags.get("mlflow.runName")
        or run.info.run_id
    )

def get_latest_runs_per_group(client: mlflow.tracking.MlflowClient) -> List:
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experimento '{MLFLOW_EXPERIMENT_NAME}' não encontrado.")

    # Busca bastante coisa; você pode adicionar filter_string se tiver padrões
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    latest_by_key = {}
    for r in runs:
        p = r.data.params

        # mantém o filtro que você já tinha (só runs "de modelo" e do mesmo cv)
        is_model = (p.get("model_name") or p.get("config_name"))
        same_cv = (p.get("cv_folds") == str(N_FOLDS) or p.get("n_folds") == str(N_FOLDS))
        if not (is_model and same_cv):
            continue

        key = run_group_key(r)

        # como runs já vêm em ordem start_time DESC, o primeiro que aparecer é o mais recente
        if key not in latest_by_key:
            latest_by_key[key] = r

    return list(latest_by_key.values())


def run_display_name(run) -> str:
    return run.data.params.get("model_name") or run.info.run_id

def load_fold_metric(client, run_id: str, metric: str) -> np.ndarray:
    hist = sorted(client.get_metric_history(run_id, metric), key=lambda m: (m.step, m.timestamp))
    by_step = {}
    for m in hist:
        if m.step is None:
            continue
        if 0 <= m.step < N_FOLDS:
            by_step[m.step] = m.value
    if len(by_step) != N_FOLDS:
        return np.array([], dtype=float)
    return np.array([by_step[i] for i in range(N_FOLDS)], dtype=float)


# -----------------------------
# Estatística: 2 modelos
# -----------------------------
def paired_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    diff = scores_a - scores_b
    _, p_sw = stats.shapiro(diff) if len(diff) >= 3 else (np.nan, 1.0)
    is_normal = (p_sw >= ALPHA) if not np.isnan(p_sw) else False

    if is_normal:
        stat, p_val = stats.ttest_rel(scores_a, scores_b)
        test_name = "Paired t-test"
    else:
        if np.allclose(diff, 0):
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = stats.wilcoxon(scores_a, scores_b)
        test_name = "Wilcoxon"

    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    se = stats.sem(diff) if len(diff) > 1 else 0.0
    if len(diff) > 1 and se > 0:
        ci = stats.t.interval(0.95, df=len(diff) - 1, loc=np.mean(diff), scale=se)
        ci = (float(ci[0]), float(ci[1]))
    else:
        ci = (float(np.mean(diff)), float(np.mean(diff)))

    return {
        "test": test_name,
        "p_value": float(p_val),
        "stat": float(stat),
        "shapiro_p": float(p_sw) if not np.isnan(p_sw) else None,
        "mean_diff": float(np.mean(diff)),
        "cohens_d": float(d),
        "ci95": ci,
        "significant": bool(p_val < ALPHA),
    }


# -----------------------------
# Estatística: 3+ modelos
# -----------------------------
def print_nemenyi_matrix(pvals, names: List[str], alpha: float = ALPHA) -> None:
    table = Table(title="Post-hoc Nemenyi (p-valores)")
    table.add_column("", style="bold")
    for n in names:
        table.add_column(n[:22], justify="right")

    for i, ni in enumerate(names):
        row = [ni[:22]]
        for j, nj in enumerate(names):
            if i == j:
                row.append("[dim]—[/dim]")
            else:
                p = float(pvals.loc[ni, nj])
                if p < alpha:
                    row.append(f"[green]{p:.4f}[/green]")
                else:
                    row.append(f"{p:.4f}")
        table.add_row(*row)

    console.print(table)


def friedman_nemenyi(all_scores: Dict[str, np.ndarray], model_names: List[str]) -> Dict:
    arrays = [all_scores[n] for n in model_names]
    stat, p = stats.friedmanchisquare(*arrays)
    result = {"friedman_stat": float(stat), "friedman_p": float(p), "significant": bool(p < ALPHA)}
    if p < ALPHA:
        data = np.column_stack(arrays)
        nemenyi = sp.posthoc_nemenyi_friedman(data)
        nemenyi.index = model_names
        nemenyi.columns = model_names
        result["nemenyi_pvals"] = nemenyi
        print_nemenyi_matrix(nemenyi, model_names, alpha=ALPHA)
    return result


def print_duel_panel(a: str, b: str, metric: str, mean_a: float, mean_b: float, res: Dict) -> None:
    sig_style = "bold green" if res["significant"] else "dim"
    sig_mark = "✓ Sig." if res["significant"] else "✗ N.S."

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Comparação", f"{a} vs {b}")
    table.add_row("Médias", f"{mean_a:.6f} vs {mean_b:.6f} (Δ={res['mean_diff']:+.6f})")
    if res["shapiro_p"] is not None:
        table.add_row("Normalidade", f"Shapiro p={res['shapiro_p']:.4f}")
    table.add_row("Teste", f"{res['test']}: stat={res['stat']:.4f}, p={res['p_value']:.6f}")
    table.add_row("Cohen's d", f"{res['cohens_d']:.4f}")
    table.add_row("IC 95% (Δ)", f"[{res['ci95'][0]:+.6f}, {res['ci95'][1]:+.6f}]")
    table.add_row("Resultado", f"[{sig_style}]{sig_mark}[/{sig_style}]")
    console.print(Panel(table, title=f"[bold]{metric} — duelo[/bold]"))


# -----------------------------
# Gate logic
# -----------------------------
def apply_gate_filter(
    scores: Dict[str, np.ndarray],
    baseline_name: str,
    gate_rel: float,
) -> Dict[str, np.ndarray]:
    if baseline_name not in scores:
        raise ValueError(f"Baseline '{baseline_name}' não encontrado entre os runs carregados.")

    baseline_mean = float(np.mean(scores[baseline_name]))
    gate = baseline_mean * (1.0 + gate_rel)

    console.rule("[bold]Gate técnico (elegibilidade)[/bold]")
    console.print(f"[dim]Baseline: {baseline_name} | mean={baseline_mean:.6f}[/dim]")
    console.print(f"[dim]Gate: mean >= {gate:.6f} (rel +{gate_rel*100:.2f}%) [/dim]\n")

    eligible = {k: v for k, v in scores.items() if float(np.mean(v)) >= gate}

    table = Table(title="Elegíveis pelo gate")
    table.add_column("model", style="cyan")
    table.add_column("mean_metric", justify="right")
    table.add_column("pass", justify="center")
    for name, v in sorted(scores.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True):
        mean_v = float(np.mean(v))
        ok = "✓" if mean_v >= gate else "✗"
        style = "bold green" if ok == "✓" else "dim"
        table.add_row(name, f"{mean_v:.6f}", f"[{style}]{ok}[/{style}]")
    console.print(table)

    console.print(f"\n[bold]Total elegíveis:[/bold] {len(eligible)} de {len(scores)}")
    return eligible


def log_summary_table_to_mlflow(scores: Dict[str, np.ndarray], metric: str) -> Path:
    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)
    rows = []
    for name, v in scores.items():
        rows.append({
            "model_name": name,
            f"mean_{metric}": float(np.mean(v)),
            f"std_{metric}": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(by=f"mean_{metric}", ascending=False)
    p = tmp_dir / f"selection_summary_{metric}.csv"
    df.to_csv(p, index=False)
    mlflow.log_artifact(str(p), artifact_path="selection")
    return p


def log_pairwise_results_to_mlflow(pairwise: list[dict], metric: str) -> Path:
    tmp_dir = Path("mlflow_artifacts_tmp")
    tmp_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(pairwise)
    p = tmp_dir / f"pairwise_tests_{metric}.csv"
    df.to_csv(p, index=False)
    mlflow.log_artifact(str(p), artifact_path="selection/tests")
    return p


def log_friedman_to_mlflow(fr: dict, names: list[str], metric: str) -> None:
    mlflow.log_metric("friedman_stat", float(fr["friedman_stat"]))
    mlflow.log_metric("friedman_p", float(fr["friedman_p"]))
    mlflow.log_param("friedman_significant", str(bool(fr["significant"])))
    mlflow.log_param("friedman_n_models", len(names))
    mlflow.log_param("friedman_metric", metric)

    # se tiver Nemenyi, loga matriz como artifact
    if "nemenyi_pvals" in fr:
        tmp_dir = Path("mlflow_artifacts_tmp")
        tmp_dir.mkdir(exist_ok=True)
        p = tmp_dir / f"nemenyi_pvals_{metric}.csv"
        fr["nemenyi_pvals"].to_csv(p, index=True)
        mlflow.log_artifact(str(p), artifact_path="selection/tests")


def register_winner(client, run, model_name, registry_name, alias, tags):
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, registry_name)
    client.set_registered_model_alias(registry_name, alias, mv.version)
    for key, value in tags.items():
        client.set_model_version_tag(registry_name, mv.version, key, str(value))
    console.print(f"  [green]✓[/green] {model_name} → {registry_name} v{mv.version} (alias: {alias})")
    return mv


def wilcoxon_vs_baseline(scores_model: np.ndarray, scores_baseline: np.ndarray) -> Dict:
    diff = scores_model - scores_baseline
    if np.allclose(diff, 0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = stats.wilcoxon(scores_model, scores_baseline)
    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0
    return {
        "test": "Wilcoxon",
        "p_value": float(p_val),
        "stat": float(stat),
        "mean_diff": float(np.mean(diff)),
        "cohens_d": float(d),
        "significant": bool(p_val < ALPHA),
    }


def decide_winner(
    scores: Dict[str, np.ndarray],
    metric: str,
    baseline: str,
    decision: str,
) -> Dict:
    names = list(scores.keys())
    ranked = sorted(names, key=lambda n: float(np.mean(scores[n])), reverse=True)

    if decision == "top2_duel":
        if len(ranked) < 2:
            raise ValueError("Precisa de pelo menos 2 modelos para top2_duel.")
        a, b = ranked[0], ranked[1]
        res = wilcoxon_vs_baseline(scores[a], scores[b])  # a - b
        # se a não for significativamente melhor, escolhe b (parcimônia: o segundo)
        if res["significant"] and res["mean_diff"] > 0:
            winner, loser = a, b
            reason = "top1_significantly_better_than_top2"
        else:
            winner, loser = b, a
            reason = "no_sig_diff_choose_top2"
        return {
            "winner": winner,
            "loser": loser,
            "baseline": None,
            "challenger": None,
            "duel_a": a,
            "duel_b": b,
            "metric": metric,
            "decision": decision,
            "result": res,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
        }

    # baseline_duel
    if baseline not in scores:
        # baseline foi filtrado (ex.: gate best_cv_score). Faz fallback.
        ranked = sorted(scores.keys(), key=lambda n: float(np.mean(scores[n])), reverse=True)
        if len(ranked) < 2:
            return {
                "winner": ranked[0],
                "loser": None,
                "baseline": None,
                "challenger": None,
                "duel_a": None,
                "duel_b": None,
                "metric": metric,
                "decision": "top2_duel_fallback_no_baseline",
                "result": None,
                "ranked": ranked,
                "means": {n: float(np.mean(scores[n])) for n in ranked},
                "reason": "baseline_not_eligible",
            }
        a, b = ranked[0], ranked[1]
        res = wilcoxon_vs_baseline(scores[a], scores[b])
        if res["significant"] and res["mean_diff"] > 0:
            winner, loser = a, b
            reason = "top1_significantly_better_than_top2"
        else:
            winner, loser = b, a
            reason = "no_sig_diff_choose_top2"
        return {
            "winner": winner,
            "loser": loser,
            "baseline": None,
            "challenger": None,
            "duel_a": a,
            "duel_b": b,
            "metric": metric,
            "decision": "top2_duel_fallback_no_baseline",
            "result": res,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
            "reason": reason,
        }

    # desafiante = melhor por média que NÃO seja o baseline
    challenger = ranked[0] if ranked[0] != baseline else (ranked[1] if len(ranked) > 1 else baseline)
    if challenger == baseline:
        # só existe baseline
        return {
            "winner": baseline,
            "loser": None,
            "baseline": baseline,
            "challenger": None,
            "duel_a": None,
            "duel_b": None,
            "metric": metric,
            "decision": decision,
            "result": None,
            "ranked": ranked,
            "means": {n: float(np.mean(scores[n])) for n in ranked},
        }

    res = wilcoxon_vs_baseline(scores[challenger], scores[baseline])  # challenger - baseline

    if res["significant"] and res["mean_diff"] > 0:
        winner = challenger
        loser = baseline
        reason = "challenger_significantly_better_than_baseline"
    else:
        winner = baseline
        loser = challenger
        reason = "no_sig_gain_over_baseline_choose_baseline"

    return {
        "winner": winner,
        "loser": loser,
        "baseline": baseline,
        "challenger": challenger,
        "duel_a": challenger,
        "duel_b": baseline,
        "metric": metric,
        "decision": decision,
        "result": res,
        "ranked": ranked,
        "means": {n: float(np.mean(scores[n])) for n in ranked},
        "reason": reason,
    }


def get_gate_score_recall(client, run, n_folds: int) -> float | None:
    v = run.data.metrics.get("best_cv_score")
    if v is not None:
        return float(v)
    v = run.data.metrics.get("cv_mean_recall")
    if v is not None:
        return float(v)
    # fallback: média do recall por fold (treino)
    hist = client.get_metric_history(run.info.run_id, "recall")
    hist = sorted(hist, key=lambda m: (m.step, m.timestamp))
    by_step = {}
    for m in hist:
        if m.step is None:
            continue
        if 0 <= m.step < n_folds:
            by_step[m.step] = m.value
    if len(by_step) == n_folds:
        return float(np.mean([by_step[i] for i in range(n_folds)]))
    return None


# -----------------------------
# MAIN
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Comparação estatística de modelos (Friedman diagnóstico + duelo final)"
    )
    parser.add_argument("--metric", default=PRIMARY_METRIC, help="Métrica para comparar (ex.: average_precision)")
    parser.add_argument(
        "--decision",
        choices=["baseline_duel", "top2_duel"],
        default="baseline_duel",
        help="Regra de decisão: baseline_duel (top vs baseline) ou top2_duel (top1 vs top2).",
    )
    parser.add_argument("--baseline-run", default="logreg", help="Baseline (model_name), usado em baseline_duel")

    # gate técnico (média vs baseline) - já existia
    parser.add_argument("--apply-gate", action="store_true", help="Aplicar gate técnico vs baseline (por média)")
    parser.add_argument("--gate-rel", type=float, default=0.02, help="Ganho relativo mínimo vs baseline (média)")

    # gate por best_cv_score (recall de treino do search) - NOVO
    parser.add_argument(
        "--gate-best-cv-score",
        type=float,
        default=0.0,
        help="Se >0, filtra runs cujo score de gate (best_cv_score/cv_mean_recall/recall por fold) >= este valor.",
    )
    parser.add_argument(
        "--gate-metric-name",
        default="best_cv_score",
        help="Nome da métrica para gate. Se 'best_cv_score', usa fallback inteligente (inclui MLP).",
    )

    parser.add_argument("--register", action="store_true", help="Registrar campeão no MLflow Model Registry")
    parser.add_argument("--registry-name", default="telco-churn-model", help="Nome do Registered Model")

    parser.add_argument("--models", nargs="+", default=None, help="Lista de model_name para comparar (opcional)")

    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    metric = args.metric

    with mlflow.start_run(run_name=f"model_selection__{metric}") as sel_run:
        mlflow.set_tag("run_stage", "selection")
        mlflow.log_param("metric", metric)
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("decision_rule", args.decision)
        mlflow.log_param("baseline_run", args.baseline_run)
        mlflow.log_param("apply_gate", bool(args.apply_gate))
        mlflow.log_param("gate_rel", float(args.gate_rel))
        mlflow.log_param("gate_best_cv_score", float(args.gate_best_cv_score))
        mlflow.log_param("gate_metric_name", args.gate_metric_name)
        mlflow.log_param("register", bool(args.register))
        mlflow.log_param("registry_name", args.registry_name)
        mlflow.log_param("models_filter", " ".join(args.models) if args.models else "")

        # 1) Seleciona runs (últimos por model + regra especial MLP)
        runs = get_latest_runs_with_mlp_from_refit(client)
        if len(runs) < 2:
            console.print("[red]Precisa de pelo menos 2 runs.[/red]")
            return

        # 2) Gate por best_cv_score (ou métrica definida)
        if args.gate_best_cv_score and args.gate_best_cv_score > 0:
            kept = []
            gate_table = Table(title=f"Gate: {args.gate_metric_name} >= {args.gate_best_cv_score}")
            gate_table.add_column("model_name", style="cyan")
            gate_table.add_column("gate_score", justify="right")
            gate_table.add_column("pass", justify="center")

            for r in runs:
                name = run_display_name(r)

                if args.gate_metric_name == "best_cv_score":
                    score = get_gate_score_recall(client, r, N_FOLDS)
                else:
                    score = r.data.metrics.get(args.gate_metric_name)
                    score = float(score) if score is not None else None

                ok = (score is not None) and (float(score) >= args.gate_best_cv_score)
                gate_table.add_row(name, f"{score:.5f}" if score is not None else "NA", "✓" if ok else "✗")

                if ok:
                    kept.append(r)

            console.print(gate_table)
            runs = kept

            if len(runs) < 2:
                console.print("[red]Gate deixou menos de 2 modelos para comparar.[/red]")
                mlflow.set_tag("status", "gate_left_less_than_2")
                return

        # 3) Carregar métrica por fold para cada run
        tech_scores: Dict[str, np.ndarray] = {}
        run_map: Dict[str, Any] = {}  # model_name -> run

        for r in runs:
            name = run_display_name(r)  # model_name
            s = load_fold_metric(client, r.info.run_id, metric)
            if len(s) == N_FOLDS:
                tech_scores[name] = s
                run_map[name] = r

        names = sorted(tech_scores.keys())
        if len(names) < 2:
            console.print(f"[red]Nenhum run tem '{metric}' logado por fold.[/red]")
            mlflow.set_tag("status", "missing_fold_metrics")
            return

        # 4) Filtrar por lista explícita de modelos (opcional)
        if args.models:
            wanted = set(args.models)
            tech_scores = {k: v for k, v in tech_scores.items() if k in wanted}
            run_map = {k: v for k, v in run_map.items() if k in wanted}
            names = sorted(tech_scores.keys())

            if len(names) < 2:
                console.print("[red]Precisa de pelo menos 2 modelos após o filtro --models.[/red]")
                console.print(f"[dim]Encontrados: {names} | Solicitados: {sorted(wanted)}[/dim]")
                mlflow.set_tag("status", "models_filter_left_less_than_2")
                return

        # loga quais runs efetivamente entraram
        mlflow.log_dict({n: run_map[n].info.run_id for n in names}, "selection/compared_runs.json")

        # 5) Gate técnico vs baseline (média)
        if args.apply_gate:
            tech_scores = apply_gate_filter(tech_scores, args.baseline_run, args.gate_rel)
            names = sorted(tech_scores.keys())
            run_map = {k: v for k, v in run_map.items() if k in set(names)}
            mlflow.log_param("n_models_after_gate", len(names))

            if len(names) < 2:
                console.print("[red]Gate deixou menos de 2 modelos elegíveis.[/red]")
                mlflow.set_tag("status", "apply_gate_left_less_than_2")
                if len(names) == 1:
                    mlflow.set_tag("winner", names[0])
                return

        # 6) Resumo
        console.rule(f"[bold]Comparação — métrica={metric} | folds={N_FOLDS} | α={ALPHA}[/bold]")
        t = Table(title="Resumo técnico (por fold)")
        t.add_column("model_name", style="cyan")
        t.add_column(f"mean_{metric}", justify="right")
        t.add_column(f"std_{metric}", justify="right")
        for n in sorted(names, key=lambda k: float(np.mean(tech_scores[k])), reverse=True):
            v = tech_scores[n]
            t.add_row(n, f"{np.mean(v):.6f}", f"{np.std(v, ddof=1):.6f}")
        console.print(t)

        # 7) Friedman (diagnóstico)
        if len(names) >= 3:
            fr = friedman_nemenyi(tech_scores, names)
            mlflow.log_metric("friedman_stat", float(fr["friedman_stat"]))
            mlflow.log_metric("friedman_p", float(fr["friedman_p"]))
            mlflow.set_tag("friedman_significant", str(bool(fr["significant"])))
            if "nemenyi_pvals" in fr:
                tmp_dir = Path("mlflow_artifacts_tmp")
                tmp_dir.mkdir(exist_ok=True)
                p = tmp_dir / f"nemenyi_pvals_{metric}.csv"
                fr["nemenyi_pvals"].to_csv(p, index=True)
                mlflow.log_artifact(str(p), artifact_path="selection/tests")

        # 8) Decisão final
        decision_out = decide_winner(
            scores=tech_scores,
            metric=metric,
            baseline=args.baseline_run,
            decision=args.decision,
        )

        winner = decision_out["winner"]
        res = decision_out["result"]

        mlflow.set_tag("winner", winner)
        mlflow.set_tag("decision_reason", decision_out.get("reason", ""))
        mlflow.log_dict(decision_out["means"], "selection/means.json")
        mlflow.log_dict({"ranked": decision_out["ranked"]}, "selection/ranking.json")

        if res is not None:
            mlflow.log_param("duel_a", decision_out["duel_a"])
            mlflow.log_param("duel_b", decision_out["duel_b"])
            mlflow.log_metric("duel_p_value", float(res["p_value"]))
            mlflow.log_metric("duel_stat", float(res["stat"]))
            mlflow.log_metric("duel_mean_diff", float(res["mean_diff"]))
            mlflow.log_metric("duel_cohens_d", float(res["cohens_d"]))
            mlflow.set_tag("duel_significant", str(bool(res["significant"])))

        console.print(f"\n[bold green]Winner:[/bold green] {winner}  [dim]({decision_out.get('reason','')})[/dim]")
        if res is not None:
            console.print(
                f"[dim]Duel: {decision_out['duel_a']} vs {decision_out['duel_b']} | "
                f"p={res['p_value']:.6f} Δ={res['mean_diff']:+.6f} d={res['cohens_d']:.3f}[/dim]"
            )

        # 9) Registry (opcional)
        if args.register:
            if winner not in run_map:
                raise RuntimeError(f"Não achei o run do winner='{winner}' para registrar.")

            tags = {
                "selection_metric": metric,
                f"mean_{metric}": f"{float(np.mean(tech_scores[winner])):.6f}",
                "decision_rule": args.decision,
                "decision_reason": decision_out.get("reason", ""),
            }
            if res is not None:
                tags.update({
                    "duel_a": decision_out["duel_a"],
                    "duel_b": decision_out["duel_b"],
                    "duel_test": res["test"],
                    "duel_p_value": f"{res['p_value']:.6f}",
                    "duel_mean_diff": f"{res['mean_diff']:+.6f}",
                    "duel_significant": str(bool(res["significant"])),
                })

            register_winner(
                client=client,
                run=run_map[winner],
                model_name=winner,
                registry_name=args.registry_name,
                alias="champion",
                tags=tags,
            )

if __name__ == "__main__":
    main()
